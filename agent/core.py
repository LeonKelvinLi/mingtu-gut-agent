"""The agent pipeline.

    用户问题
        ↓
    风险分流 (确定性筛查 + 模型分类，只能升级不能降级)
        ├── emergency ──→ 固定升级话术，不经过模型
        ↓
    查询拆解
        ↓
    混合检索 (BM25 + 向量 + 元数据配额)
        ↓
    实时补充 (本地证据不足或问题需要时效信息时走 Perplexity)
        ↓
    重排 (选出 5-8 段)
        ↓
    Claude 作答 (缓存的策略层 + 本轮证据)
        ↓
    引用校验 + 合规校验 ──→ 不合格则带纠正指令重写
        ↓
    回答 + 来源
"""

import re

from agent import safety
from agent.classify import classify, rerank
from agent.live import needs_live, search_live
from agent.llm import MODEL, client
from agent.prompt import build_evidence_block, build_system_prompt
from agent.retrieve import Retriever

MAX_REPAIR_ROUNDS = 2
TOP_N = 8

_CITATION = re.compile(r"\[(\d{1,2})\]")

# Sentences that assert something material about efficacy, dosage, or rules.
# Used to decide whether a citation-free answer is acceptable.
_MATERIAL = re.compile(
    r"有效|无效|有助于|可以改善|能改善|证据|研究|试验|荟萃|"
    r"剂量|CFU|每日|疗程|菌株|规定|法规|不得|允许|建议服用|适用于"
)


class Agent:
    def __init__(self, verbose=False):
        self.retriever = Retriever()
        self.system = build_system_prompt()
        self.verbose = verbose

    # -- retrieval ----------------------------------------------------------
    def gather(self, query, routing):
        filters = {
            "population": routing.get("populations") or [],
            "regulatory": routing.get("regulatory", False),
        }

        pool, seen = [], set()
        for sub in routing["sub_queries"][:3]:
            for chunk in self.retriever.search(sub, filters, top_n=TOP_N):
                if chunk["id"] in seen:
                    continue
                seen.add(chunk["id"])
                pool.append(chunk)

        best = max((c["_score"] for c in pool), default=0.0)
        used_live = False
        if routing.get("needs_current_info") or needs_live(query, best):
            for chunk in search_live(query):
                if chunk["id"] not in seen:
                    seen.add(chunk["id"])
                    pool.append(chunk)
                    used_live = True

        pool.sort(key=lambda c: c["_score"], reverse=True)
        passages = rerank(query, pool[:24], top_n=TOP_N)
        return passages, used_live

    # -- validation ---------------------------------------------------------
    @staticmethod
    def validate_citations(answer, passages):
        """Every cited index must exist, and material claims must cite something."""
        problems = []
        cited = {int(n) for n in _CITATION.findall(answer)}
        valid = set(range(1, len(passages) + 1))

        invented = cited - valid
        if invented:
            problems.append(
                f"引用了不存在的资料编号 {sorted(invented)}，本轮只有 1-{len(passages)} 号"
            )

        if passages and not cited:
            material = [
                s for s in re.split(r"(?<=[。！？\n])", answer)
                if _MATERIAL.search(s) and len(s.strip()) > 12
            ]
            # An honest abstention has no citations and that is correct.
            abstained = re.search(r"没有覆盖|资料不足|证据不足|无法回答|不确定|没有查到", answer)
            if material and not abstained:
                problems.append("回答里有实质性结论但完全没有标注来源编号")
        return problems

    # -- generation ---------------------------------------------------------
    def answer(self, query, history=None):
        routing = classify(query)
        result = {"routing": routing, "passages": [], "used_live": False,
                  "repairs": 0, "answer": "", "sources": []}

        # Emergency short-circuits the model entirely.
        if routing["risk"] == "emergency":
            result["answer"] = safety.escalation_message(routing["red_flags"])
            result["escalated"] = True
            return result
        result["escalated"] = False

        passages, used_live = self.gather(query, routing)
        result["passages"] = passages
        result["used_live"] = used_live

        c = client()
        if c is None:
            result["answer"] = (
                "【未配置 ANTHROPIC_API_KEY，无法生成回答】\n"
                f"本轮检索到 {len(passages)} 条相关资料，检索与安全层工作正常。"
            )
            result["error"] = "no_anthropic_key"
            return result

        messages = list(history or [])
        messages.append({
            "role": "user",
            "content": f"{build_evidence_block(passages)}\n\n## 用户问题\n\n{query}",
        })
        if routing.get("safety_instruction"):
            messages.append({"role": "system", "content": routing["safety_instruction"]})

        answer = ""
        for attempt in range(MAX_REPAIR_ROUNDS + 1):
            answer = self._generate(messages)
            if answer is None:
                result["answer"] = "抱歉，这一轮没能生成回答，请再问一次。"
                result["error"] = "generation_failed"
                return result

            problems = self.validate_citations(answer, passages)
            violations = safety.screen_output(answer)

            if not problems and not violations:
                break
            if attempt == MAX_REPAIR_ROUNDS:
                # Never ship an answer that failed the gates.
                result["answer"] = (
                    "这个问题我没能给出符合规范的回答。可能是我手上的资料不足以支撑结论，"
                    "建议你直接咨询消化科医生或临床营养师。"
                )
                result["error"] = "validation_failed"
                result["problems"] = problems + [v[1] for v in violations]
                return result

            result["repairs"] += 1
            messages.append({"role": "assistant", "content": answer})
            fixes = []
            if problems:
                fixes.append("【引用指令】" + "；".join(problems) + "。请重写，为每个实质性结论标注正确的资料编号。")
            if violations:
                fixes.append(safety.correction_for(violations))
            messages.append({"role": "system", "content": "\n".join(fixes)})

        result["answer"] = answer
        result["sources"] = self._sources(answer, passages)
        return result

    def _generate(self, messages):
        c = client()
        try:
            with c.messages.stream(
                model=MODEL,
                max_tokens=8000,
                system=[{
                    "type": "text",
                    "text": self.system,
                    "cache_control": {"type": "ephemeral"},
                }],
                output_config={"effort": "high"},
                messages=messages,
            ) as stream:
                resp = stream.get_final_message()
            if resp.stop_reason == "refusal":
                return None
            return next((b.text for b in resp.content if b.type == "text"), "")
        except Exception:
            return None

    @staticmethod
    def _sources(answer, passages):
        """Only report sources the answer actually cited."""
        out = []
        for n in sorted({int(x) for x in _CITATION.findall(answer)}):
            if 1 <= n <= len(passages):
                p = passages[n - 1]
                out.append({
                    "n": n,
                    "heading": p.get("heading", ""),
                    "source": p.get("source", ""),
                    "grade": p.get("evidence_grade"),
                    "is_live": bool(p.get("is_live")),
                    "citations": (p.get("citations") or [])[:4],
                })
        return out
