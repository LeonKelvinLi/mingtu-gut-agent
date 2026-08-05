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
from agent.cache import AnswerCache
from agent.classify import classify, rerank
from agent.live import needs_live, search_live
from agent.llm import available, generate, provider_label
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
    def __init__(self, verbose=False, use_cache=True):
        self.retriever = Retriever()
        self.system = build_system_prompt()
        # 可关闭：测试里必须能关掉，否则 Agent 会载入生产缓存，
        # 命中后直接短路，被测的生成与校验逻辑根本不会执行。
        self.cache = AnswerCache() if use_cache else None
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
        degraded = any(p.get("_rerank_fallback") for p in passages)
        return passages, used_live, degraded

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
        # 1) 确定性报警症状筛查永远第一步，且必须在缓存之前。
        #    急症输入既不查缓存也不写缓存——任何缓存都不得短路就医提示。
        flags, instruction = safety.screen_input(query)
        result = {"passages": [], "used_live": False, "repairs": 0,
                  "answer": "", "sources": [], "degraded": [], "cached": False}

        if flags and safety.mentions_self(query):
            result["routing"] = {"risk": "emergency", "red_flags": flags,
                                 "_fallback": False, "populations": [],
                                 "regulatory": False, "sub_queries": [query],
                                 "safety_instruction": instruction}
            result["answer"] = safety.escalation_message(flags)
            result["escalated"] = True
            return result

        # 2) 缓存。只对首轮提问生效：追问依赖上文，同样的字面问题在不同上下文里
        #    答案并不相同，用缓存回答追问会答非所问。
        # 必须写成 is not None：AnswerCache 定义了 __len__，所以 bool(空缓存) 为 False。
        # 写成 bool(self.cache) 的后果是：缓存一旦被清空（例如语料更新导致整体失效），
        # 就再也不会写入任何条目，自己把自己锁死在空状态。
        cacheable = (self.cache is not None) and (not history) and (not flags)
        query_vector = None
        if cacheable:
            hit = self.cache.lookup_literal(query)
            if hit is None:
                try:
                    query_vector = self.retriever._embed_query(query)
                    hit = self.cache.lookup_semantic(query_vector)
                except Exception:
                    hit = None
            if hit:
                result.update({
                    "answer": hit["answer"], "sources": hit.get("sources", []),
                    "cached": True, "escalated": False,
                    "routing": {"risk": "normal", "red_flags": [], "_fallback": False,
                                "populations": [], "regulatory": False,
                                "sub_queries": [query], "safety_instruction": None},
                })
                return result

        routing = classify(query)
        result["routing"] = routing

        # 分类器只能升级风险不能降级，这里再兜一次。
        if routing["risk"] == "emergency":
            result["answer"] = safety.escalation_message(routing["red_flags"] or flags)
            result["escalated"] = True
            return result
        result["escalated"] = False

        passages, used_live, rerank_degraded = self.gather(query, routing)
        result["passages"] = passages
        result["used_live"] = used_live
        # 哪些 LLM 阶段没真正跑起来，必须让调用方看得见。
        # 静默降级看上去和「很快」一模一样，这正是最危险的地方。
        result["degraded"] = [s for s, bad in
                              (("分流", routing.get("_fallback")),
                               ("重排", rerank_degraded)) if bad]

        result["provider"] = provider_label()
        if not available():
            result["answer"] = (
                "【未配置任何模型 API Key，无法生成回答】\n"
                f"本轮检索到 {len(passages)} 条相关资料，检索层与安全层工作正常。\n"
                "请在项目根目录 .env 中配置 ANTHROPIC_API_KEY、OPENAI_API_KEY "
                "或 PERPLEXITY_API_KEY 中的任意一个。"
            )
            result["error"] = "no_model_key"
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

        # 只缓存真正通过了引用与合规校验的答案。兜底话术、弃答、生成失败都不写入，
        # 否则一次偶发失败会被固化成这个问题的永久标准答案。
        if cacheable and not result.get("error") and result["sources"]:
            try:
                self.cache.put(query, answer, result["sources"], query_vector)
            except Exception:
                pass
        return result

    def _generate(self, messages):
        return generate(self.system, messages, max_tokens=8000, effort="high")

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

    # -- 流式作答 ------------------------------------------------------------
    def answer_stream(self, query, history=None):
        """产出 (kind, payload)：meta / delta / done。

        安全边界与非流式路径完全一致：报警症状筛查在最前，缓存与急症直接一次性返回，
        逐句合规闸门保证违规内容永远不会流到前端。闸门拦下或引用越界时，
        自动退回非流式重写流程，前端用整段结果替换已显示的内容。
        """
        from agent.stream import stream_answer

        flags, instruction = safety.screen_input(query)

        if flags and safety.mentions_self(query):
            yield ("meta", {"escalated": True, "cached": False})
            yield ("done", {"answer": safety.escalation_message(flags), "sources": []})
            return

        # 必须写成 is not None：AnswerCache 定义了 __len__，所以 bool(空缓存) 为 False。
        # 写成 bool(self.cache) 的后果是：缓存一旦被清空（例如语料更新导致整体失效），
        # 就再也不会写入任何条目，自己把自己锁死在空状态。
        cacheable = (self.cache is not None) and (not history) and (not flags)
        query_vector = None
        if cacheable:
            hit = self.cache.lookup_literal(query)
            if hit is None:
                try:
                    query_vector = self.retriever._embed_query(query)
                    hit = self.cache.lookup_semantic(query_vector)
                except Exception:
                    hit = None
            if hit:
                yield ("meta", {"escalated": False, "cached": True})
                yield ("done", {"answer": hit["answer"], "sources": hit.get("sources", [])})
                return

        routing = classify(query)
        if routing["risk"] == "emergency":
            yield ("meta", {"escalated": True, "cached": False})
            yield ("done", {"answer": safety.escalation_message(routing["red_flags"] or flags),
                            "sources": []})
            return

        passages, used_live, _ = self.gather(query, routing)
        yield ("meta", {"escalated": False, "cached": False,
                        "passages": len(passages), "used_live": used_live})

        if not available():
            yield ("done", {"answer": "当前未配置模型，无法生成回答。", "sources": []})
            return

        messages = list(history or [])
        messages.append({"role": "user",
                         "content": f"{build_evidence_block(passages)}\n\n## 用户问题\n\n{query}"})
        if routing.get("safety_instruction"):
            messages.append({"role": "system", "content": routing["safety_instruction"]})

        final = None
        for kind, payload in stream_answer(self.system, messages, passages):
            if kind == "delta":
                yield ("delta", payload)
            elif kind == "done":
                final = payload
            elif kind == "reject":
                # 流式结果不合格：走完整重写流程，用整段替换。
                fallback = self.answer(query, history)
                yield ("replace", {"answer": fallback["answer"],
                                   "sources": fallback.get("sources", [])})
                return

        if final is None:
            fallback = self.answer(query, history)
            yield ("replace", {"answer": fallback["answer"],
                               "sources": fallback.get("sources", [])})
            return

        sources = self._sources(final, passages)
        if cacheable and sources:
            try:
                self.cache.put(query, final, sources, query_vector)
            except Exception:
                pass
        yield ("done", {"answer": final, "sources": sources})
