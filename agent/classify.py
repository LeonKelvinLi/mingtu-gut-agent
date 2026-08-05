"""Stage 1 (risk + intent classification) and stage 4 (reranking).

Both are LLM stages with deterministic fallbacks. The emergency decision is
never delegated to the model alone: safety.screen_input runs first and its
verdict can only be escalated by the classifier, never downgraded. A model that
decides 便血 is "probably fine" must not be able to suppress the referral.
"""

import os

from agent import safety
from agent.llm import call_json
from agent.retrieve import infer_filters

CATEGORIES = ["菌株证据", "饮食与益生元", "法规与产品", "安全性与禁忌", "机制与全身健康", "症状咨询", "其他"]

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "risk": {"type": "string", "enum": ["emergency", "caution", "normal"]},
        "category": {"type": "string", "enum": CATEGORIES},
        "populations": {"type": "array", "items": {
            "type": "string",
            "enum": ["婴幼儿", "儿童", "孕产妇", "老年", "免疫抑制", "成人", "通用"]}},
        "regulatory": {"type": "boolean"},
        "needs_current_info": {"type": "boolean"},
        "sub_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["risk", "category", "populations", "regulatory",
                 "needs_current_info", "sub_queries"],
    "additionalProperties": False,
}

CLASSIFY_SYSTEM = """你是一个医疗健康问答系统的前置分流模块，只做分类，不回答问题。

risk 判定：
- emergency：用户在描述自己出现消化道报警症状（便血、黑便、呕血、不明原因体重下降、
  吞咽困难、持续呕吐、贫血、腹部包块、夜间痛醒、黄疸、发热伴腹痛、剧烈腹痛、严重脱水）
- caution：涉及特殊人群（婴幼儿、孕产妇、免疫抑制、正在服药）或用户在问要不要停药、
  能不能替代治疗
- normal：一般科普咨询

sub_queries：把问题拆成 1-3 个用于检索的独立子问题，每个都要能独立检索。
如果问题本身已经很单一，就只返回它本身。子查询用陈述性关键词，不要用疑问句。"""


def classify(query):
    """Return the routing decision for one user turn.

    The deterministic screen runs first and sets the floor; the model may raise
    the risk level but never lower it.
    """
    flags, instruction = safety.screen_input(query)
    floor = "emergency" if (flags and safety.mentions_self(query)) else (
        "caution" if flags else "normal")

    result = call_json(
        CLASSIFY_SYSTEM,
        f"用户消息：{query}",
        CLASSIFY_SCHEMA,
        effort="low",
        max_tokens=1024,
        fast=True,
    )

    # _loose_json 现在也接受裸数组（重排需要），所以这里必须校验类型：
    # 模型偶尔会给 classify 返回一个数组，直接当 dict 用会 500。
    # 形状不对就走确定性兜底，这本来就是兜底存在的意义。
    if not isinstance(result, dict):
        result = None

    if result is None:
        # Deterministic fallback — retrieval and safety still work.
        filters = infer_filters(query)
        result = {
            "risk": floor,
            "category": "其他",
            "populations": filters.get("population", ["通用"]),
            "regulatory": filters.get("regulatory", False),
            "needs_current_info": False,
            "sub_queries": [query],
            "_fallback": True,
        }
    else:
        result["_fallback"] = False

    # Escalate-only merge with the deterministic screen.
    order = {"normal": 0, "caution": 1, "emergency": 2}
    if order[floor] > order[result["risk"]]:
        result["risk"] = floor

    result["red_flags"] = flags
    result["safety_instruction"] = instruction

    # Union the deterministic filters in, so a missed population in the model
    # output does not cost us the paediatric passages.
    det = infer_filters(query)
    populations = set(result.get("populations") or []) | set(det.get("population") or [])
    populations.discard("通用")
    result["populations"] = sorted(populations)
    result["regulatory"] = bool(result.get("regulatory")) or det.get("regulatory", False)

    if not result.get("sub_queries"):
        result["sub_queries"] = [query]
    return result


# --- Stage 4: reranking -------------------------------------------------------

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                # reason 只用于调试，但 12 个候选各写一句理由会撑爆输出上限，
                # 截断后 JSON 解析失败，整个重排阶段静默退回。删掉它。
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "integer"},
                },
                "required": ["index", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["selected"],
    "additionalProperties": False,
}

RERANK_SYSTEM = """你在为一个肠道健康问答系统挑选证据段落。

给你用户的问题和一组候选段落（带编号）。对每个段落按「能否直接支撑回答这个问题」
打分 0-10，只返回得分 >= 5 的段落，最多 8 个，按得分从高到低排列。

评分原则：
- 直接回答问题中具体实体（菌株号、剂量、人群、法规条款）的段落得分最高
- 只是主题相关但回答不了问题的段落给低分
- 段落之间内容重复时，只保留信息最全的那个
- 法规类问题必须保留法规段落，即使措辞不完全匹配
- 不要因为证据等级低就排除段落：用户需要知道某个说法证据不足，
  这类段落同样有价值"""


def rerank(query, candidates, top_n=8):
    """Cull the candidate pool to the passages that actually answer the question.

    Falls back to the hybrid ordering when the model is unavailable, which is
    already metadata-weighted and quota-balanced — degraded, not broken.
    """
    if not candidates:
        return []
    if len(candidates) <= 3:
        return candidates

    # 重排的输入长度直接决定本地模型的耗时：24 段 × 600 字的提示在 14B 上要 134 秒，
    # 而重排只是排序，不需要看全文。截到 12 段 × 320 字后质量基本不变，耗时降一个量级。
    # 云端模型没有这个约束，所以上限可配。
    max_cands = int(os.environ.get("MINGTU_RERANK_CANDIDATES", "12"))
    max_chars = int(os.environ.get("MINGTU_RERANK_CHARS", "320"))
    candidates = candidates[:max_cands]

    listing = "\n\n".join(
        f"[{i}] 证据等级={c.get('evidence_grade') or '未标注'} "
        f"法规={'是' if c.get('is_regulatory') else '否'} "
        f"人群={'/'.join(c.get('population') or [])}\n{c['text'][:max_chars]}"
        for i, c in enumerate(candidates)
    )

    result = call_json(
        RERANK_SYSTEM,
        f"用户问题：{query}\n\n候选段落：\n\n{listing}",
        RERANK_SCHEMA,
        effort="low",
        max_tokens=2048,
        fast=True,
    )

    # Models disagree on the shape they return here: some emit the documented
    # {"selected":[...]}, others a bare array, and they variously name the field
    # index / id / i. Rejecting all but one spelling meant the rerank stage
    # silently fell back on every provider we tried. Accept the variants.
    items = None
    if isinstance(result, list):
        items = result
    elif isinstance(result, dict):
        for key in ("selected", "results", "items", "passages"):
            if isinstance(result.get(key), list):
                items = result[key]
                break

    if not items:
        # Mark the degradation so it surfaces instead of looking like a fast path.
        out = candidates[:top_n]
        for c in out:
            c["_rerank_fallback"] = True
        return out

    chosen = []
    for item in items[:top_n]:
        if not isinstance(item, dict):
            continue
        idx = next((item[k] for k in ("index", "id", "i", "idx")
                    if isinstance(item.get(k), int)), None)
        if idx is not None and 0 <= idx < len(candidates):
            chunk = dict(candidates[idx])
            chunk["_rerank_score"] = item.get("score")
            chunk["_rerank_reason"] = item.get("reason", "")
            chosen.append(chunk)

    return chosen or candidates[:top_n]
