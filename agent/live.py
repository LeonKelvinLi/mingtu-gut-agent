"""Live evidence retrieval via Perplexity, for what the local index cannot cover.

The curated index is graded, stable and cheap, but it is a snapshot. Three
question types fall outside it and are exactly where a health agent invents
things if you let it:

  * 现况类 — market, product, brand, price, availability
  * 时效类 — a regulation, standard or approved-strain list that has moved
  * 新证据 — a trial or review published after the corpus was built

Passages come back in the same shape as indexed chunks, carry their source URLs,
and are marked `is_live` so the answering model can say the claim came from a
live lookup rather than the graded corpus. They go through the same reranking
and the same citation validation — a live passage gets no special trust.

Note on model choice: the research pipeline uses sonar-deep-research because it
can afford minutes. This path is inside an interactive turn, so it uses
sonar-pro. Deep research here would make the agent unusable, not better.
"""

import hashlib
import json
import re
import subprocess

from agent.chunker import extract_metadata
from tools.research import key as perplexity_key, one_line

LIVE_MODEL = "sonar-pro"
TIMEOUT_SECONDS = 60

# Signals that the answer depends on facts newer than the corpus.
_NEEDS_LIVE = re.compile(
    r"最新|现在|目前|今年|去年|202[4-9]|新规|新版|更新|"
    r"哪个牌子|什么牌子|品牌|多少钱|价格|买|购买|哪里有|"
    r"上市|获批|新批准|最近的?研究|新研究|新证据"
)

LIVE_RIGOUR = (
    "用中文回答。这是给一个面向中国普通用户的肠道健康智能体做实时补充检索，"
    "要求：1 只回答有来源支撑的内容，没有来源就直说没查到；"
    "2 每条结论标注证据等级（A=人体RCT荟萃分析，B=单个人体RCT，C=观察性研究，"
    "D=动物或体外，E=专家意见或推测），只有D或E级必须写明「证据不足」；"
    "3 涉及菌株写到菌株号，涉及剂量写清单位和疗程；"
    "4 涉及法规写明文号和发布年份；5 不要给医疗诊断或治疗建议；"
    "6 结尾附来源URL列表。回答控制在800字以内。"
)


def needs_live(query, best_local_score, threshold=2.0):
    """Decide whether to spend a live lookup on this turn.

    Two triggers: the question explicitly asks for current facts, or local
    retrieval came back weak enough that answering from it would be guessing.
    """
    if _NEEDS_LIVE.search(query):
        return True
    return best_local_score < threshold


def search_live(query, model=LIVE_MODEL):
    """Query Perplexity and return chunk-shaped passages. [] on any failure."""
    # 本地模式必须在这里也失败关闭。llm.py 与 embed.py 都严格失败关闭，
    # 唯独实时检索曾经完全绕过闸门——用户的健康问题照样被发到美国的 API，
    # 而产品对客户的承诺是「数据不出本机」。少一个功能可以，承诺不成立不可以。
    from agent.llm import local_only
    if local_only():
        return []

    api_key = perplexity_key()
    if not api_key:
        return []

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": one_line(f"{query}\n\n{LIVE_RIGOUR}")}],
    }
    try:
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", str(TIMEOUT_SECONDS), "-X", "POST",
             "https://api.perplexity.ai/chat/completions",
             "-H", f"Authorization: Bearer {api_key}",
             "-H", "content-type: application/json",
             "--data-binary", "@-"],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True, text=True,
        )
        data = json.loads(proc.stdout)
    except Exception:
        return []

    if "choices" not in data:
        return []

    body = data["choices"][0]["message"]["content"]
    urls = []
    for c in (data.get("citations") or data.get("search_results") or []):
        url = c if isinstance(c, str) else c.get("url", "")
        if url:
            urls.append(url)

    # Split into paragraph-sized passages so a single long answer does not
    # occupy every rerank slot.
    passages = [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.strip()) >= 80]
    if not passages:
        passages = [body.strip()] if body.strip() else []

    chunks = []
    for i, passage in enumerate(passages[:6]):
        meta = extract_metadata(passage, "live:perplexity", "实时检索")
        meta.update({
            "text": f"【实时检索 · {query}】\n{passage}",
            "id": hashlib.sha1(f"live:{query}:{i}".encode("utf-8")).hexdigest()[:16],
            "is_live": True,
            "citations": (meta.get("citations") or []) + urls[:8],
            "_score": 0.0,
            "_bm25": 0.0,
            "_dense": 0.0,
        })
        chunks.append(meta)
    return chunks
