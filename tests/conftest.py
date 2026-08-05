"""Shared fixtures and conventions for the mingtu test suite.

Run with:  .venv/bin/python -m pytest tests/ -q

Three conventions this suite follows, because they decide what a green run means:

1. Nothing here touches the network by default. Anything that would call OpenAI,
   Anthropic or Perplexity is marked `live` and skipped unless MINGTU_LIVE_TESTS=1
   is set AND the relevant key resolves. A green run therefore proves the
   deterministic layers work with every API down — which is the property the
   safety design actually claims.

2. Retrieval is still tested against the real 443-chunk index. Only the query
   embedding is stubbed (see `offline_retriever`), so BM25, fusion, metadata
   scoring, quotas and near-duplicate suppression all run on real data offline.

3. A test that documents a real, unfixed defect is marked
   `pytest.mark.xfail(strict=True)` via `known_gap()`. It stays green today and
   turns the run red the moment the defect is fixed, which is the signal to
   delete the marker. Grep for `known_gap` to see every open gap in one list.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RESEARCH_DIR = REPO / "research"
KNOWLEDGE_DIR = REPO / "knowledge"
CHUNKS_PATH = KNOWLEDGE_DIR / "chunks.jsonl"

RESEARCH_READY = RESEARCH_DIR.exists() and any(RESEARCH_DIR.glob("*.md"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: hits a real API; needs MINGTU_LIVE_TESTS=1 and a key")


def pytest_make_parametrize_id(config, val, argname):
    """Keep Chinese test IDs readable.

    pytest escapes non-ASCII ids to \\uXXXX by default, which turns a failure
    report into hex. The failing phrasing is the whole point of this suite —
    someone reading the report has to see 「拉屎带血」, not an escape sequence.
    """
    if isinstance(val, str) and any(ord(ch) > 127 for ch in val):
        return val
    return None


def known_gap(reason):
    """Mark a test that encodes a real, currently-unfixed defect.

    strict=True on purpose: when the defect is fixed the test XPASSes and the
    run goes red, forcing whoever fixed it to delete the marker. A silent xfail
    would let a fixed gap sit in the file forever looking broken.
    """
    return pytest.mark.xfail(
        strict=True,
        reason=f"KNOWN GAP (not yet fixed): {reason} — delete this marker once fixed",
    )


# --- live-API gating ----------------------------------------------------------

def _key(name):
    try:
        from agent.llm import _read_key
        return _read_key(name)
    except Exception:
        return os.environ.get(name)


LIVE_ENABLED = os.environ.get("MINGTU_LIVE_TESTS") == "1"

needs_openai = pytest.mark.skipif(
    not (LIVE_ENABLED and _key("OPENAI_API_KEY")),
    reason="live test: set MINGTU_LIVE_TESTS=1 and configure OPENAI_API_KEY")

needs_answering_model = pytest.mark.skipif(
    not (LIVE_ENABLED and (_key("ANTHROPIC_API_KEY") or _key("OPENAI_API_KEY")
                           or _key("PERPLEXITY_API_KEY"))),
    reason="live test: set MINGTU_LIVE_TESTS=1 and configure an answering-model key")

needs_perplexity = pytest.mark.skipif(
    not (LIVE_ENABLED and _key("PERPLEXITY_API_KEY")),
    reason="live test: set MINGTU_LIVE_TESTS=1 and configure PERPLEXITY_API_KEY")


# --- retrieval fixtures -------------------------------------------------------

@pytest.fixture(scope="session")
def offline_retriever():
    """The real index, with the query embedding stubbed so nothing leaves the box.

    The stub is not noise: it returns the stored vector of the chunk BM25 likes
    most, so the dense leg behaves like a real semantic neighbourhood lookup
    around the best lexical match. Fusion, metadata scoring, quota reservation
    and near-duplicate suppression are all exercised for real.
    """
    from agent.retrieve import Retriever

    try:
        from agent.index import tokenize
    except ImportError:                     # tokenizer moved to the embed layer
        from agent.embed import tokenize

    try:
        # Asking the loader beats checking a filename: vectors are stored per
        # embedding backend, so the path is not fixed.
        retriever = Retriever()
    except SystemExit as exc:
        pytest.skip(f"索引未构建，真实索引检索测试跳过：{exc}")

    def _embed(query):
        scores = retriever.bm25.get_scores(tokenize(query))
        return retriever.vectors[int(np.argmax(scores))]

    # Instance attribute shadows the bound method; the class stays untouched so
    # a live test in the same session can still build a real Retriever.
    retriever._embed_query = _embed
    return retriever


SYNTHETIC_CHUNKS = [
    {
        "id": "reg1", "source": "03-china-reg", "heading": "4.2 广告法边界",
        "text": "【4.2 广告法边界】\n《广告法》第十七条规定，除医疗、药品、医疗器械广告外，"
                "不得涉及疾病治疗功能。普通食品与保健食品的功能宣称边界由此划定。",
        "evidence_grade": None, "authority": "guideline", "population": ["通用"],
        "year": 2023, "is_regulatory": True, "citations": [],
    },
    {
        "id": "reg2", "source": "03-china-reg", "heading": "3.1 菌种名单",
        "text": "【3.1 菌种名单】\n国家卫健委发布的可用于食品的菌种名单收录了嗜酸乳杆菌等菌种，"
                "另设可用于婴幼儿食品的菌种名单，两份名单适用范围不同。",
        "evidence_grade": None, "authority": "guideline", "population": ["婴幼儿"],
        "year": 2022, "is_regulatory": True, "citations": [],
    },
    {
        "id": "strainA", "source": "01-strains", "heading": "2.1 LGG",
        "text": "【2.1 LGG】\n证据等级：A。一项荟萃分析显示 LGG ATCC 53103 每日100亿CFU "
                "可缩短儿童急性腹泻病程约一天，2024年更新。",
        "evidence_grade": "A", "authority": "meta-analysis", "population": ["儿童"],
        "year": 2024, "is_regulatory": False, "citations": ["https://pubmed.example/1"],
    },
    {
        "id": "strainD", "source": "01-strains", "heading": "2.9 动物数据",
        "text": "【2.9 动物数据】\n证据等级：D。小鼠实验提示某菌株可改变肠道屏障通透性，"
                "2015年发表，人体尚无验证。",
        "evidence_grade": "D", "authority": "animal", "population": ["通用"],
        "year": 2015, "is_regulatory": False, "citations": [],
    },
    {
        "id": "diet1", "source": "02-diet", "heading": "4.1 膳食纤维",
        "text": "【4.1 膳食纤维】\n证据等级：B。随机对照试验显示每日25至30克膳食纤维摄入与"
                "排便频率改善相关，成人受试者，2022年。",
        "evidence_grade": "B", "authority": "rct", "population": ["成人"],
        "year": 2022, "is_regulatory": False, "citations": [],
    },
    {
        # Same (source, heading) as diet1: this is what the chunk overlap window
        # produces in the real corpus, and what _too_similar exists to suppress.
        "id": "diet1b", "source": "02-diet", "heading": "4.1 膳食纤维",
        "text": "【4.1 膳食纤维】\n证据等级：B。膳食纤维摄入量超过每日40克时腹胀与产气"
                "报告增加，成人受试者，2022年。",
        "evidence_grade": "B", "authority": "rct", "population": ["成人"],
        "year": 2022, "is_regulatory": False, "citations": [],
    },
    {
        "id": "diet2", "source": "02-diet", "heading": "6.2 发酵食品",
        "text": "【6.2 发酵食品】\n证据等级：C。队列研究观察到发酵食品摄入与菌群多样性"
                "呈正相关，因果关系尚不明确，2021年。",
        "evidence_grade": "C", "authority": "cohort", "population": ["成人"],
        "year": 2021, "is_regulatory": False, "citations": [],
    },
    {
        "id": "limit1", "source": "04-limits", "heading": "3.4 禁忌人群",
        "text": "【3.4 禁忌人群】\n证据等级：E。专家意见认为早产儿、免疫抑制状态、"
                "中心静脉置管人群使用前应先咨询医生。",
        "evidence_grade": "E", "authority": "opinion", "population": ["婴幼儿", "免疫抑制"],
        "year": 2019, "is_regulatory": False, "citations": [],
    },
]


@pytest.fixture
def synthetic_retriever():
    """A hand-built 8-chunk index with controlled metadata and orthogonal vectors.

    Lets a test state exactly which chunk the dense leg should favour, which is
    impossible against the real embedding space.
    """
    from agent.index import tokenize
    from agent.retrieve import Retriever
    from rank_bm25 import BM25Okapi

    chunks = [dict(c) for c in SYNTHETIC_CHUNKS]
    retriever = Retriever.__new__(Retriever)
    retriever.chunks = chunks
    retriever.vectors = np.eye(len(chunks), dtype=np.float32)
    retriever.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    retriever._client = None
    # Dense leg favours strainA (index 2) unless a test overrides it.
    retriever._embed_query = lambda query: retriever.vectors[2]
    return retriever
