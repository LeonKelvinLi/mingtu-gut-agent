"""Hybrid retrieval: fusion, metadata weighting, quotas, near-duplicate control.

No network. The query embedding is stubbed in both fixtures — see conftest —
so BM25, RRF fusion, metadata scoring and slot reservation all run for real,
against the real 443-chunk index where it matters.

The property that carries the most weight here is the regulatory quota. A boost
only reorders what already ranked; a compliance question whose wording shares no
vocabulary with the regulation text needs the regulatory passages *reserved*, or
the model answers a legal question from the nutrition corpus.
"""

import numpy as np
import pytest

from agent.retrieve import (
    AUTHORITY_BOOST,
    GRADE_BOOST,
    Retriever,
    infer_filters,
)
from conftest import known_gap, needs_openai


# --- infer_filters: the deterministic fallback --------------------------------
# This exists for when the LLM classifier is down, so it has to stand alone.

@pytest.mark.parametrize("query,population", [
    ("我家宝宝能吃益生菌吗", "婴幼儿"),
    ("六个月的婴儿可以补充吗", "婴幼儿"),
    ("奶粉里加的益生菌有用吗", "婴幼儿"),
    ("孩子拉肚子吃什么菌株", "儿童"),
    ("小孩便秘怎么办", "儿童"),
    ("青少年可以长期吃吗", "儿童"),
    ("怀孕期间能吃益生菌吗", "孕产妇"),
    ("哺乳期可以吃吗", "孕产妇"),
    ("老人便秘能不能吃", "老年"),
    ("化疗期间能吃益生菌吗", "免疫抑制"),
    ("器官移植后可以吃吗", "免疫抑制"),
])
def test_population_filter_inferred(query, population):
    assert population in infer_filters(query).get("population", [])


@pytest.mark.parametrize("query", [
    "益生菌能不能宣称调节肠道菌群",
    "保健食品批准文号怎么查",
    "这个广告说法违法吗",
    "可用于食品的菌种名单在哪看",
    "益生菌产品能不能说改善便秘",
])
def test_regulatory_filter_inferred(query):
    assert infer_filters(query).get("regulatory") is True


def test_plain_question_gets_no_filters():
    assert infer_filters("益生菌一般要吃多久才有效") == {}


@pytest.mark.parametrize("query", ["婴幼儿可以吃哪些菌种", "婴幼儿食品菌种名单里有什么"])
@known_gap("infer_filters' 婴幼儿 pattern lists 婴儿|新生儿|宝宝|奶粉|早产 but not the "
           "literal word 婴幼儿 — which is the term the regulation itself uses "
           "(可用于婴幼儿食品的菌种名单). This is the classifier's fallback path, so "
           "it matters exactly when the LLM stage is already down")
def test_literal_infant_term_is_inferred(query):
    assert "婴幼儿" in infer_filters(query).get("population", [])


@known_gap("infer_filters' 免疫抑制 pattern misses the lay phrasing 免疫力低下 / "
           "抵抗力差, which is how a user describes the highest-risk group for "
           "probiotic sepsis")
@pytest.mark.parametrize("query", ["免疫力低下的人能吃益生菌吗", "抵抗力差可以补充吗"])
def test_lay_immunocompromised_phrasing_is_inferred(query):
    assert "免疫抑制" in infer_filters(query).get("population", [])


# --- metadata weighting -------------------------------------------------------

def scores_by_id(retriever, scored):
    return {retriever.chunks[idx]["id"]: score for score, idx in scored}


def test_evidence_grade_and_authority_lift_equal_base_scores(synthetic_retriever):
    fused = {i: 1.0 for i in range(len(synthetic_retriever.chunks))}
    scored = synthetic_retriever._apply_metadata(fused, None)
    by_id = scores_by_id(synthetic_retriever, scored)
    assert by_id["strainA"] > by_id["diet1"] > by_id["diet2"] > by_id["strainD"]


def test_grade_and_authority_boosts_are_mild_by_design(synthetic_retriever):
    """A D-grade passage must stay reachable; the model needs to be able to say
    'this is only animal evidence' rather than never seeing the passage."""
    assert max(GRADE_BOOST.values()) < 1.0
    assert max(AUTHORITY_BOOST.values()) < 1.0
    fused = {3: 10.0}                       # strainD, strongly relevant
    fused.update({2: 1.0})                  # strainA, weakly relevant
    scored = synthetic_retriever._apply_metadata(fused, None)
    assert synthetic_retriever.chunks[scored[0][1]]["id"] == "strainD"


def test_regulatory_filter_boosts_regulatory_chunks(synthetic_retriever):
    fused = {i: 1.0 for i in range(len(synthetic_retriever.chunks))}
    plain = scores_by_id(synthetic_retriever, synthetic_retriever._apply_metadata(fused, None))
    with_reg = scores_by_id(
        synthetic_retriever, synthetic_retriever._apply_metadata(fused, {"regulatory": True}))
    assert with_reg["reg1"] - plain["reg1"] == pytest.approx(0.45)
    assert with_reg["diet1"] == pytest.approx(plain["diet1"])


def test_population_match_is_a_boost_not_a_hard_filter(synthetic_retriever):
    """A paediatric question must still see the general-population evidence."""
    fused = {i: 1.0 for i in range(len(synthetic_retriever.chunks))}
    scored = synthetic_retriever._apply_metadata(fused, {"population": ["儿童"]})
    returned = {synthetic_retriever.chunks[idx]["id"] for _, idx in scored}
    assert returned == {c["id"] for c in synthetic_retriever.chunks}
    by_id = scores_by_id(synthetic_retriever, scored)
    assert by_id["strainA"] > by_id["diet1"]


def test_old_evidence_is_penalised_only_when_a_year_is_known(synthetic_retriever):
    fused = {i: 1.0 for i in range(len(synthetic_retriever.chunks))}
    plain = scores_by_id(synthetic_retriever, synthetic_retriever._apply_metadata(fused, None))
    filtered = scores_by_id(
        synthetic_retriever, synthetic_retriever._apply_metadata(fused, {"min_year": 2020}))
    assert filtered["strainD"] < plain["strainD"]      # 2015, penalised
    assert filtered["diet1"] == pytest.approx(plain["diet1"])   # 2022, untouched


# --- quotas -------------------------------------------------------------------

def test_regulatory_quota_reserves_slots_for_badly_ranked_regulations(synthetic_retriever):
    """The whole point: regulations that rank last still get in."""
    ordered = [(9.0, 4), (8.0, 6), (7.0, 2), (6.0, 3), (5.0, 7),
               (0.2, 0), (0.1, 1)]         # reg1/reg2 dead last
    _, used = synthetic_retriever._quota_select(ordered, {"regulatory": True}, top_n=4)
    chosen = {synthetic_retriever.chunks[i]["id"] for i in used}
    assert {"reg1", "reg2"} <= chosen, f"法规问题没有召回法规段落，实得 {chosen}"
    assert len(chosen) == 4


def test_population_quota_reserves_slots(synthetic_retriever):
    ordered = [(9.0, 4), (8.0, 6), (7.0, 3), (0.1, 7)]   # limit1 (婴幼儿) last
    _, used = synthetic_retriever._quota_select(ordered, {"population": ["婴幼儿"]}, top_n=3)
    chosen = {synthetic_retriever.chunks[i]["id"] for i in used}
    assert "limit1" in chosen


def test_no_quota_means_pure_score_order(synthetic_retriever):
    ordered = [(9.0, 4), (8.0, 6), (0.1, 0)]
    _, used = synthetic_retriever._quota_select(ordered, None, top_n=2)
    assert {synthetic_retriever.chunks[i]["id"] for i in used} == {"diet1", "diet2"}


def test_quota_never_exceeds_top_n(synthetic_retriever):
    ordered = [(float(len(synthetic_retriever.chunks) - i), i)
               for i in range(len(synthetic_retriever.chunks))]
    for top_n in (1, 2, 3, 5):
        _, used = synthetic_retriever._quota_select(
            ordered, {"regulatory": True, "population": ["婴幼儿"]}, top_n=top_n)
        assert len(used) <= top_n


# --- near-duplicate suppression ----------------------------------------------

def test_too_similar_matches_on_source_and_heading(synthetic_retriever):
    chunks = {c["id"]: c for c in synthetic_retriever.chunks}
    assert Retriever._too_similar(chunks["diet1b"], [chunks["diet1"]]) is True
    assert Retriever._too_similar(chunks["diet2"], [chunks["diet1"]]) is False
    assert Retriever._too_similar(chunks["reg1"], [chunks["diet1"]]) is False


def test_overlapping_chunks_do_not_occupy_two_slots(synthetic_retriever):
    """diet1 and diet1b are the same section; only one may be selected."""
    ordered = [(9.0, 4), (8.9, 5), (7.0, 6), (6.0, 2)]
    _, used = synthetic_retriever._quota_select(ordered, None, top_n=3)
    chosen = [synthetic_retriever.chunks[i]["id"] for i in used]
    assert not {"diet1", "diet1b"} <= set(chosen), f"近重复未被抑制：{chosen}"
    assert "diet2" in chosen, "被抑制的槽位应该让给别的章节"


# --- search() end to end (offline) -------------------------------------------

def test_search_returns_scored_sorted_deduplicated_chunks(synthetic_retriever):
    results = synthetic_retriever.search("膳食纤维每天吃多少", top_n=4)
    assert results
    scores = [c["_score"] for c in results]
    assert scores == sorted(scores, reverse=True)
    for chunk in results:
        assert {"_score", "_bm25", "_dense"} <= set(chunk)
    keys = [(c["source"], c["heading"]) for c in results]
    assert len(keys) == len(set(keys))


def test_search_does_not_mutate_the_index(synthetic_retriever):
    before = [dict(c) for c in synthetic_retriever.chunks]
    synthetic_retriever.search("益生菌 法规", {"regulatory": True}, top_n=4)
    assert synthetic_retriever.chunks == before, "检索不能把 _score 写回索引"


def test_search_overfetches_for_the_reranker(synthetic_retriever):
    """search() deliberately returns more than top_n; classify.rerank culls.

    Documented here because the parameter name reads like a hard limit and a
    caller that trusts it will hand the model 20+ passages.
    """
    results = synthetic_retriever.search("膳食纤维", top_n=2)
    assert len(results) > 2


# --- against the real index (offline: embedding stubbed) ---------------------

REGULATORY_QUERIES = [
    "益生菌产品能不能宣称预防疾病",
    "保健食品和普通食品的功能宣称边界是什么",
    "可用于婴幼儿食品的菌种名单收录了哪些菌种",
]


@pytest.mark.parametrize("query", REGULATORY_QUERIES)
def test_regulatory_question_recalls_regulatory_passages(offline_retriever, query):
    results = offline_retriever.search(query, {"regulatory": True}, top_n=8)
    regulatory = [c for c in results if c["is_regulatory"]]
    assert len(regulatory) >= 3, (
        f"法规问题只召回 {len(regulatory)} 条法规段落：{query}")


def test_real_index_results_are_never_near_duplicates(offline_retriever):
    for query in ["益生菌要吃多久", "婴幼儿能吃哪些菌株", "膳食纤维每天多少克",
                  "肠道菌群检测靠谱吗", "益生菌有什么风险"]:
        results = offline_retriever.search(query, top_n=8)
        keys = [(c["source"], c["heading"]) for c in results]
        assert len(keys) == len(set(keys)), f"{query} 返回了同一章节的多个块"


def test_strain_identifiers_survive_retrieval(offline_retriever):
    """BM25 is in the mix precisely so DSM 17938 does not blur into DSM 9843."""
    results = offline_retriever.search("DSM 17938 有什么证据", top_n=8)
    assert any("17938" in c["text"] for c in results)


def test_population_filter_pulls_matching_passages(offline_retriever):
    results = offline_retriever.search(
        "婴儿可以吃哪些益生菌", {"population": ["婴幼儿"]}, top_n=8)
    matched = [c for c in results if "婴幼儿" in (c["population"] or [])]
    assert matched, "婴幼儿问题没有召回婴幼儿段落"


def test_every_indexed_chunk_carries_the_metadata_retrieval_depends_on(offline_retriever):
    for chunk in offline_retriever.chunks:
        assert {"id", "text", "source", "heading", "population",
                "authority", "is_regulatory"} <= set(chunk)
        assert chunk["evidence_grade"] in {None, "A", "B", "C", "D", "E"}
        assert isinstance(chunk["population"], list) and chunk["population"]


def test_index_vectors_are_normalised(offline_retriever):
    norms = np.linalg.norm(offline_retriever.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "检索用点积代替余弦，向量必须已归一化"


def test_index_rows_match_chunk_count(offline_retriever):
    assert offline_retriever.vectors.shape[0] == len(offline_retriever.chunks)


# --- live -------------------------------------------------------------------

@needs_openai
def test_live_embedding_retrieval_finds_a_paraphrased_question():
    """Dense retrieval earns its place on questions that share no keywords."""
    retriever = Retriever()
    results = retriever.search("吃了益生菌肚子反而更胀了是怎么回事", top_n=8)
    assert results
    assert any("胀" in c["text"] or "产气" in c["text"] for c in results)
