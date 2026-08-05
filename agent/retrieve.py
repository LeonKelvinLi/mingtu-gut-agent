"""Hybrid retrieval: BM25 + dense, fused, metadata-filtered, then reranked.

Pipeline per query:
    BM25 top-30  ─┐
                  ├─ reciprocal rank fusion ─ metadata scoring ─ rerank ─ top-8
    dense top-30 ─┘

Reciprocal rank fusion rather than score blending, because BM25 scores and
cosine similarities are not on comparable scales and normalising them is a
tuning exercise that never stops. RRF only needs the rank order from each.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from agent.index import load_index, tokenize

RRF_K = 60           # standard RRF damping constant
CANDIDATES = 30      # per retriever, before fusion
DEFAULT_TOP_N = 8    # passages handed to the answering model

# Metadata weighting. Deliberately mild — these nudge ordering, they do not
# override relevance. A perfectly relevant D-grade passage still beats an
# irrelevant A-grade one, because the model needs to be able to say
# "this is only animal evidence" rather than never seeing it at all.
GRADE_BOOST = {"A": 0.30, "B": 0.20, "C": 0.10, "D": 0.0, "E": 0.0}
AUTHORITY_BOOST = {
    "meta-analysis": 0.25, "guideline": 0.22, "rct": 0.18,
    "cohort": 0.08, "animal": 0.0, "opinion": 0.0, "unspecified": 0.05,
}


class Retriever:
    def __init__(self):
        self.chunks, self.vectors = load_index()
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._client = None

    # -- embedding -----------------------------------------------------------
    def _embed_query(self, query):
        """Embed via the same backend that built the index.

        Routing this through agent.embed rather than calling a provider directly
        is what keeps the query and the corpus in one vector space — the earlier
        hardcoded OpenAI call silently produced 3072-dim queries against a
        1024-dim local index.
        """
        from agent.embed import embed as embed_texts
        return embed_texts([query])[0]

    # -- fusion --------------------------------------------------------------
    def _fuse(self, query):
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_rank = np.argsort(bm25_scores)[::-1][:CANDIDATES]

        dense_scores = self.vectors @ self._embed_query(query)
        dense_rank = np.argsort(dense_scores)[::-1][:CANDIDATES]

        fused = {}
        for rank, idx in enumerate(bm25_rank):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(dense_rank):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

        for idx in fused:
            fused[idx] *= 100.0  # scale so metadata boosts are comparable
        return fused, bm25_scores, dense_scores

    # -- metadata ------------------------------------------------------------
    def _apply_metadata(self, fused, filters):
        filters = filters or {}
        want_populations = set(filters.get("population") or [])
        want_regulatory = filters.get("regulatory", False)
        min_year = filters.get("min_year")

        scored = []
        for idx, base in fused.items():
            chunk = self.chunks[idx]
            score = base

            score += GRADE_BOOST.get(chunk.get("evidence_grade") or "", 0.0)
            score += AUTHORITY_BOOST.get(chunk.get("authority", "unspecified"), 0.0)

            # Population match is a boost, not a hard filter: a question about a
            # child should prefer paediatric passages but must not be blind to
            # the general-population evidence when no paediatric data exists.
            if want_populations:
                if want_populations & set(chunk.get("population") or []):
                    score += 0.35
                elif "通用" in (chunk.get("population") or []):
                    score += 0.05

            if want_regulatory and chunk.get("is_regulatory"):
                score += 0.45

            # Recency only where the source stated a year at all.
            year = chunk.get("year")
            if min_year and year and year < min_year:
                score -= 0.20
            if year and year >= 2022:
                score += 0.10

            scored.append((score, idx))

        scored.sort(reverse=True)
        return scored

    # -- selection helpers ---------------------------------------------------
    @staticmethod
    def _too_similar(chunk, chosen):
        """Suppress near-duplicates produced by the chunk overlap window.

        Adjacent chunks share an 90-char tail, so the same section can occupy
        several rerank slots and crowd out a different section entirely.
        """
        for other in chosen:
            if chunk["source"] != other["source"]:
                continue
            if chunk["heading"] == other["heading"]:
                return True
        return False

    def _quota_select(self, scored, filters, top_n):
        """Guarantee that requested metadata classes actually appear.

        A boost only reorders within what already ranked; for a regulatory
        question the regulatory corpus must be present even if its lexical
        overlap with the question is poor. Reserving slots is the only way to
        make that a guarantee rather than a hope.
        """
        filters = filters or {}
        quotas = []
        if filters.get("regulatory"):
            quotas.append(("is_regulatory", lambda c: c.get("is_regulatory"), 3))
        wanted = set(filters.get("population") or [])
        if wanted:
            quotas.append(("population", lambda c: wanted & set(c.get("population") or []), 2))

        chosen, used = [], set()

        for _, predicate, count in quotas:
            filled = 0
            for score, idx in scored:
                if filled >= count or len(chosen) >= top_n:
                    break
                if idx in used:
                    continue
                chunk = self.chunks[idx]
                if not predicate(chunk):
                    continue
                if self._too_similar(chunk, chosen):
                    continue
                chosen.append(chunk)
                used.add(idx)
                filled += 1

        for score, idx in scored:
            if len(chosen) >= top_n:
                break
            if idx in used:
                continue
            chunk = self.chunks[idx]
            if self._too_similar(chunk, chosen):
                continue
            chosen.append(chunk)
            used.add(idx)

        return [(idx, self.chunks[idx]) for idx in used], used

    # -- public --------------------------------------------------------------
    def search(self, query, filters=None, top_n=DEFAULT_TOP_N):
        """Return candidate passages, best first, before LLM reranking."""
        fused, bm25_scores, dense_scores = self._fuse(query)
        scored = self._apply_metadata(fused, filters)
        score_by_idx = {idx: score for score, idx in scored}

        # Over-fetch here: the reranker culls to the final 5-8.
        pool = max(top_n * 3, 20)
        _, used = self._quota_select(scored, filters, pool)

        results = []
        for idx in used:
            chunk = dict(self.chunks[idx])
            chunk["_score"] = round(float(score_by_idx.get(idx, 0.0)), 4)
            chunk["_bm25"] = round(float(bm25_scores[idx]), 3)
            chunk["_dense"] = round(float(dense_scores[idx]), 3)
            results.append(chunk)
        results.sort(key=lambda c: c["_score"], reverse=True)
        return results


def infer_filters(query):
    """Cheap, deterministic metadata filters read straight off the question.

    The LLM classifier in classify.py refines these; this exists so retrieval
    still filters sensibly if the classifier call fails.
    """
    filters = {}
    populations = []
    if re.search(r"婴儿|新生儿|宝宝|奶粉|早产", query):
        populations.append("婴幼儿")
    if re.search(r"孩子|儿童|小孩|小儿|青少年", query):
        populations.append("儿童")
    if re.search(r"孕妇|怀孕|妊娠|哺乳", query):
        populations.append("孕产妇")
    if re.search(r"老人|老年|高龄", query):
        populations.append("老年")
    if re.search(r"化疗|免疫抑制|移植|白细胞低", query):
        populations.append("免疫抑制")
    if populations:
        filters["population"] = populations

    if re.search(r"法规|合法|违法|宣称|广告|保健食品|批准|名单|标准|能不能说", query):
        filters["regulatory"] = True

    return filters
