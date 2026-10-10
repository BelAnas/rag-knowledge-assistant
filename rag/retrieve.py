"""Retrieval pipeline: keyword search + meaning search -> fusion -> optional rerank."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .chunking import Chunk
from .embed import Embedder
from .index import KnowledgeIndex
from .rerank import Reranker

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
RRF_K = 60  # the constant from the original reciprocal rank fusion paper; rarely worth tuning


@dataclass
class Hit:
    chunk: Chunk
    index: int
    keyword_rank: int | None = None   # position in the keyword list (1 = best), None = not in it
    dense_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.chunk.key, "doc_id": self.chunk.doc_id, "title": self.chunk.title,
            "section": self.chunk.section, "text": self.chunk.text,
            "keyword_rank": self.keyword_rank, "dense_rank": self.dense_rank,
            "fused_score": round(self.fused_score, 5),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 3),
        }


@dataclass
class Retrieval:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    top_similarity: float | None = None  # best cosine similarity of any passage to the query


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Merge ranked lists using positions only: score = sum of 1 / (k + rank).

    BM25 scores and cosine similarities live on unrelated scales, so adding them needs
    per-corpus weights. Ranks are comparable by construction.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


def weighted_fusion(
    keyword: list[tuple[int, float]], dense: list[tuple[int, float]], dense_weight: float
) -> dict[int, float]:
    """Merge by score: dense_weight x meaning score + (1 - dense_weight) x keyword score.

    Each list is first rescaled to 0-1 (best = 1, last candidate = 0) so the two scales
    become comparable. Unlike rank fusion this keeps the size of the gaps: a keyword match
    that is far ahead of the rest counts for more than one that barely leads. The price is
    one number to tune, which `python -m evals.tune_fusion` does on training queries.
    """

    def rescale(pairs: list[tuple[int, float]]) -> dict[int, float]:
        if not pairs:
            return {}
        high, low = pairs[0][1], pairs[-1][1]
        return {i: (score - low) / (high - low) if high > low else 1.0 for i, score in pairs}

    fused: dict[int, float] = {}
    for weight, scores in ((1 - dense_weight, rescale(keyword)), (dense_weight, rescale(dense))):
        for i, score in scores.items():
            fused[i] = fused.get(i, 0.0) + weight * score
    return fused


class Retriever:
    def __init__(
        self,
        index: KnowledgeIndex,
        embedder: Embedder,
        reranker: Reranker | None,
        candidates: int = 20,
        rerank_top_n: int = 20,
        dense_weight: float | None = 0.7,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rerank_top_n = rerank_top_n
        # None = reciprocal rank fusion; a number between 0 and 1 = weighted score fusion.
        self.dense_weight = dense_weight

    def search(self, query: str, k: int = 4, mode: str = "hybrid_rerank") -> Retrieval:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        timings: dict[str, float] = {}
        hits: dict[int, Hit] = {}

        def hit(i: int) -> Hit:
            if i not in hits:
                hits[i] = Hit(self.index.chunks[i], i)
            return hits[i]

        keyword: list[tuple[int, float]] = []
        dense: list[tuple[int, float]] = []
        if mode != "dense":
            start = time.perf_counter()
            keyword = self.index.search_keyword(query, self.candidates)
            timings["keyword"] = (time.perf_counter() - start) * 1000
            for rank, (i, _) in enumerate(keyword, start=1):
                hit(i).keyword_rank = rank
        if mode != "bm25":
            start = time.perf_counter()
            vector = self.embedder.embed_query(query).tolist()
            timings["embed_query"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            dense = self.index.search_dense(vector, self.candidates)
            timings["dense"] = (time.perf_counter() - start) * 1000
            for rank, (i, _) in enumerate(dense, start=1):
                hit(i).dense_rank = rank

        if self.dense_weight is None or mode in ("bm25", "dense"):
            fused = reciprocal_rank_fusion([[i for i, _ in keyword], [i for i, _ in dense]])
        else:
            fused = weighted_fusion(keyword, dense, self.dense_weight)
        for i, score in fused.items():
            hits[i].fused_score = score
        # Ties are broken by passage index so that two runs always return the same order.
        ordered = sorted(hits.values(), key=lambda h: (-h.fused_score, h.index))

        if mode == "hybrid_rerank" and self.reranker is not None and ordered:
            start = time.perf_counter()
            shortlist = ordered[: self.rerank_top_n]
            scores = self.reranker.score(query, [h.chunk.embed_text for h in shortlist])
            for h, score in zip(shortlist, scores):
                h.rerank_score = score
            ordered = sorted(shortlist, key=lambda h: (-(h.rerank_score or 0.0), h.index))
            timings["rerank"] = (time.perf_counter() - start) * 1000

        return Retrieval(ordered[:k], {name: round(ms, 2) for name, ms in timings.items()},
                         dense[0][1] if dense else None)
