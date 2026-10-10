"""BM25 keyword search.

Embeddings are weak on exact strings (an error code, a part number): 'PU-204' and 'PU-240'
look almost the same to them. BM25 scores a passage by the query words it contains, weighted
by how rare each word is, and that is exactly what those questions need.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .text import tokenize


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.size = len(texts)
        lengths = np.zeros(self.size, dtype=np.float32)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(i)
                tfs.append(tf)
        avg = float(lengths.mean()) if self.size else 1.0
        # Length normalisation is precomputed per passage: long passages mention more words
        # by chance, so their term counts are worth less.
        self._norm = k1 * (1 - b + b * lengths / max(avg, 1e-9))
        self._postings = {
            term: (np.asarray(ids, dtype=np.int32), np.asarray(tfs, dtype=np.float32))
            for term, (ids, tfs) in postings.items()
        }
        self._idf = {
            term: math.log(1 + (self.size - len(ids) + 0.5) / (len(ids) + 0.5))
            for term, (ids, _) in self._postings.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to `k` (passage index, score) pairs, best first. Zero scores are dropped."""
        scores = np.zeros(self.size, dtype=np.float32)
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if posting is None:
                continue
            ids, tfs = posting
            scores[ids] += self._idf[term] * tfs * (self.k1 + 1) / (tfs + self._norm[ids])
        if not scores.any():
            return []
        k = min(k, self.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
