"""Rerankers: a second, slower look at the few candidates the first search returned.

An embedding compresses a passage into one vector before it ever sees the question. A
cross-encoder reads the question and the passage together, so it can tell 'the fee for a
late pickup' from 'the pickup for a late fee'. It is too slow to run over every passage,
which is why it only reorders the short list.
"""
from __future__ import annotations

from typing import Protocol

from .text import tokenize


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self._model = TextCrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.rerank(query, texts, batch_size=16)]


class LexicalReranker:
    """Offline stand-in: the share of the question's words that the passage contains."""

    name = "lexical-overlap"

    def score(self, query: str, texts: list[str]) -> list[float]:
        terms = set(tokenize(query))
        if not terms:
            return [0.0] * len(texts)
        return [len(terms & set(tokenize(t))) / len(terms) for t in texts]


def make_reranker(kind: str, model_name: str) -> Reranker | None:
    if kind == "none":
        return None
    if kind == "lexical":
        return LexicalReranker()
    if kind == "cross-encoder":
        return CrossEncoderReranker(model_name)
    raise ValueError(f"Unknown RERANKER '{kind}'. Use 'cross-encoder', 'lexical' or 'none'.")
