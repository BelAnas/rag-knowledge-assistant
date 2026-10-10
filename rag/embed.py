"""Embedders: text in, vector out. Passages with similar meaning get vectors that are close."""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    """A real embedding model, run on CPU through ONNX Runtime (no PyTorch needed)."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # imported late: loading it takes about a second

        self.name = model_name
        self._model = TextEmbedding(model_name)
        self.dim = int(self.embed_query("dimension probe").shape[0])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.passage_embed(texts, batch_size=8)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # Queries and passages go through different entry points because some models add a
        # short instruction in front of queries.
        return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)


class HashEmbedder:
    """Offline stand-in used by the tests: hashed word counts, no model, no download.

    It has no notion of meaning (it will not match 'late' with 'delayed'). It exists so the
    whole pipeline can be exercised deterministically in a second, not to be good.
    """

    name = "hash-256"
    dim = 256

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            # hashlib, not hash(): Python salts hash() per process, which would make the
            # index built in one run useless in the next.
            digest = hashlib.md5(token.encode()).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


def make_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder(model_name)
    raise ValueError(f"Unknown EMBEDDER '{kind}'. Use 'fastembed' or 'hash'.")
