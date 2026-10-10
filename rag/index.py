"""The knowledge index: passages stored twice, once for keyword search and once for meaning search."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable

from qdrant_client import QdrantClient, models

from .bm25 import BM25Index
from .chunking import Chunk
from .embed import Embedder

COLLECTION = "passages"
_BATCH = 128


def fingerprint(chunks: list[Chunk], embedder_name: str) -> str:
    """A hash of everything the stored vectors depend on. If it changes, the index is stale."""
    digest = hashlib.sha256(embedder_name.encode())
    for chunk in chunks:
        digest.update(chunk.key.encode())
        digest.update(chunk.embed_text.encode())
    return digest.hexdigest()[:16]


class KnowledgeIndex:
    def __init__(self, chunks: list[Chunk], client: QdrantClient) -> None:
        self.chunks = chunks
        self.client = client
        # BM25 is rebuilt in memory at start-up: it takes milliseconds for thousands of
        # passages, and it means there is only one stored copy of the text to keep in sync.
        self.bm25 = BM25Index([c.embed_text for c in chunks])

    # ---- building and loading -------------------------------------------------------------

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        client: QdrantClient,
        progress: Callable[[int, int], None] | None = None,
    ) -> "KnowledgeIndex":
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
        )
        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start : start + _BATCH]
            vectors = embedder.embed_passages([c.embed_text for c in batch])
            client.upsert(
                COLLECTION,
                points=models.Batch(
                    ids=list(range(start, start + len(batch))),
                    vectors=vectors.tolist(),
                    # The payload repeats the source so the collection is usable on its own
                    # when it lives on a shared Qdrant server.
                    payloads=[{"key": c.key, "doc_id": c.doc_id, "section": c.section} for c in batch],
                ),
            )
            if progress:
                progress(start + len(batch), len(chunks))
        return cls(chunks, client)

    @classmethod
    def open(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        index_dir: Path | None,
        qdrant_url: str = "",
        progress: Callable[[int, int], None] | None = None,
    ) -> "KnowledgeIndex":
        """Load the index from `index_dir` if it matches the documents, otherwise rebuild it.

        `index_dir=None` keeps everything in memory (used by the tests).
        """
        if index_dir is None:
            return cls.build(chunks, embedder, QdrantClient(":memory:"), progress)

        manifest_path = index_dir / "manifest.json"
        wanted = {"fingerprint": fingerprint(chunks, embedder.name), "count": len(chunks),
                  "embedder": embedder.name, "qdrant_url": qdrant_url}
        fresh = manifest_path.exists() and json.loads(manifest_path.read_text()) == wanted

        if qdrant_url:
            client = QdrantClient(url=qdrant_url)
        else:
            if not fresh and (index_dir / "qdrant").exists():
                shutil.rmtree(index_dir / "qdrant")
            index_dir.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(index_dir / "qdrant"))

        if fresh and client.collection_exists(COLLECTION):
            return cls(chunks, client)
        index = cls.build(chunks, embedder, client, progress)
        manifest_path.write_text(json.dumps(wanted, indent=2))
        return index

    # ---- the two searches -----------------------------------------------------------------

    def search_keyword(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.bm25.search(query, k)

    def search_dense(self, query_vector: list[float], k: int) -> list[tuple[int, float]]:
        result = self.client.query_points(COLLECTION, query=query_vector, limit=k, with_payload=False)
        return [(int(p.id), float(p.score)) for p in result.points]

    def close(self) -> None:
        self.client.close()
