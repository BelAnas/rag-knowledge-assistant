"""Phase check: build the index and print what each search mode returns for a few questions."""
import sys
import time

from rag.chunking import load_markdown_dir
from rag.config import get_settings
from rag.embed import make_embedder
from rag.index import KnowledgeIndex
from rag.rerank import make_reranker
from rag.retrieve import MODES, Retriever

settings = get_settings()
chunks = load_markdown_dir(settings.docs_dir)
start = time.perf_counter()
embedder = make_embedder(settings.embedder, settings.embedding_model)
index = KnowledgeIndex.open(chunks, embedder, settings.index_dir, settings.qdrant_url)
reranker = make_reranker(settings.reranker, settings.reranker_model)
print(f"{len(chunks)} passages indexed with {embedder.name} in {time.perf_counter() - start:.1f}s")
retriever = Retriever(index, embedder, reranker, settings.candidates, settings.rerank_top_n)
for question in sys.argv[1:] or ["What does PU-204 mean?", "My parcel showed up late, do I get money back?"]:
    print(f"\nQ: {question}")
    for mode in MODES:
        result = retriever.search(question, k=3, mode=mode)
        print(f"  {mode:14}", [h.chunk.key for h in result.hits], result.timings_ms)
index.close()
