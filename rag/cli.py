"""Command line.

    python -m rag.cli index                 # build (or refresh) the index
    python -m rag.cli ask "What does PU-204 mean?"
    python -m rag.cli search "late parcel refund" --mode hybrid
    python -m rag.cli serve --port 8021     # API and web page
"""
from __future__ import annotations

import argparse
import time

from .answer import build_assistant
from .config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index")
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--mode", default="hybrid_rerank", choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
    search.add_argument("-k", type=int, default=5)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8021)
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run("rag.api:app", host=args.host, port=args.port)
        return

    start = time.perf_counter()
    assistant = build_assistant(get_settings())
    index = assistant.retriever.index
    try:
        if args.command == "index":
            docs = len({c.doc_id for c in index.chunks})
            print(f"Index ready: {docs} documents, {len(index.chunks)} passages, "
                  f"embedder {assistant.retriever.embedder.name}, {time.perf_counter() - start:.1f}s")
        elif args.command == "search":
            result = assistant.retriever.search(args.query, k=args.k, mode=args.mode)
            for n, h in enumerate(result.hits, start=1):
                score = f"rerank {h.rerank_score:.2f}" if h.rerank_score is not None else f"fused {h.fused_score:.4f}"
                print(f"{n}. {h.chunk.title} > {h.chunk.section}  "
                      f"[keyword #{h.keyword_rank or '-'}, meaning #{h.dense_rank or '-'}, {score}]")
            print(f"timings (ms): {result.timings_ms}")
        else:
            answer = assistant.ask(args.question)
            print(f"\n{answer.answer}\n")
            if answer.refused:
                print(f"(refused: {'relevance gate' if answer.refusal_reason == 'gate' else 'the model found no answer in the sources'})")
            for n in answer.citations:
                chunk = answer.sources[n - 1].chunk
                print(f"[{n}] {chunk.title} > {chunk.section}  (data/handbook/{chunk.doc_id}.md)")
            for warning in answer.warnings:
                print(f"warning: {warning}")
            t = answer.trace
            print(f"\nmodel {t['model']}, {t['prompt_tokens']}+{t['completion_tokens']} tokens, "
                  f"${t['cost_usd']}, {t['timings_ms']['total']} ms")  # type: ignore[index]
    finally:
        index.close()


if __name__ == "__main__":
    main()
