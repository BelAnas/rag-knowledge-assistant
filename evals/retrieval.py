"""Retrieval evaluation: how often does each search stage put the right passage on top?

    python -m evals.retrieval --dataset handbook
    python -m evals.retrieval --dataset scifact
    python -m evals.retrieval --dataset scifact --rerankers Xenova/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base

Each stage is measured separately (keyword only, meaning only, hybrid, hybrid + rerank) so
the report shows what every component adds, and what it costs in latency.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path

from rag.config import get_settings
from rag.embed import make_embedder
from rag.index import KnowledgeIndex
from rag.rerank import make_reranker
from rag.retrieve import Retriever

from .datasets import ROOT, Query, load_handbook, load_scifact

DEPTH = 10  # metrics look at the top 10 results


def unique_keys(keys: list[str]) -> list[str]:
    """A long section split into two passages must not count twice."""
    return list(dict.fromkeys(keys))


def score_query(ranked: list[str], gold: frozenset[str]) -> dict[str, float]:
    """Metrics for one query. `ranked` is the result list, best first."""
    ranked = unique_keys(ranked)[:DEPTH]
    relevant = [key in gold for key in ranked]
    first = next((i for i, hit in enumerate(relevant, start=1) if hit), None)
    dcg = sum(1 / math.log2(i + 1) for i, hit in enumerate(relevant, start=1) if hit)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), DEPTH) + 1))
    return {
        "hit@1": float(any(relevant[:1])),
        "hit@4": float(any(relevant[:4])),          # 4 = how many passages the model receives
        "recall@10": sum(relevant) / len(gold),
        "mrr@10": 1 / first if first else 0.0,
        "ndcg@10": dcg / ideal,
    }


def evaluate(retriever: Retriever, queries: list[Query], mode: str) -> dict[str, object]:
    rows, latencies = [], []
    by_type: dict[str, list[float]] = {}
    for query in queries:
        start = time.perf_counter()
        result = retriever.search(query.text, k=DEPTH, mode=mode)
        latencies.append((time.perf_counter() - start) * 1000)
        row = score_query([h.chunk.key for h in result.hits], query.gold)
        rows.append(row)
        by_type.setdefault(query.type, []).append(row["hit@1"])
    summary: dict[str, object] = {m: round(statistics.fmean(r[m] for r in rows), 4) for m in rows[0]}
    summary["latency_ms_median"] = round(statistics.median(latencies), 1)
    summary["hit@1_by_type"] = {t: round(statistics.fmean(v), 4) for t, v in sorted(by_type.items())}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["handbook", "scifact"], default="handbook")
    parser.add_argument("--rerankers", default="", help="comma-separated cross-encoder models to compare")
    args = parser.parse_args()

    settings = get_settings()
    chunks, queries = load_handbook(settings.docs_dir) if args.dataset == "handbook" else load_scifact()
    queries = [q for q in queries if q.gold]  # unanswerable questions belong to the answer evaluation

    embedder = make_embedder(settings.embedder, settings.embedding_model)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", embedder.name)
    index_dir = settings.index_dir / (slug if args.dataset == "handbook" else f"scifact-{slug}")
    start = time.perf_counter()
    index = KnowledgeIndex.open(chunks, embedder, index_dir,
                                progress=lambda done, total: print(f"\r  embedding {done}/{total}", end="", flush=True))
    print(f"\n{args.dataset}: {len(chunks)} passages, {len(queries)} queries, index ready in {time.perf_counter() - start:.1f}s")

    report: dict[str, object] = {
        "dataset": args.dataset, "passages": len(chunks), "queries": len(queries),
        "embedding_model": embedder.name, "candidates_per_search": settings.candidates,
        "rerank_top_n": settings.rerank_top_n, "dense_weight": settings.dense_weight, "stages": {},
    }
    stages: dict[str, dict[str, object]] = report["stages"]  # type: ignore[assignment]
    base = Retriever(index, embedder, None, settings.candidates, settings.rerank_top_n, settings.dense_weight)
    for mode, label in (("bm25", "Keyword (BM25)"), ("dense", "Meaning (embeddings)"), ("hybrid", "Hybrid (weighted fusion)")):
        stages[label] = evaluate(base, queries, mode)

    # Rerankers are measured only on request: on this CPU one adds seconds per query.
    for model in [m for m in args.rerankers.split(",") if m]:
        reranker = make_reranker("cross-encoder", model)
        assert reranker is not None
        retriever = Retriever(index, embedder, reranker, settings.candidates, settings.rerank_top_n, settings.dense_weight)
        stages[f"Hybrid + rerank ({reranker.name})"] = evaluate(retriever, queries, "hybrid_rerank")
    index.close()

    print("\n| Stage | Hit@1 | Hit@4 | Recall@10 | MRR@10 | nDCG@10 | Median latency |")
    print("|---|---|---|---|---|---|---|")
    for label, s in stages.items():
        print(f"| {label} | {s['hit@1']:.3f} | {s['hit@4']:.3f} | {s['recall@10']:.3f} | {s['mrr@10']:.3f} "
              f"| {s['ndcg@10']:.3f} | {s['latency_ms_median']} ms |")
    if args.dataset == "handbook":
        print("\nHit@1 by question type:")
        for label, s in stages.items():
            print(f"  {label}: {s['hit@1_by_type']}")

    suffix = "" if settings.embedder == "fastembed" else f"-{settings.embedder}"
    out = ROOT / "evals" / f"report-retrieval-{args.dataset}{suffix}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
