"""Choose how keyword and meaning results are merged, without peeking at the test questions.

    python -m evals.tune_fusion

Every fusion setting is scored on the 809 SciFact *training* queries. The winner is then
reported, once, on the 300 SciFact test queries and on the handbook questions. Neither of
those two sets is used to pick the setting.
"""
from __future__ import annotations

import json
import re
import statistics

from rag.config import get_settings
from rag.embed import make_embedder
from rag.index import KnowledgeIndex
from rag.retrieve import reciprocal_rank_fusion, weighted_fusion

from .datasets import ROOT, Query, load_handbook, load_scifact
from .retrieval import score_query

WEIGHTS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
Lists = tuple[list[tuple[int, float]], list[tuple[int, float]]]


def candidate_lists(index: KnowledgeIndex, embedder, queries: list[Query], k: int) -> list[Lists]:
    return [(index.search_keyword(q.text, k), index.search_dense(embedder.embed_query(q.text).tolist(), k))
            for q in queries]


def ndcg(index: KnowledgeIndex, queries: list[Query], lists: list[Lists], dense_weight: float | None) -> float:
    scores = []
    for query, (keyword, dense) in zip(queries, lists):
        if dense_weight is None:
            fused = reciprocal_rank_fusion([[i for i, _ in keyword], [i for i, _ in dense]])
        else:
            fused = weighted_fusion(keyword, dense, dense_weight)
        ranked = sorted(fused, key=lambda i: (-fused[i], i))
        scores.append(score_query([index.chunks[i].key for i in ranked], query.gold)["ndcg@10"])
    return round(statistics.fmean(scores), 4)


def main() -> None:
    settings = get_settings()
    embedder = make_embedder(settings.embedder, settings.embedding_model)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", embedder.name)
    labels = ["Keyword only", "Meaning only", "Rank fusion (RRF)"] + [f"Weighted, meaning {w}" for w in WEIGHTS]
    table: dict[str, dict[str, float]] = {label: {} for label in labels}

    def run(name: str, index: KnowledgeIndex, queries: list[Query]) -> None:
        lists = candidate_lists(index, embedder, queries, settings.candidates)
        # Weight 0 and 1 are the single searches; they bracket what fusion has to beat.
        for label, weight in zip(labels, (0.0, 1.0, None) + WEIGHTS):
            table[label][name] = ndcg(index, queries, lists, weight)

    chunks, train = load_scifact("train")
    index = KnowledgeIndex.open(chunks, embedder, settings.index_dir / f"scifact-{slug}")
    run("scifact_train", index, train)
    run("scifact_test", index, load_scifact("test")[1])
    index.close()

    chunks, questions = load_handbook(settings.docs_dir)
    index = KnowledgeIndex.open(chunks, embedder, settings.index_dir / slug)
    run("handbook", index, [q for q in questions if q.gold])
    index.close()

    fusions = labels[2:]
    chosen = max(fusions, key=lambda label: table[label]["scifact_train"])
    print("nDCG@10 for each way of merging the two searches\n")
    print("| Fusion | SciFact train (used to choose) | SciFact test | Handbook |")
    print("|---|---|---|---|")
    for label in labels:
        row = table[label]
        mark = " **(chosen)**" if label == chosen else ""
        print(f"| {label}{mark} | {row['scifact_train']:.3f} | {row['scifact_test']:.3f} | {row['handbook']:.3f} |")
    out = ROOT / "evals" / "report-fusion.json"
    out.write_text(json.dumps({"metric": "ndcg@10", "chosen_on": "scifact_train", "chosen": chosen,
                               "embedding_model": embedder.name, "candidates_per_search": settings.candidates,
                               "results": table}, indent=2) + "\n")
    print(f"\nChosen on the training queries: {chosen}\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
