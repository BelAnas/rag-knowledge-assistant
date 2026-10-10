"""The two evaluation datasets, in one shape: passages, and queries with the keys that are relevant."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from rag.chunking import Chunk, load_markdown_dir

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "evals" / "questions.jsonl"
SCIFACT = ROOT / "data" / "scifact"


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    gold: frozenset[str]   # keys of the passages that answer it; empty = not answerable
    type: str = ""
    facts: tuple[str, ...] = ()


def load_handbook(docs_dir: Path) -> tuple[list[Chunk], list[Query]]:
    queries = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        queries.append(Query(row["id"], row["question"], frozenset(row["gold"]), row["type"], tuple(row["facts"])))
    return load_markdown_dir(docs_dir), queries


def load_scifact(split: str = "test") -> tuple[list[Chunk], list[Query]]:
    """`split` is "test" (300 queries, used for every reported number) or "train" (809, used for tuning)."""
    if not (SCIFACT / f"qrels-{split}.tsv").exists():
        raise SystemExit("SciFact is not downloaded. Run: python -m scripts.download_scifact")
    chunks = []
    for line in (SCIFACT / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        # One abstract = one passage, as in the published BEIR results, so numbers are comparable.
        chunks.append(Chunk(row["_id"], row["_id"], row["title"], "", row["text"]))
    gold: dict[str, set[str]] = {}
    with (SCIFACT / f"qrels-{split}.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["score"]) > 0:
                gold.setdefault(row["query-id"], set()).add(row["corpus-id"])
    queries = []
    for line in (SCIFACT / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["_id"] in gold:  # the file holds the queries of every split
            queries.append(Query(row["_id"], row["text"], frozenset(gold[row["_id"]]), "claim"))
    return chunks, queries
