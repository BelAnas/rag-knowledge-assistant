"""Download the BEIR SciFact benchmark (5,183 scientific abstracts, 300 test queries and 809
training queries with human relevance labels) into data/scifact/. It is a public dataset and is not committed.

    python -m scripts.download_scifact
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
TARGET = Path(__file__).resolve().parent.parent / "data" / "scifact"
WANTED = {"scifact/corpus.jsonl": "corpus.jsonl", "scifact/queries.jsonl": "queries.jsonl",
          "scifact/qrels/test.tsv": "qrels-test.tsv", "scifact/qrels/train.tsv": "qrels-train.tsv"}


def main() -> None:
    if all((TARGET / name).exists() for name in WANTED.values()):
        print(f"SciFact is already in {TARGET}")
        return
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    # Only the four known members are written, under fixed names, so nothing in the
    # archive can choose where it lands on disk.
    for member, name in WANTED.items():
        (TARGET / name).write_bytes(archive.read(member))
        print(f"  wrote {TARGET / name}")


if __name__ == "__main__":
    main()
