from pathlib import Path

from rag.bm25 import BM25Index
from rag.chunking import chunk_markdown, load_markdown_dir, split_long
from rag.text import split_sentences, tokenize

DOC = """# Pickups

Intro line before any heading.

## Booking a pickup
Book before 12:00 for same-day collection.

## Missed pickups
The driver waits 5 minutes. Code PU-204 is recorded.
"""


def test_document_is_split_at_headings_and_keeps_its_title():
    chunks = chunk_markdown("pickups", DOC)
    assert [c.section for c in chunks] == ["Overview", "Booking a pickup", "Missed pickups"]
    assert {c.title for c in chunks} == {"Pickups"}
    assert chunks[1].key == "pickups#booking-a-pickup"


def test_heading_is_part_of_the_indexed_text():
    chunk = chunk_markdown("pickups", DOC)[2]
    assert chunk.embed_text.startswith("Pickups: Missed pickups\n")


def test_long_section_is_split_under_the_limit_and_shares_one_key():
    body = "\n\n".join(" ".join(f"word{p}_{i}" for i in range(60)) + "." for p in range(5))
    chunks = chunk_markdown("doc", f"# T\n\n## Long\n{body}", max_words=100)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 100 for c in chunks)
    assert {c.key for c in chunks} == {"doc#long"}


def test_single_huge_paragraph_falls_back_to_sentences():
    text = " ".join(f"Sentence number {i} has some words." for i in range(60))
    pieces = split_long(text, max_words=50)
    assert len(pieces) > 1 and all(len(p.split()) <= 50 for p in pieces)
    assert " ".join(pieces) == text  # nothing lost, nothing duplicated


def test_empty_sections_are_dropped():
    assert chunk_markdown("d", "# T\n\n## Empty\n\n## Full\ntext") [0].section == "Full"


def test_handbook_loads(settings):
    chunks = load_markdown_dir(settings.docs_dir)
    assert len({c.doc_id for c in chunks}) == 20
    assert len(chunks) == len({c.key for c in chunks}) == 85


def test_missing_folder_gives_no_chunks(tmp_path: Path):
    assert load_markdown_dir(tmp_path / "nope") == []


def test_tokenize_drops_stopwords_folds_plurals_and_splits_codes():
    assert tokenize("What does PU-204 mean for the parcels?") == ["pu", "204", "mean", "parcel"]
    assert tokenize("class address") == ["class", "addres"] or tokenize("class")[0] == "class"


def test_split_sentences_keeps_decimals_together():
    assert split_sentences("It costs 4.50 EUR per parcel. It cannot be added later.") == [
        "It costs 4.50 EUR per parcel.", "It cannot be added later."]


def test_bm25_finds_the_exact_code():
    index = BM25Index(["PU-204 means premises closed", "DP-220 means arrived at depot", "DP-235 means sorting delay"])
    assert index.search("what is DP-235", 3)[0][0] == 2


def test_bm25_rare_words_weigh_more_than_common_ones():
    index = BM25Index(["parcel parcel parcel", "parcel webhook", "parcel label", "parcel depot"])
    assert index.search("parcel webhook", 4)[0][0] == 1


def test_bm25_returns_nothing_for_unknown_words_and_empty_queries():
    index = BM25Index(["alpha beta", "gamma delta"])
    assert index.search("zzz", 5) == []
    assert index.search("", 5) == []
    assert index.search("the of and", 5) == []


def test_bm25_k_larger_than_corpus_is_safe():
    index = BM25Index(["alpha beta", "alpha"])
    assert len(index.search("alpha", 50)) == 2
