"""Turn Markdown documents into passages ("chunks") that are small enough to retrieve and cite."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .text import split_sentences

MAX_WORDS = 200  # a passage longer than this is split, so one hit never floods the prompt


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage.

    `key` identifies the unit that relevance is judged on (document + section). Two chunks
    share a key when one long section had to be split.
    """

    key: str
    doc_id: str
    title: str
    section: str
    text: str

    @property
    def embed_text(self) -> str:
        # The title and heading are part of what gets indexed: a passage that says
        # "It costs 9 EUR" is only findable if the index knows what "it" is.
        head = f"{self.title}: {self.section}" if self.section else self.title
        return f"{head}\n{self.text}"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _pack(units: list[str], max_words: int) -> list[str]:
    """Greedily join units (paragraphs or sentences) into pieces of at most `max_words`."""
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        words = len(unit.split())
        if current and count + words > max_words:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(unit)
        count += words
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_long(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Split on paragraphs first and on sentences only when one paragraph is too long."""
    if len(text.split()) <= max_words:
        return [text]
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph.split()) > max_words:
            units.extend(split_sentences(paragraph))
        else:
            units.append(paragraph)
    return _pack(units, max_words)


def chunk_markdown(doc_id: str, markdown: str, max_words: int = MAX_WORDS) -> list[Chunk]:
    """Split one Markdown document at its `##` headings.

    Headings are where the author changed subject, so they are better boundaries than a
    fixed character count that cuts a rule in half.
    """
    title = doc_id
    sections: list[tuple[str, list[str]]] = [("Overview", [])]
    for line in markdown.splitlines():
        if line.startswith("# ") and title == doc_id:
            title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for section, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        for piece in split_long(body, max_words):
            chunks.append(Chunk(f"{doc_id}#{slugify(section)}", doc_id, title, section, piece))
    return chunks


def load_markdown_dir(folder: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8")))
    return chunks
