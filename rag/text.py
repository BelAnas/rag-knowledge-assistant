"""Tokenising shared by keyword search, the offline embedder and the answer checks."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i if in into is it its "
    "my no not of on or our so than that the their then there these they this to was we what "
    "when where which who why will with you your".split()
)


def _stem(word: str) -> str:
    # Only plurals are folded ("parcels" -> "parcel"). A full stemmer would match more word
    # forms but also merges unrelated words; meaning search already covers word variation.
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, without stopwords. 'PU-204' becomes ['pu', '204']."""
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.replace("\n", " ")) if s.strip()]
