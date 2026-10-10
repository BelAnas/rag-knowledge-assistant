"""Shared fixtures. Everything here is offline: hash embedder, word-overlap reranker, mock writer."""
from __future__ import annotations

import pytest

from rag.answer import Assistant, build_assistant
from rag.config import Settings, get_settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings(embedder="hash", reranker="lexical", llm_provider="mock", min_relevance=0.5, qdrant_url="")


@pytest.fixture(scope="session")
def assistant(settings: Settings) -> Assistant:
    return build_assistant(settings, in_memory=True)
