"""Settings, read once from the environment (and from a local .env file if there is one)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader. Real environment variables win over the file."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# The relevance gate compares the best reranker score with a threshold. The two rerankers
# score on different scales (model logits vs. a 0-1 word overlap), so each has its own
# default. The cross-encoder value was chosen from the score distribution printed by
# `python -m evals.answers --gate-sweep` on the handbook question set.
DEFAULT_MIN_RELEVANCE = {"cross-encoder": 0.0, "lexical": 0.5, "none": float("-inf")}


@dataclass(frozen=True)
class Settings:
    docs_dir: Path
    index_dir: Path
    llm_provider: str
    openai_api_key: str
    openai_model: str
    openai_base_url: str
    max_output_tokens: int
    embedder: str
    embedding_model: str
    reranker: str
    reranker_model: str
    qdrant_url: str
    candidates: int      # how many passages each search (keyword, meaning) contributes
    rerank_top_n: int    # how many fused candidates the reranker reads
    final_k: int         # how many passages the language model receives
    min_relevance: float # relevance gate: below this, the assistant refuses


def get_settings(**overrides: object) -> Settings:
    _load_dotenv(ROOT / ".env")
    env = os.environ.get
    reranker = env("RERANKER", "cross-encoder")
    values: dict[str, object] = dict(
        docs_dir=Path(env("DOCS_DIR", str(ROOT / "data" / "handbook"))),
        index_dir=Path(env("INDEX_DIR", str(ROOT / ".index"))),
        llm_provider=env("LLM_PROVIDER", "mock"),
        openai_api_key=env("OPENAI_API_KEY", ""),
        openai_model=env("OPENAI_MODEL", "gpt-4o-mini"),
        openai_base_url=env("OPENAI_BASE_URL", ""),
        max_output_tokens=int(env("MAX_OUTPUT_TOKENS", "300")),
        embedder=env("EMBEDDER", "fastembed"),
        embedding_model=env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        reranker=reranker,
        reranker_model=env("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        qdrant_url=env("QDRANT_URL", ""),
        candidates=int(env("CANDIDATES", "20")),
        rerank_top_n=int(env("RERANK_TOP_N", "20")),
        final_k=int(env("FINAL_K", "4")),
        min_relevance=float(env("MIN_RELEVANCE", "nan")),
    )
    values.update(overrides)
    if values["min_relevance"] != values["min_relevance"]:  # NaN means "not set"
        values["min_relevance"] = DEFAULT_MIN_RELEVANCE.get(str(values["reranker"]), float("-inf"))
    return Settings(**values)  # type: ignore[arg-type]
