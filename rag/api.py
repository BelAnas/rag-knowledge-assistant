"""HTTP API and demo page.

    uvicorn rag.api:app --port 8021
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .answer import Assistant, build_assistant
from .config import get_settings

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    # The length cap bounds the cost of one request: the question goes into the prompt.
    question: str = Field(min_length=3, max_length=500)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    k: int = Field(default=5, ge=1, le=20)


def create_app(assistant: Assistant | None = None) -> FastAPI:
    """`assistant` can be injected (the tests pass one built on offline components)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Models and index load once at start-up, not on the first question.
        app.state.assistant = assistant or build_assistant(get_settings())
        yield
        if assistant is None:
            app.state.assistant.retriever.index.close()

    app = FastAPI(title="RAG knowledge assistant", version="1.0.0", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        a: Assistant = request.app.state.assistant
        return {
            "status": "ok", "passages": len(a.retriever.index.chunks),
            "documents": len({c.doc_id for c in a.retriever.index.chunks}),
            "embedder": a.retriever.embedder.name,
            "reranker": a.retriever.reranker.name if a.retriever.reranker else None,
            "llm": a.llm.name,
        }

    @app.get("/documents")
    def documents(request: Request) -> list[dict[str, object]]:
        a: Assistant = request.app.state.assistant
        docs: dict[str, dict[str, object]] = {}
        for chunk in a.retriever.index.chunks:
            doc = docs.setdefault(chunk.doc_id, {"doc_id": chunk.doc_id, "title": chunk.title, "sections": []})
            if chunk.section not in doc["sections"]:  # type: ignore[operator]
                doc["sections"].append(chunk.section)  # type: ignore[attr-defined]
        return list(docs.values())

    @app.get("/documents/{doc_id}")
    def document(doc_id: str, request: Request) -> dict[str, object]:
        a: Assistant = request.app.state.assistant
        # The text is served from the index, never from a path built out of user input.
        chunks = [c for c in a.retriever.index.chunks if c.doc_id == doc_id]
        if not chunks:
            raise HTTPException(404, "Unknown document")
        return {"doc_id": doc_id, "title": chunks[0].title,
                "sections": [{"key": c.key, "section": c.section, "text": c.text} for c in chunks]}

    # Plain `def` endpoints run in FastAPI's thread pool, so a slow model call does not
    # block the other requests.
    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> dict[str, object]:
        a: Assistant = request.app.state.assistant
        try:
            return a.ask(body.question.strip()).to_dict()
        except Exception as error:  # a provider outage must be a clean 502, not a stack trace
            raise HTTPException(502, f"The language model could not be reached: {type(error).__name__}") from error

    @app.post("/search")
    def search(body: SearchRequest, request: Request) -> dict[str, object]:
        a: Assistant = request.app.state.assistant
        result = a.retriever.search(body.query.strip(), k=body.k, mode=body.mode)
        return {"mode": body.mode, "hits": [h.to_dict() for h in result.hits], "timings_ms": result.timings_ms}

    return app


app = create_app()
