"""The assistant: retrieve, decide whether to answer, write the answer, check its citations."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .chunking import load_markdown_dir
from .config import Settings
from .embed import make_embedder
from .index import KnowledgeIndex
from .llm import LLM, REFUSAL, make_llm
from .rerank import make_reranker
from .retrieve import Hit, Retriever

_CITATION = re.compile(r"\[(\d+)\]")
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None      # "gate" = stopped before the model, "model" = the model declined
    citations: list[int]            # source numbers the answer cites (1 = first source)
    sources: list[Hit]              # the passages the model was given, in order
    warnings: list[str] = field(default_factory=list)
    trace: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question, "answer": self.answer, "refused": self.refused,
            "refusal_reason": self.refusal_reason, "citations": self.citations,
            "sources": [dict(h.to_dict(), n=n) for n, h in enumerate(self.sources, start=1)],
            "warnings": self.warnings, "trace": self.trace,
        }


def _numbers(text: str) -> set[str]:
    """Figures in a text, normalised so '1,000' equals '1000' and '4.50' equals '4.5'."""
    found: set[str] = set()
    for raw in _NUMBER.findall(text):
        value = raw.replace(",", "")
        if "." in value and ":" not in value:
            value = value.rstrip("0").rstrip(".")
        found.add(value)
    return found


def check_citations(text: str, sources: list[Hit]) -> tuple[str, list[int], list[str]]:
    """Validate the model's citations against what it was actually given.

    Returns the cleaned answer, the valid citation numbers and a list of warnings. This is
    a cheap consistency check, not proof that the answer is right: it catches a citation to
    a source that does not exist, an answer with no citation at all, and a figure that does
    not appear in any cited passage.
    """
    warnings: list[str] = []
    cited = sorted({int(n) for n in _CITATION.findall(text)})
    valid = [n for n in cited if 1 <= n <= len(sources)]
    if len(valid) != len(cited):
        invalid = [n for n in cited if n not in valid]
        text = _CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", text)
        text = re.sub(r"\s+([.,;])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
        warnings.append(f"Removed citations to sources that were not provided: {invalid}")
    if not valid:
        warnings.append("The answer cites no source. Treat it as unverified.")
        return text, valid, warnings

    cited_text = " ".join(sources[n - 1].chunk.embed_text for n in valid)
    missing = sorted(_numbers(_CITATION.sub("", text)) - _numbers(cited_text))
    if missing:
        warnings.append(f"Figures not found in the cited sources: {', '.join(missing)}")
    return text, valid, warnings


class Assistant:
    def __init__(self, retriever: Retriever, llm: LLM, final_k: int = 4, min_relevance: float = float("-inf")) -> None:
        self.retriever = retriever
        self.llm = llm
        self.final_k = final_k
        self.min_relevance = min_relevance

    def ask(self, question: str) -> Answer:
        started = time.perf_counter()
        retrieval = self.retriever.search(question, k=self.final_k, mode="hybrid_rerank")
        hits = retrieval.hits
        trace: dict[str, object] = {"timings_ms": dict(retrieval.timings_ms), "model": self.llm.name,
                                    "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

        def finish(answer: Answer) -> Answer:
            trace["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000, 2)  # type: ignore[index]
            return answer

        # Relevance gate. Refusing here, before the model is called, costs nothing and does
        # not depend on the model obeying an instruction to say "I don't know".
        top_score = hits[0].rerank_score if hits else None
        trace["top_rerank_score"] = None if top_score is None else round(top_score, 3)
        trace["min_relevance"] = None if self.min_relevance == float("-inf") else self.min_relevance
        if not hits or (top_score is not None and top_score < self.min_relevance):
            return finish(Answer(question, REFUSAL, True, "gate", [], hits, [], trace))

        llm_started = time.perf_counter()
        reply = self.llm.answer(question, [h.chunk.embed_text for h in hits])
        trace["timings_ms"]["llm"] = round((time.perf_counter() - llm_started) * 1000, 2)  # type: ignore[index]
        trace.update(prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens,
                     cost_usd=round(reply.cost_usd(self.llm.name), 6))

        if "could not find this in the documents" in reply.text.lower():
            return finish(Answer(question, REFUSAL, True, "model", [], hits, [], trace))

        text, citations, warnings = check_citations(reply.text, hits)
        return finish(Answer(question, text, False, None, citations, hits, warnings, trace))


def build_assistant(settings: Settings, in_memory: bool = False) -> Assistant:
    """Wire everything from settings. `in_memory=True` skips the on-disk index (tests)."""
    chunks = load_markdown_dir(settings.docs_dir)
    if not chunks:
        raise RuntimeError(f"No Markdown documents found in {settings.docs_dir}")
    embedder = make_embedder(settings.embedder, settings.embedding_model)
    # Each embedder gets its own folder: vectors from different models cannot be mixed.
    index_dir = None if in_memory else settings.index_dir / re.sub(r"[^a-zA-Z0-9]+", "-", embedder.name)
    index = KnowledgeIndex.open(chunks, embedder, index_dir, settings.qdrant_url)
    reranker = make_reranker(settings.reranker, settings.reranker_model)
    retriever = Retriever(index, embedder, reranker, settings.candidates, settings.rerank_top_n)
    llm = make_llm(settings.llm_provider, settings.openai_model, settings.openai_api_key,
                   settings.openai_base_url, settings.max_output_tokens)
    return Assistant(retriever, llm, settings.final_k, settings.min_relevance)
