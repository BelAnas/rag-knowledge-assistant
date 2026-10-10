"""Answer evaluation on the handbook question set (86 answerable, 14 not answerable).

    python -m evals.answers                      # offline writer, no key
    LLM_PROVIDER=openai python -m evals.answers  # a real model
    python -m evals.answers --gate-sweep         # show how the refusal threshold trades off

For an answerable question the reply counts as correct when it (1) is not a refusal,
(2) cites a passage that really contains the answer and (3) states the expected fact.
For a question the handbook does not cover, the only correct reply is a refusal.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics

from rag.answer import Answer, Assistant, build_assistant
from rag.config import get_settings

from .datasets import ROOT, Query, load_handbook


def has_fact(answer: str, fact: str) -> bool:
    """`fact` lists accepted wordings separated by '|'. Whole words only, so 'no' does not match 'not'."""
    text = answer.lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(option)}(?![a-z0-9])", text) for option in fact.split("|"))


def judge(query: Query, answer: Answer) -> dict[str, object]:
    cited_keys = [answer.sources[n - 1].chunk.key for n in answer.citations]
    row: dict[str, object] = {
        "id": query.id, "type": query.type, "question": query.text, "answer": answer.answer,
        "refused": answer.refused, "refusal_reason": answer.refusal_reason, "cited": cited_keys,
        "warnings": answer.warnings, "top_rerank_score": answer.trace.get("top_rerank_score"),
    }
    if query.gold:
        row["cited_gold"] = any(key in query.gold for key in cited_keys)
        row["facts_ok"] = all(has_fact(answer.answer, fact) for fact in query.facts)
        row["correct"] = (not answer.refused) and bool(row["cited_gold"]) and bool(row["facts_ok"])
    else:
        row["correct"] = answer.refused
    return row


def gate_sweep(assistant: Assistant, queries: list[Query]) -> None:
    """Print, for several thresholds, how many good questions the gate would wrongly refuse
    and how many out-of-scope questions it would stop. This is how MIN_RELEVANCE was chosen."""
    scores = []
    for query in queries:
        hits = assistant.retriever.search(query.text, k=1).hits
        scores.append((hits[0].rerank_score if hits and hits[0].rerank_score is not None else float("-inf"), bool(query.gold)))
    answerable = [s for s, good in scores if good]
    unanswerable = [s for s, good in scores if not good]
    print(f"top rerank score, answerable:   min {min(answerable):.2f}  median {statistics.median(answerable):.2f}")
    print(f"top rerank score, unanswerable: max {max(unanswerable):.2f}  median {statistics.median(unanswerable):.2f}")
    print("\n| Threshold | Answerable wrongly refused | Unanswerable stopped by the gate |")
    print("|---|---|---|")
    for threshold in (-6, -4, -2, -1, 0, 1, 2, 3, 4):
        wrong = sum(s < threshold for s in answerable)
        stopped = sum(s < threshold for s in unanswerable)
        print(f"| {threshold} | {wrong} / {len(answerable)} | {stopped} / {len(unanswerable)} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="only the first N answerable and N unanswerable questions")
    parser.add_argument("--gate-sweep", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    _, queries = load_handbook(settings.docs_dir)
    assistant = build_assistant(settings)
    if args.gate_sweep:
        gate_sweep(assistant, queries)
        return
    if args.limit:
        queries = [q for q in queries if q.gold][: args.limit] + [q for q in queries if not q.gold][: args.limit]

    rows, tokens_in, tokens_out, cost, latencies = [], 0, 0, 0.0, []
    for n, query in enumerate(queries, start=1):
        answer = assistant.ask(query.text)
        rows.append(judge(query, answer))
        tokens_in += int(answer.trace["prompt_tokens"])       # type: ignore[call-overload]
        tokens_out += int(answer.trace["completion_tokens"])  # type: ignore[call-overload]
        cost += float(answer.trace["cost_usd"])               # type: ignore[arg-type]
        latencies.append(float(answer.trace["timings_ms"]["total"]))  # type: ignore[index]
        print(f"\r  {n}/{len(queries)}", end="", flush=True)
    print()

    good = [r for r in rows if r["type"] != "unanswerable"]
    bad = [r for r in rows if r["type"] == "unanswerable"]
    summary = {
        "llm": assistant.llm.name, "reranker": settings.reranker_model if settings.reranker == "cross-encoder" else settings.reranker,
        "embedder": assistant.retriever.embedder.name, "min_relevance": settings.min_relevance,
        "answerable": len(good),
        "answerable_correct": sum(bool(r["correct"]) for r in good),
        "answerable_cited_right_passage": sum(bool(r["cited_gold"]) for r in good),
        "answerable_fact_present": sum(bool(r["facts_ok"]) and not r["refused"] for r in good),
        "answerable_wrongly_refused": sum(bool(r["refused"]) for r in good),
        "unanswerable": len(bad),
        "unanswerable_refused": sum(bool(r["refused"]) for r in bad),
        "unanswerable_refused_by_gate": sum(r["refusal_reason"] == "gate" for r in bad),
        "answers_with_warnings": sum(bool(r["warnings"]) for r in rows),
        "prompt_tokens": tokens_in, "completion_tokens": tokens_out, "cost_usd": round(cost, 5),
        "latency_ms_median": round(statistics.median(latencies), 1),
    }
    for key, value in summary.items():
        print(f"  {key}: {value}")
    failures = [r for r in rows if not r["correct"]]
    if failures:
        print("\nNot correct:")
        for r in failures:
            print(f"  {r['id']} {r['question']}\n     -> {str(r['answer'])[:160]}  cited={r['cited']}")

    name = re.sub(r"[^a-zA-Z0-9.]+", "-", assistant.llm.name)
    suffix = "" if settings.embedder == "fastembed" else f"-{settings.embedder}"
    out = ROOT / "evals" / f"report-answers-{name}{suffix}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
