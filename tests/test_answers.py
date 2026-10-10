import pytest

from rag.answer import Assistant, check_citations
from rag.chunking import Chunk
from rag.llm import REFUSAL, LLMReply, MockLLM, build_user_message, make_llm
from rag.retrieve import Hit


def _hits(*texts: str) -> list[Hit]:
    return [Hit(Chunk(f"d#{i}", "d", "Doc", f"S{i}", t), i) for i, t in enumerate(texts)]


class ScriptedLLM:
    """A stand-in model that returns a fixed reply and records what it was sent."""

    name = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply, self.calls = reply, []

    def answer(self, question, sources):
        self.calls.append((question, sources))
        return LLMReply(self.reply, prompt_tokens=100, completion_tokens=20)


def _with(assistant: Assistant, llm, min_relevance: float = 0.5) -> Assistant:
    return Assistant(assistant.retriever, llm, final_k=4, min_relevance=min_relevance)


# ---- citation checks ----------------------------------------------------------------------

def test_valid_citations_pass_without_warnings():
    text, cited, warnings = check_citations("Labels last 14 days [1].", _hits("A label is valid for 14 days."))
    assert (text, cited, warnings) == ("Labels last 14 days [1].", [1], [])


def test_citation_to_a_source_that_was_not_given_is_removed():
    text, cited, warnings = check_citations("Valid 14 days [1]. Free reprints [7].", _hits("valid 14 days"))
    assert "[7]" not in text and "[1]" in text and cited == [1]
    assert "not provided" in warnings[0]


def test_answer_without_citation_is_flagged():
    _, cited, warnings = check_citations("Labels last 14 days.", _hits("valid 14 days"))
    assert cited == [] and "cites no source" in warnings[0]


def test_figure_that_is_not_in_the_cited_source_is_flagged():
    _, _, warnings = check_citations("The fee is 25 EUR [1].", _hits("The fee is 15 EUR.", "Other: 25 EUR."))
    assert warnings == ["Figures not found in the cited sources: 25"]


def test_figures_are_compared_after_normalising_format():
    hits = _hits("More than 1,000 parcels cost 4.50 EUR before 16:00.")
    _, _, warnings = check_citations("Over 1000 parcels: 4.5 EUR, cut-off 16:00 [1].", hits)
    assert warnings == []


def test_figures_in_the_section_heading_count_as_source_text():
    hits = [Hit(Chunk("d#p", "d", "Priority 1 incidents", "Targets", "Reply in 30 minutes."), 0)]
    assert check_citations("Priority 1 gets a reply in 30 minutes [1].", hits)[2] == []


# ---- the assistant ------------------------------------------------------------------------

def test_answer_quotes_the_handbook_and_cites_it(assistant):
    answer = assistant.ask("What does PU-204 mean?")
    assert not answer.refused and answer.citations == [1]
    assert "premises were closed" in answer.answer
    assert answer.sources[0].chunk.key == "tracking-status-codes#pickup-and-depot-codes"
    assert answer.warnings == []


def test_out_of_scope_question_is_refused_before_the_model_is_called(assistant):
    llm = ScriptedLLM("Paris [1].")
    answer = _with(assistant, llm).ask("Who won the football world cup in 2018?")
    assert answer.refused and answer.refusal_reason == "gate" and answer.answer == REFUSAL
    assert llm.calls == [], "the gate must stop the request before any model call"
    assert answer.trace["prompt_tokens"] == 0


def test_model_refusal_is_reported_as_a_refusal(assistant):
    answer = _with(assistant, ScriptedLLM("I could not find this in the documents. [1]")).ask("What does PU-204 mean?")
    assert answer.refused and answer.refusal_reason == "model" and answer.citations == []


def test_model_gets_at_most_final_k_numbered_sources_with_headings(assistant):
    llm = ScriptedLLM("Closed premises [1].")
    _with(assistant, llm).ask("What does PU-204 mean?")
    question, sources = llm.calls[0]
    assert question == "What does PU-204 mean?" and 1 <= len(sources) <= 4
    assert sources[0].startswith("Tracking and status codes: Pickup and depot codes\n")


def test_hallucinated_citation_is_stripped_and_flagged(assistant):
    answer = _with(assistant, ScriptedLLM("The premises were closed [1][9].")).ask("What does PU-204 mean?")
    assert answer.citations == [1] and "[9]" not in answer.answer and answer.warnings


def test_tokens_and_timings_are_traced(assistant):
    answer = _with(assistant, ScriptedLLM("Closed [1].")).ask("What does PU-204 mean?")
    assert answer.trace["prompt_tokens"] == 100 and answer.trace["completion_tokens"] == 20
    assert {"keyword", "dense", "rerank", "llm", "total"} <= set(answer.trace["timings_ms"])


def test_gate_is_off_when_threshold_is_minus_infinity(assistant):
    llm = ScriptedLLM(REFUSAL)
    answer = _with(assistant, llm, float("-inf")).ask("Who won the football world cup in 2018?")
    assert len(llm.calls) == 1 and answer.refusal_reason == "model"


def test_answer_serialises_to_plain_json(assistant):
    import json

    data = json.loads(json.dumps(assistant.ask("How long is a return label valid?").to_dict()))
    assert data["sources"][0]["n"] == 1 and "30 days" in data["answer"]


# ---- answer writers -----------------------------------------------------------------------

def test_user_message_numbers_the_sources():
    message = build_user_message("Why?", ["first", "second"])
    assert "[1] first" in message and "[2] second" in message and message.endswith("Question: Why?")


def test_mock_writer_refuses_when_the_top_source_shares_no_word():
    assert MockLLM().answer("webhook retries", ["Title: S\nLabels are valid."]).text == REFUSAL
    assert MockLLM().answer("anything", []).text == REFUSAL


def test_cost_uses_the_price_table_and_is_zero_for_unknown_models():
    reply = LLMReply("x", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert reply.cost_usd("gpt-4o-mini") == pytest.approx(0.75)
    assert reply.cost_usd("llama3.2") == 0.0


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        make_llm("carrier-pigeon", "", "", "", 10)
