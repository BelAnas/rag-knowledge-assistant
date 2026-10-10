import json

import pytest
from fastapi.testclient import TestClient

from evals.answers import has_fact, judge
from evals.datasets import QUESTIONS, Query, load_handbook
from evals.retrieval import score_query
from rag.answer import Assistant
from rag.api import create_app


@pytest.fixture(scope="module")
def client(assistant):
    with TestClient(create_app(assistant)) as test_client:
        yield test_client


def test_health_reports_the_loaded_components(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 20 and body["passages"] == 85
    assert body["llm"] == "mock-extractive"


def test_ask_returns_answer_sources_and_trace(client):
    body = client.post("/ask", json={"question": "How much is the Saturday delivery surcharge?"}).json()
    assert "4.50 EUR" in body["answer"] and body["citations"] == [1] and not body["refused"]
    assert body["sources"][0]["doc_id"] == "service-levels"
    assert body["trace"]["timings_ms"]["total"] > 0


def test_ask_refuses_out_of_scope(client):
    body = client.post("/ask", json={"question": "What is the capital of Australia?"}).json()
    assert body["refused"] and body["refusal_reason"] == "gate"


@pytest.mark.parametrize("payload", [{"question": "a"}, {"question": "x" * 501}, {}, {"question": 5}])
def test_ask_rejects_bad_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_search_exposes_each_mode_and_validates_it(client):
    body = client.post("/search", json={"query": "customs hold CU-301", "mode": "bm25", "k": 2}).json()
    assert len(body["hits"]) == 2 and body["hits"][0]["keyword_rank"] == 1
    assert client.post("/search", json={"query": "x y", "mode": "nope"}).status_code == 422
    assert client.post("/search", json={"query": "x y", "k": 500}).status_code == 422


def test_documents_listing_and_detail(client):
    docs = client.get("/documents").json()
    assert len(docs) == 20 and {"doc_id", "title", "sections"} <= set(docs[0])
    detail = client.get("/documents/webhooks").json()
    assert detail["title"] == "Webhooks" and len(detail["sections"]) == 5
    assert client.get("/documents/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "Knowledge assistant" in response.text


def test_model_outage_becomes_a_502_without_internal_details(assistant):
    class Broken:
        name = "broken"

        def answer(self, question, sources):
            raise RuntimeError("secret-internal-detail")

    with TestClient(create_app(Assistant(assistant.retriever, Broken(), 4, 0.3))) as broken_client:
        response = broken_client.post("/ask", json={"question": "What does PU-204 mean?"})
    assert response.status_code == 502 and "secret-internal-detail" not in response.text


# ---- evaluation code ----------------------------------------------------------------------

def test_question_set_is_consistent_with_the_handbook(settings):
    chunks, queries = load_handbook(settings.docs_dir)
    keys = {c.key for c in chunks}
    assert len(queries) == 100 == len({q.id for q in queries})
    assert sum(1 for q in queries if not q.gold) == 14
    for query in queries:
        assert query.gold <= keys, f"{query.id} points to a passage that does not exist"
        assert bool(query.gold) == (query.type != "unanswerable")
        if query.gold:
            text = " ".join(c.text.lower() for c in chunks if c.key in query.gold)
            for fact in query.facts:
                assert has_fact(text, fact), f"{query.id}: expected fact '{fact}' is not in its own passage"


def test_question_file_is_valid_jsonl():
    for line in QUESTIONS.read_text().splitlines():
        assert {"id", "type", "question", "gold", "facts"} <= set(json.loads(line))


def test_score_query_metrics():
    perfect = score_query(["a", "b"], frozenset({"a"}))
    assert perfect == {"hit@1": 1.0, "hit@4": 1.0, "recall@10": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0}
    third = score_query(["x", "y", "a"], frozenset({"a"}))
    assert third["hit@1"] == 0.0 and third["hit@4"] == 1.0 and third["mrr@10"] == pytest.approx(1 / 3)
    assert third["ndcg@10"] == pytest.approx(0.5)
    miss = score_query(["x"], frozenset({"a", "b"}))
    assert miss["recall@10"] == 0.0 and miss["mrr@10"] == 0.0
    assert score_query(["a", "x"], frozenset({"a", "b"}))["recall@10"] == 0.5


def test_score_query_counts_a_split_section_once():
    assert score_query(["x", "x", "a"], frozenset({"a"}))["mrr@10"] == pytest.approx(1 / 2)


def test_has_fact_matches_whole_words_and_alternatives():
    assert has_fact("It costs 4.50 EUR.", "4.50")
    assert not has_fact("It costs 14.50 EUR.", "4.50")
    assert not has_fact("A notification is sent.", "no|cannot")
    assert has_fact("No, it cannot.", "no|cannot")
    assert has_fact("Reply within six tries", "6|six")


def test_judge_requires_right_passage_fact_and_no_refusal(assistant):
    query = Query("t", "What does PU-204 mean?", frozenset({"tracking-status-codes#pickup-and-depot-codes"}), "code", ("closed",))
    assert judge(query, assistant.ask(query.text))["correct"] is True
    wrong_gold = Query("t", query.text, frozenset({"returns#return-labels"}), "code", ("closed",))
    assert judge(wrong_gold, assistant.ask(query.text))["correct"] is False
    out_of_scope = Query("u", "What is the capital of Australia?", frozenset(), "unanswerable")
    assert judge(out_of_scope, assistant.ask(out_of_scope.text))["correct"] is True
