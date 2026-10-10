import numpy as np
import pytest

from rag.chunking import Chunk, chunk_markdown
from rag.embed import HashEmbedder, make_embedder
from rag.index import KnowledgeIndex, fingerprint
from rag.rerank import LexicalReranker, make_reranker
from rag.retrieve import RRF_K, Retriever, reciprocal_rank_fusion


def test_rrf_scores_use_ranks_only():
    fused = reciprocal_rank_fusion([[7, 3], [3, 9]])
    assert fused[3] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert fused[7] == pytest.approx(1 / (RRF_K + 1))
    assert max(fused, key=fused.get) == 3  # found by both searches beats first place in one


def test_rrf_of_nothing_is_empty():
    assert reciprocal_rank_fusion([[], []]) == {}


def test_hash_embedder_is_deterministic_and_normalised():
    a, b = HashEmbedder().embed_query("parcel label"), HashEmbedder().embed_query("parcel label")
    assert np.array_equal(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0)
    assert HashEmbedder().embed_passages([]).shape == (0, 256)


def test_unknown_component_names_are_rejected():
    with pytest.raises(ValueError):
        make_embedder("word2vec", "")
    with pytest.raises(ValueError):
        make_reranker("magic", "")
    assert make_reranker("none", "") is None


def test_lexical_reranker_scores_share_of_question_words():
    scores = LexicalReranker().score("webhook retries", ["webhook retries six times", "webhook events", "labels"])
    assert scores == [1.0, 0.5, 0.0]
    assert LexicalReranker().score("the of", ["anything"]) == [0.0]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_every_mode_finds_the_status_code_passage(assistant, mode):
    result = assistant.retriever.search("What does PU-204 mean?", k=3, mode=mode)
    assert result.hits[0].chunk.key == "tracking-status-codes#pickup-and-depot-codes"
    assert len(result.hits) == 3


def test_hybrid_records_where_each_passage_ranked(assistant):
    hit = assistant.retriever.search("webhook signature HMAC", k=1, mode="hybrid").hits[0]
    assert hit.chunk.key == "webhooks#verifying-the-signature"
    assert hit.keyword_rank == 1 and hit.dense_rank is not None and hit.rerank_score is None


def test_rerank_mode_orders_by_rerank_score(assistant):
    hits = assistant.retriever.search("How many times is a webhook retried?", k=4).hits
    scores = [h.rerank_score for h in hits]
    assert all(s is not None for s in scores) and scores == sorted(scores, reverse=True)


def test_search_is_repeatable(assistant):
    first = [h.chunk.key for h in assistant.retriever.search("return label validity", k=5).hits]
    assert first == [h.chunk.key for h in assistant.retriever.search("return label validity", k=5).hits]


def test_invalid_mode_raises(assistant):
    with pytest.raises(ValueError):
        assistant.retriever.search("x", mode="telepathy")


def test_query_with_no_known_words_still_returns_dense_results(assistant):
    result = assistant.retriever.search("qwertyuiop", k=2, mode="hybrid")
    assert all(h.keyword_rank is None for h in result.hits)


class CountingEmbedder(HashEmbedder):
    def __init__(self) -> None:
        self.calls = 0

    def embed_passages(self, texts):
        self.calls += len(texts)
        return super().embed_passages(texts)


def _chunks(text: str) -> list[Chunk]:
    return chunk_markdown("doc", f"# Doc\n\n## One\n{text}\n\n## Two\nLabels are valid for 14 days.")


def test_index_is_reused_from_disk_and_rebuilt_when_documents_change(tmp_path):
    embedder = CountingEmbedder()
    index = KnowledgeIndex.open(_chunks("Pickups before 12:00."), embedder, tmp_path)
    assert embedder.calls == 2
    index.close()

    index = KnowledgeIndex.open(_chunks("Pickups before 12:00."), embedder, tmp_path)
    assert embedder.calls == 2, "an unchanged index must not be embedded again"
    retriever = Retriever(index, embedder, None)
    assert retriever.search("label validity", k=1, mode="dense").hits[0].chunk.section == "Two"
    index.close()

    index = KnowledgeIndex.open(_chunks("Pickups before 10:00."), embedder, tmp_path)
    assert embedder.calls == 4, "edited documents must trigger a rebuild"
    index.close()


def test_fingerprint_depends_on_text_and_embedder():
    a, b = _chunks("alpha"), _chunks("beta")
    assert fingerprint(a, "m1") != fingerprint(b, "m1")
    assert fingerprint(a, "m1") != fingerprint(a, "m2")
    assert fingerprint(a, "m1") == fingerprint(_chunks("alpha"), "m1")


def test_weighted_fusion_rescales_each_list_before_mixing():
    from rag.retrieve import weighted_fusion

    fused = weighted_fusion([(1, 10.0), (2, 5.0)], [(2, 0.9), (3, 0.5)], dense_weight=0.25)
    assert fused == {1: pytest.approx(0.75), 2: pytest.approx(0.25), 3: pytest.approx(0.0)}


def test_weighted_fusion_edge_cases():
    from rag.retrieve import weighted_fusion

    assert weighted_fusion([], [], 0.5) == {}
    assert weighted_fusion([(4, 3.0)], [], 0.5) == {4: pytest.approx(0.5)}  # a lone candidate is the best one
    ranking = weighted_fusion([(1, 9.0), (2, 1.0)], [(2, 0.8), (1, 0.2)], dense_weight=1.0)
    assert max(ranking, key=ranking.get) == 2  # weight 1.0 = meaning search alone


def test_retriever_can_use_weighted_fusion(assistant):
    retriever = Retriever(assistant.retriever.index, assistant.retriever.embedder, None, dense_weight=0.5)
    hits = retriever.search("customs hold CU-301", k=3, mode="hybrid").hits
    assert hits[0].chunk.key in {"customs-international#customs-holds", "tracking-status-codes#exception-codes"}
    assert 0.0 <= hits[-1].fused_score <= hits[0].fused_score <= 1.0
