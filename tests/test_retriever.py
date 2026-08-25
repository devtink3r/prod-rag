"""Retrieval pipeline logic with fake components (no models, no services)."""

from dataclasses import dataclass

from rag.config import load_config
from rag.retrieval.retriever import Retriever, ScoredChunk


@dataclass
class FakeReranker:
    scores: dict[str, float]

    def rerank(self, query, texts):
        return [self.scores.get(t, 0.0) for t in texts]


class FakeRegistry:
    def __init__(self, parents):
        self.parents = parents

    def get_parents(self, ids):
        return {k: v for k, v in self.parents.items() if k in ids}


def chunk(cid, parent_id, text) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid, parent_id=parent_id, doc_id="d1", text=text,
        section_path=["Rule 1: X"], page_start=1, page_end=2,
        source_path="/a.pdf", title="Doc",
    )


def make_retriever(chunks, scores, parents):
    cfg = load_config()
    r = Retriever(cfg, embedder=None, store=None,
                  reranker=FakeReranker(scores), registry=FakeRegistry(parents))
    r._hybrid_search = lambda q: list(chunks)
    return r


PARENTS = {
    "p1": {"parent_id": "p1", "doc_id": "d1", "text": "PARENT ONE FULL SECTION",
           "section_path": ["Rule 1: X"], "page_start": 1, "page_end": 3},
}


def test_rerank_orders_and_expands_to_parent():
    chunks = [chunk("c1", "p1", "low relevance"), chunk("c2", "p1", "high relevance")]
    r = make_retriever(chunks, {"low relevance": 0.3, "high relevance": 0.9}, PARENTS)
    result = r.retrieve("q")
    assert not result.no_answer
    assert result.top_score == 0.9
    # both children share p1 -> one merged block with best score, 2 matches
    assert len(result.blocks) == 1
    assert result.blocks[0].text == "PARENT ONE FULL SECTION"
    assert result.blocks[0].score == 0.9
    assert result.blocks[0].matched_chunks == 2
    assert result.blocks[0].page_end == 3


def test_score_floor_triggers_no_answer():
    chunks = [chunk("c1", "p1", "irrelevant")]
    r = make_retriever(chunks, {"irrelevant": 0.02}, PARENTS)
    result = r.retrieve("q")
    assert result.no_answer and result.blocks == []
    assert result.top_score == 0.02


def test_missing_parent_falls_back_to_child_text():
    chunks = [chunk("c1", "p-unknown", "orphan text")]
    r = make_retriever(chunks, {"orphan text": 0.8}, PARENTS)
    result = r.retrieve("q")
    assert result.blocks[0].text == "orphan text"


def test_token_budget_drops_lowest_ranked():
    cfg = load_config()
    huge = "x" * (cfg.retrieval.context_token_budget * 4)
    chunks = [chunk("c1", None, huge), chunk("c2", None, "small but lower score")]
    r = make_retriever(chunks, {huge: 0.9, "small but lower score": 0.5}, {})
    result = r.retrieve("q")
    assert len(result.blocks) == 1 and result.blocks[0].score == 0.9
