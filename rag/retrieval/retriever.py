"""Retrieval pipeline: hybrid search -> rerank -> score floor -> parent
expansion -> deduplicated, relevance-ordered context blocks.

Components are injected so the pipeline logic is testable without models
or services (see tests/test_retriever.py).
"""

from dataclasses import dataclass, field

from rag.config import Config


@dataclass
class ScoredChunk:
    """A child chunk as returned by hybrid search + reranking."""

    chunk_id: str
    parent_id: str | None
    doc_id: str
    text: str
    section_path: list[str]
    page_start: int | None
    page_end: int | None
    source_path: str
    title: str
    rerank_score: float = 0.0


@dataclass
class ContextBlock:
    """A parent-level block handed to the LLM, with citation metadata."""

    text: str
    score: float
    doc_id: str
    title: str
    source_path: str
    section_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    matched_chunks: int = 1


@dataclass
class RetrievalResult:
    blocks: list[ContextBlock]
    no_answer: bool  # best rerank score below the floor -> refuse, don't guess
    top_score: float


class Retriever:
    def __init__(self, cfg: Config, embedder, store, reranker, registry):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.registry = registry

    def _hybrid_search(self, query: str) -> list[ScoredChunk]:
        from qdrant_client import models as qm

        emb = self.embedder.encode_query(query)
        k = self.cfg.retrieval.fused_top_k
        res = self.store.client.query_points(
            self.store.cfg.collection,
            prefetch=[
                qm.Prefetch(query=emb.dense, using="dense", limit=k),
                qm.Prefetch(
                    query=qm.SparseVector(indices=emb.sparse_indices, values=emb.sparse_values),
                    using="sparse",
                    limit=k,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return [
            ScoredChunk(
                chunk_id=p.payload["chunk_id"],
                parent_id=p.payload.get("parent_id"),
                doc_id=p.payload["doc_id"],
                text=p.payload["text"],
                section_path=p.payload.get("section_path") or [],
                page_start=p.payload.get("page_start"),
                page_end=p.payload.get("page_end"),
                source_path=p.payload.get("source_path", ""),
                title=p.payload.get("title", ""),
            )
            for p in res.points
        ]

    def retrieve(self, query: str) -> RetrievalResult:
        candidates = self._hybrid_search(query)
        if not candidates:
            return RetrievalResult([], no_answer=True, top_score=0.0)

        scores = self.reranker.rerank(query, [c.text for c in candidates])
        for c, s in zip(candidates, scores):
            c.rerank_score = s
        candidates.sort(key=lambda c: c.rerank_score, reverse=True)

        floor = self.cfg.retrieval.rerank_score_floor
        top = [c for c in candidates if c.rerank_score >= floor][: self.cfg.retrieval.rerank_top_n]
        if not top:
            return RetrievalResult([], no_answer=True, top_score=candidates[0].rerank_score)

        blocks = self._expand_to_parents(top)
        blocks = self._apply_token_budget(blocks)
        return RetrievalResult(blocks, no_answer=False, top_score=top[0].rerank_score)

    def _expand_to_parents(self, chunks: list["ScoredChunk"]) -> list[ContextBlock]:
        """Swap children for parent sections; merge children sharing a parent."""
        parent_ids = [c.parent_id for c in chunks if c.parent_id]
        parents = self.registry.get_parents(parent_ids) if parent_ids else {}

        blocks: dict[str, ContextBlock] = {}
        order: list[str] = []
        for c in chunks:
            parent = parents.get(c.parent_id) if c.parent_id else None
            key = c.parent_id if parent else c.chunk_id
            if key in blocks:  # sibling hit: keep best score, count the match
                blocks[key].score = max(blocks[key].score, c.rerank_score)
                blocks[key].matched_chunks += 1
                continue
            blocks[key] = ContextBlock(
                text=parent["text"] if parent else c.text,
                score=c.rerank_score,
                doc_id=c.doc_id,
                title=c.title,
                source_path=c.source_path,
                section_path=list(parent["section_path"]) if parent else c.section_path,
                page_start=parent["page_start"] if parent else c.page_start,
                page_end=parent["page_end"] if parent else c.page_end,
            )
            order.append(key)
        result = [blocks[k] for k in order]
        result.sort(key=lambda b: b.score, reverse=True)
        return result

    def _apply_token_budget(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        budget = self.cfg.retrieval.context_token_budget
        kept: list[ContextBlock] = []
        used = 0
        for b in blocks:  # best-first; drop lowest-ranked when budget runs out
            tokens = max(1, len(b.text) // 4)
            if used + tokens > budget and kept:
                continue
            kept.append(b)
            used += tokens
        return kept


def build_retriever(cfg: Config, secrets) -> Retriever:
    """Wire real components (loads models — slow on first call)."""
    from rag.index.embedder import BgeM3Embedder
    from rag.index.qdrant_store import QdrantStore
    from rag.index.registry import Registry
    from rag.retrieval.reranker import BgeReranker

    return Retriever(
        cfg,
        embedder=BgeM3Embedder(cfg.embedding),
        store=QdrantStore(cfg.vector_store),
        reranker=BgeReranker(cfg.retrieval),
        registry=Registry(secrets.postgres_dsn, cfg.registry.schema_name),
    )
