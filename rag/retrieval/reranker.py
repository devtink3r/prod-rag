"""Cross-encoder reranking: reads query+chunk together, far more precise than
embedding similarity. bge-reranker-v2-m3 pairs naturally with BGE-M3.
"""

from rag.config import RetrievalConfig


class BgeReranker:
    def __init__(self, cfg: RetrievalConfig):
        from rag.config import export_hf_token

        export_hf_token()
        from FlagEmbedding import FlagReranker

        self.cfg = cfg
        self.model = FlagReranker(cfg.reranker_model, use_fp16=True)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Returns a 0..1 relevance score per text (sigmoid-normalized)."""
        if not texts:
            return []
        scores = self.model.compute_score([[query, t] for t in texts], normalize=True)
        if isinstance(scores, float):  # single pair returns a bare float
            scores = [scores]
        return [float(s) for s in scores]
