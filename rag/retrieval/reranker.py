"""Cross-encoder reranking: reads query+chunk together, far more precise than
embedding similarity. bge-reranker-v2-m3 pairs naturally with BGE-M3.

Implemented directly on transformers (AutoModelForSequenceClassification)
rather than FlagEmbedding's reranker wrapper, which breaks on newer
transformers versions (removed `prepare_for_model`).
"""

from rag.config import RetrievalConfig

_BATCH = 8
_MAX_LENGTH = 1024  # chunks are <=1024 tokens by design


class BgeReranker:
    def __init__(self, cfg: RetrievalConfig):
        from rag.config import export_hf_token

        export_hf_token()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.reranker_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.reranker_model)
        self.model.eval()

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Returns a 0..1 relevance score per text (sigmoid of the logit)."""
        if not texts:
            return []
        torch = self.torch
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), _BATCH):
                batch = texts[i : i + _BATCH]
                inputs = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation="only_second",
                    max_length=_MAX_LENGTH,
                    return_tensors="pt",
                )
                logits = self.model(**inputs).logits.squeeze(-1)
                scores.extend(torch.sigmoid(logits).tolist())
        return [float(s) for s in scores]
