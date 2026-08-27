"""Cross-encoder reranking: reads query+chunk together, far more precise than
embedding similarity. bge-reranker-v2-m3 pairs naturally with BGE-M3.

Implemented directly on transformers (AutoModelForSequenceClassification)
rather than FlagEmbedding's reranker wrapper, which breaks on newer
transformers versions (removed `prepare_for_model`).
"""

from rag.config import RetrievalConfig

class BgeReranker:
    def __init__(self, cfg: RetrievalConfig, device: str | None = None):
        from rag.config import export_hf_token, resolve_device

        export_hf_token()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        self.device = device or resolve_device("auto")[0]
        self.batch = 32 if self.device == "cuda" else 8
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.reranker_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.reranker_model)
        self.model.eval()
        if self.device == "cuda":
            self.model = self.model.half().to("cuda")
        elif cfg.rerank_quantize:
            try:  # int8 dynamic quantization: CPU speedup, small accuracy cost
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                pass  # unsupported build -> keep fp32

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Returns a 0..1 relevance score per text (sigmoid of the logit)."""
        if not texts:
            return []
        torch = self.torch
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch):
                batch = texts[i : i + self.batch]
                inputs = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation="only_second",
                    max_length=self.cfg.rerank_max_length,
                    return_tensors="pt",
                )
                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                logits = self.model(**inputs).logits.view(-1).float()
                scores.extend(torch.sigmoid(logits).tolist())
        return [float(s) for s in scores]
