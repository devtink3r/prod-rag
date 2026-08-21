"""BGE-M3 embeddings: one forward pass yields dense (1024-dim) + sparse
(lexical weights) vectors — exactly what Qdrant hybrid search consumes.
"""

from dataclasses import dataclass

from rag.config import EmbeddingConfig

DENSE_DIM = 1024


@dataclass
class Embedding:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class BgeM3Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        from FlagEmbedding import BGEM3FlagModel

        device = None if cfg.device == "auto" else cfg.device
        self.cfg = cfg
        kwargs = {"use_fp16": device != "cpu"}
        if device:
            try:
                self.model = BGEM3FlagModel(cfg.model, devices=[device], **kwargs)
                return
            except TypeError:
                kwargs["device"] = device
        self.model = BGEM3FlagModel(cfg.model, **kwargs)

    def encode(self, texts: list[str]) -> list[Embedding]:
        out = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            max_length=self.cfg.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        results: list[Embedding] = []
        for dense, lex in zip(out["dense_vecs"], out["lexical_weights"]):
            indices = [int(k) for k in lex.keys()]
            values = [float(v) for v in lex.values()]
            results.append(Embedding(dense.tolist(), indices, values))
        return results

    def encode_query(self, text: str) -> Embedding:
        return self.encode([text])[0]
