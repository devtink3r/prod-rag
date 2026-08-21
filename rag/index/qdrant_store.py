"""Qdrant collection management + upsert/delete for child chunks.

Hybrid schema: named dense vector ("dense", cosine) + named sparse vector
("sparse", BGE-M3 lexical weights). Parents live in Postgres, not here.
"""

from rag.config import VectorStoreConfig
from rag.index.embedder import DENSE_DIM, Embedding
from rag.ingestion.models import Chunk


class QdrantStore:
    def __init__(self, cfg: VectorStoreConfig):
        from qdrant_client import QdrantClient

        self.cfg = cfg
        self.client = QdrantClient(url=cfg.url, timeout=30)

    def ensure_collection(self, embedding_model: str, pipeline_version: int) -> None:
        from qdrant_client import models as qm

        if self.client.collection_exists(self.cfg.collection):
            return
        self.client.create_collection(
            collection_name=self.cfg.collection,
            vectors_config={"dense": qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams()},
        )
        self.client.create_payload_index(
            self.cfg.collection, "doc_id", qm.PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            self.cfg.collection, "doc_type", qm.PayloadSchemaType.KEYWORD
        )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[Embedding], extra: dict) -> None:
        import uuid

        from qdrant_client import models as qm

        points = []
        for chunk, emb in zip(chunks, embeddings):
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "parent_id": chunk.parent_id,
                "text": chunk.text,
                "section_path": chunk.section_path,
                "element_type": chunk.element_type,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "seq": chunk.seq,
                **extra,
            }
            points.append(
                qm.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.chunk_id)),
                    vector={
                        "dense": emb.dense,
                        "sparse": qm.SparseVector(
                            indices=emb.sparse_indices, values=emb.sparse_values
                        ),
                    },
                    payload=payload,
                )
            )
        self.client.upsert(self.cfg.collection, points=points, wait=True)

    def delete_doc(self, doc_id: str) -> None:
        from qdrant_client import models as qm

        self.client.delete(
            self.cfg.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
                )
            ),
            wait=True,
        )

    def count(self) -> int:
        return self.client.count(self.cfg.collection, exact=True).count
