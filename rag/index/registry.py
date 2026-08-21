"""Postgres document registry: the brain of incremental ingestion.

Tracks every source file's content hash, status, and pipeline version, and
stores parent chunks (children live in Qdrant with their vectors).
"""

import json
from dataclasses import dataclass

from rag.ingestion.models import Chunk

_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS {s};
CREATE TABLE IF NOT EXISTS {s}.documents (
    source_path      text PRIMARY KEY,
    doc_id           text NOT NULL,
    content_hash     text NOT NULL,
    status           text NOT NULL DEFAULT 'pending',
    error            text,
    title            text,
    doc_type         text,
    num_pages        int,
    chunk_count      int,
    pipeline_version int,
    last_indexed_at  timestamptz,
    updated_at       timestamptz DEFAULT now()
);
CREATE TABLE IF NOT EXISTS {s}.parents (
    parent_id    text PRIMARY KEY,
    doc_id       text NOT NULL,
    text         text NOT NULL,
    section_path jsonb DEFAULT '[]',
    page_start   int,
    page_end     int,
    seq          int
);
CREATE INDEX IF NOT EXISTS parents_doc_id_idx ON {s}.parents (doc_id);
"""


@dataclass
class RegistryEntry:
    source_path: str
    doc_id: str
    content_hash: str
    status: str
    pipeline_version: int


class Registry:
    def __init__(self, dsn: str, schema: str = "rag"):
        import psycopg

        self.schema = schema
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute(_SCHEMA_SQL.format(s=schema))

    def entries(self) -> dict[str, RegistryEntry]:
        rows = self.conn.execute(
            f"SELECT source_path, doc_id, content_hash, status, pipeline_version "
            f"FROM {self.schema}.documents"
        ).fetchall()
        return {r[0]: RegistryEntry(*r) for r in rows}

    def mark(self, source_path: str, doc_id: str, content_hash: str, status: str,
             error: str | None = None, **fields) -> None:
        cols = {"doc_id": doc_id, "content_hash": content_hash, "status": status,
                "error": error, **fields}
        assignments = ", ".join(f"{k} = %s" for k in cols)
        self.conn.execute(
            f"INSERT INTO {self.schema}.documents (source_path, {', '.join(cols)}) "
            f"VALUES (%s, {', '.join(['%s'] * len(cols))}) "
            f"ON CONFLICT (source_path) DO UPDATE SET {assignments}, updated_at = now()",
            [source_path, *cols.values(), *cols.values()],
        )

    def delete_doc(self, source_path: str) -> None:
        self.conn.execute(
            f"DELETE FROM {self.schema}.documents WHERE source_path = %s", [source_path]
        )

    def replace_parents(self, old_doc_id: str | None, parents: list[Chunk]) -> None:
        with self.conn.transaction():
            if old_doc_id:
                self.conn.execute(
                    f"DELETE FROM {self.schema}.parents WHERE doc_id = %s", [old_doc_id]
                )
            for p in parents:
                self.conn.execute(
                    f"INSERT INTO {self.schema}.parents "
                    f"(parent_id, doc_id, text, section_path, page_start, page_end, seq) "
                    f"VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    f"ON CONFLICT (parent_id) DO UPDATE SET text = EXCLUDED.text",
                    [p.chunk_id, p.doc_id, p.text, json.dumps(p.section_path),
                     p.page_start, p.page_end, p.seq],
                )

    def get_parents(self, parent_ids: list[str]) -> dict[str, dict]:
        if not parent_ids:
            return {}
        rows = self.conn.execute(
            f"SELECT parent_id, doc_id, text, section_path, page_start, page_end "
            f"FROM {self.schema}.parents WHERE parent_id = ANY(%s)",
            [parent_ids],
        ).fetchall()
        return {
            r[0]: {"parent_id": r[0], "doc_id": r[1], "text": r[2],
                   "section_path": r[3], "page_start": r[4], "page_end": r[5]}
            for r in rows
        }

    def delete_parents(self, doc_id: str) -> None:
        self.conn.execute(f"DELETE FROM {self.schema}.parents WHERE doc_id = %s", [doc_id])
