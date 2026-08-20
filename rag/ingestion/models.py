"""Intermediate representation (IR) produced by parsing + cleaning."""

from pathlib import Path

from pydantic import BaseModel, Field


class ParseStats(BaseModel):
    num_pages: int = 0
    num_tables: int = 0
    num_headings: int = 0
    ocr_used: bool = False
    parse_seconds: float = 0.0
    boilerplate_lines_removed: int = 0


class Chunk(BaseModel):
    """A retrieval unit. Children (~512 tok) are embedded and searched;
    parents (~2048 tok) are what the LLM reads after retrieval."""

    chunk_id: str
    doc_id: str
    parent_id: str | None = None
    kind: str = "child"  # child | parent
    text: str
    embed_text: str = ""  # breadcrumb + contextual summary + text
    section_path: list[str] = Field(default_factory=list)
    element_type: str = "text"  # text | table
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0
    seq: int = 0  # order within document


class ParsedDoc(BaseModel):
    """One document after parsing + cleaning.

    `markdown` is the cleaned, structure-preserving export used for preview
    and debugging. `docling_json_path` points to the cached DoclingDocument,
    which phase 3 chunking consumes directly (richer than the markdown).
    """

    doc_id: str
    source_path: Path
    doc_type: str
    title: str = ""
    markdown: str = ""
    docling_json_path: Path | None = None
    stats: ParseStats = Field(default_factory=ParseStats)
    warnings: list[str] = Field(default_factory=list)
