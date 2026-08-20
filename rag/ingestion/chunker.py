"""Structure-aware chunking with parent-child hierarchy.

Why not Docling's HybridChunker: legal/regulatory documents carry their
structure in numbered rules ("3. Arrangements for...") that parse as list
items, not headings. This chunker treats heading changes AND rule starts as
section boundaries, then packs sections into token-bounded chunks.

Pipeline: DoclingDocument -> blocks -> sections -> children (~512 tok)
                                                 -> parents  (~2048 tok)
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag.config import Config
from rag.ingestion.cleaners import is_toc_line, normalize_text
from rag.ingestion.models import Chunk, ParsedDoc

RULE_RE = re.compile(r"^(\d{1,3})\.\s+(?=[A-Z])")
_SENT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\d])")


class TokenCounter:
    """HF tokenizer of the embedding model when available, else ~chars/4."""

    def __init__(self, model_name: str):
        self._tok = None
        try:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            pass

    def count(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False))
        return max(1, len(text) // 4)


@dataclass
class Block:
    text: str
    is_table: bool = False
    pages: list[int] = field(default_factory=list)
    rule_title: str | None = None  # set when this block starts a numbered rule


@dataclass
class Section:
    path: list[str]
    blocks: list[Block] = field(default_factory=list)


def rule_title_of(text: str) -> str | None:
    m = RULE_RE.match(text)
    if not m:
        return None
    rest = text[m.end() :]
    cuts = [i for sep in (".—", ".-", ". ", ".(", ".") if 0 < (i := rest.find(sep)) <= 150]
    if cuts:
        return f"Rule {m.group(1)}: {rest[: min(cuts)].strip()}"
    return f"Rule {m.group(1)}: {rest[:100].strip()}"


def load_docling_json(json_path: Path):
    from docling_core.types.doc.document import DoclingDocument

    return DoclingDocument.model_validate(json.loads(json_path.read_text()))


def _pages_of(item) -> list[int]:
    return sorted({p.page_no for p in getattr(item, "prov", []) or []})


def build_sections(doc, doc_title: str) -> list[Section]:
    from docling_core.types.doc.document import (
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    sections: list[Section] = []
    heading_path: list[str] = []
    current = Section(path=[])

    def flush() -> None:
        nonlocal current
        if current.blocks:
            sections.append(current)
        current = Section(path=list(heading_path))

    for item, level in doc.iterate_items():
        if isinstance(item, TitleItem):
            continue
        if isinstance(item, SectionHeaderItem):
            text = normalize_text(item.text)
            if not text:
                continue
            hlevel = getattr(item, "level", 1)
            heading_path[:] = heading_path[: max(0, hlevel - 1)] + [text]
            flush()
            continue
        if isinstance(item, TableItem):
            try:
                md = item.export_to_markdown(doc=doc)
            except TypeError:
                md = item.export_to_markdown()
            if md.strip():
                current.blocks.append(Block(md.strip(), is_table=True, pages=_pages_of(item)))
            continue
        if isinstance(item, TextItem):
            text = normalize_text(item.text)
            if not text or is_toc_line(text):
                continue
            # Docling strips list numbering into `marker` ("3."): restore it
            # so numbered rules are detectable and readable in chunk text.
            marker = getattr(item, "marker", "") or ""
            if getattr(item, "enumerated", False) and re.fullmatch(r"\d{1,3}\.", marker):
                text = f"{marker} {text}"
            title = rule_title_of(text)
            if title:  # a numbered rule begins -> new section
                flush()
                current.path = list(heading_path) + [title]
            current.blocks.append(Block(text, pages=_pages_of(item), rule_title=title))

    flush()
    return sections


def _split_oversize(text: str, max_tokens: int, overlap: int, tc: TokenCounter) -> list[str]:
    sentences = _SENT_RE.split(text)
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for s in sentences:
        n = tc.count(s)
        if buf and size + n > max_tokens:
            parts.append(" ".join(buf))
            tail: list[str] = []
            tsize = 0
            for prev in reversed(buf):  # sentence-level overlap
                tsize += tc.count(prev)
                if tsize > overlap:
                    break
                tail.insert(0, prev)
            buf, size = tail + [s], tsize + n
        else:
            buf.append(s)
            size += n
    if buf:
        parts.append(" ".join(buf))
    return parts


def _split_table(md: str, max_tokens: int, tc: TokenCounter) -> list[str]:
    """Split an oversize markdown table into row groups, repeating the header."""
    lines = md.strip().split("\n")
    if len(lines) < 4:
        return [md]
    header, rows = lines[:2], lines[2:]
    header_n = tc.count("\n".join(header))
    parts: list[str] = []
    buf: list[str] = []
    size = header_n
    for row in rows:
        n = tc.count(row)
        if buf and size + n > max_tokens:
            parts.append("\n".join(header + buf))
            buf, size = [], header_n
        buf.append(row)
        size += n
    if buf:
        parts.append("\n".join(header + buf))
    return parts


def _pack_blocks(blocks: list[Block], cfg: Config, tc: TokenCounter) -> list[list[Block]]:
    """Greedy pack blocks into groups <= target tokens. Tables stay atomic
    unless they exceed max_tokens, in which case they split by row groups."""
    target = cfg.chunking.target_tokens
    max_t = cfg.chunking.max_tokens
    groups: list[list[Block]] = []
    buf: list[Block] = []
    size = 0
    for b in blocks:
        if b.is_table and tc.count(b.text) > max_t:
            if buf:
                groups.append(buf)
                buf, size = [], 0
            for part in _split_table(b.text, max_t, tc):
                groups.append([Block(part, is_table=True, pages=b.pages)])
            continue
        n = tc.count(b.text)
        if not b.is_table and n > max_t:  # oversize prose -> sentence split
            for part in _split_oversize(b.text, target, cfg.chunking.forced_split_overlap, tc):
                yield_block = Block(part, pages=b.pages, rule_title=b.rule_title)
                if buf and size + tc.count(part) > target:
                    groups.append(buf)
                    buf, size = [], 0
                buf.append(yield_block)
                size += tc.count(part)
            continue
        if buf and size + n > target:
            groups.append(buf)
            buf, size = [], 0
        buf.append(b)
        size += n
    if buf:
        groups.append(buf)
    # merge a trailing tiny group into its predecessor
    if len(groups) >= 2 and sum(tc.count(b.text) for b in groups[-1]) < cfg.chunking.min_tokens:
        groups[-2].extend(groups.pop())
    return groups


def _mk_id(*parts: object) -> str:
    return hashlib.sha1(":".join(str(p) for p in parts).encode()).hexdigest()[:16]


def chunk_document(parsed: ParsedDoc, cfg: Config) -> tuple[list[Chunk], list[Chunk]]:
    """Returns (children, parents)."""
    doc = load_docling_json(parsed.docling_json_path)
    sections = build_sections(doc, parsed.title)
    tc = TokenCounter(cfg.embedding.model)

    children: list[Chunk] = []
    parents: list[Chunk] = []
    seq = 0
    for s_idx, section in enumerate(sections):
        groups = _pack_blocks(section.blocks, cfg, tc)
        breadcrumb = " > ".join([parsed.title, *section.path])

        # assemble children, then group them into parents
        sec_children: list[Chunk] = []
        for g_idx, group in enumerate(groups):
            text = "\n\n".join(b.text for b in group)
            pages = sorted({p for b in group for p in b.pages})
            sec_children.append(
                Chunk(
                    chunk_id=_mk_id(parsed.doc_id, s_idx, g_idx),
                    doc_id=parsed.doc_id,
                    text=text,
                    embed_text=f"{breadcrumb}\n\n{text}",
                    section_path=section.path,
                    element_type="table" if any(b.is_table for b in group) else "text",
                    page_start=pages[0] if pages else None,
                    page_end=pages[-1] if pages else None,
                    token_count=tc.count(text),
                    seq=seq + g_idx,
                )
            )
        seq += len(sec_children)

        p_buf: list[Chunk] = []
        p_size = 0
        p_idx = 0

        def flush_parent() -> None:
            nonlocal p_buf, p_size, p_idx
            if not p_buf:
                return
            text = "\n\n".join(c.text for c in p_buf)
            parent = Chunk(
                chunk_id=_mk_id(parsed.doc_id, s_idx, "p", p_idx),
                doc_id=parsed.doc_id,
                kind="parent",
                text=text,
                section_path=section.path,
                page_start=p_buf[0].page_start,
                page_end=p_buf[-1].page_end,
                token_count=p_size,
                seq=p_buf[0].seq,
            )
            for c in p_buf:
                c.parent_id = parent.chunk_id
            parents.append(parent)
            p_idx += 1
            p_buf, p_size = [], 0

        for c in sec_children:
            if p_buf and p_size + c.token_count > cfg.chunking.parent_tokens:
                flush_parent()
            p_buf.append(c)
            p_size += c.token_count
        flush_parent()
        children.extend(sec_children)

    return children, parents
