"""Docling-based parsing: any supported file -> ParsedDoc IR.

Heavy imports (docling pulls torch) are deferred inside functions so the CLI
stays fast for non-parsing commands.
"""

import hashlib
import json
import time
from pathlib import Path

from rag.config import Config
from rag.ingestion.cleaners import (
    find_boilerplate,
    normalize_for_matching,
    normalize_markdown,
)
from rag.ingestion.models import ParsedDoc, ParseStats

_converter = None


def compute_doc_id(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _get_converter():
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_table_structure = True
        pdf_opts.do_ocr = True  # applies only to pages without a text layer
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
        )
    return _converter


def _strip_boilerplate(doc, frequency: float) -> int:
    """Remove repeated header/footer-like text items from a DoclingDocument.

    Returns number of items removed. Never touches tables or headings.
    """
    from docling_core.types.doc.document import SectionHeaderItem, TextItem

    page_texts: dict[int, list[str]] = {}
    candidates = []
    for item, _level in doc.iterate_items():
        if not isinstance(item, TextItem) or isinstance(item, SectionHeaderItem):
            continue
        if not item.prov or not item.text.strip():
            continue
        key = normalize_for_matching(item.text)
        page = item.prov[0].page_no
        page_texts.setdefault(page, []).append(key)
        candidates.append((item, key))

    boilerplate = find_boilerplate(page_texts, frequency)
    if not boilerplate:
        return 0
    to_delete = [item for item, key in candidates if key in boilerplate]
    try:
        doc.delete_items(node_items=to_delete)
    except Exception:
        for item in to_delete:  # fallback: blank them out
            item.text = ""
    return len(to_delete)


def parse_file(path: Path, cfg: Config, max_pages: int | None = None) -> ParsedDoc:
    from docling_core.types.doc.document import SectionHeaderItem, TableItem

    t0 = time.time()
    doc_id = compute_doc_id(path)
    kwargs = {"page_range": (1, max_pages)} if max_pages else {}
    result = _get_converter().convert(str(path), **kwargs)
    doc = result.document

    removed = _strip_boilerplate(doc, cfg.cleaning.boilerplate_page_frequency)

    num_tables = num_headings = 0
    for item, _level in doc.iterate_items():
        if isinstance(item, TableItem):
            num_tables += 1
        elif isinstance(item, SectionHeaderItem):
            num_headings += 1

    cache_dir = cfg.resolve_path(cfg.paths.cache_dir) / "parsed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = cache_dir / f"{doc_id}.json"
    json_path.write_text(json.dumps(doc.export_to_dict()), encoding="utf-8")

    markdown = normalize_markdown(doc.export_to_markdown())

    warnings = [str(e.error_message) for e in getattr(result, "errors", [])][:10]
    return ParsedDoc(
        doc_id=doc_id,
        source_path=path,
        doc_type=path.suffix.lstrip(".").lower(),
        title=(doc.name or path.stem),
        markdown=markdown,
        docling_json_path=json_path,
        stats=ParseStats(
            num_pages=doc.num_pages(),
            num_tables=num_tables,
            num_headings=num_headings,
            parse_seconds=round(time.time() - t0, 1),
            boilerplate_lines_removed=removed,
        ),
        warnings=warnings,
    )
