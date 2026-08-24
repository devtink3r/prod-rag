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
        from rag.config import export_hf_token

        export_hf_token()
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


SEGMENT_PAGES = 40  # PDF pages per conversion segment (for progress reporting)


def _segments(total: int, size: int) -> list[tuple[int, int]]:
    return [(s, min(s + size - 1, total)) for s in range(1, total + 1, size)]


def _pdf_page_count(path: Path) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def _convert(path: Path, max_pages: int | None, progress) -> tuple[object, list[str]]:
    """Convert a file; PDFs are parsed in page segments so progress is visible.
    Returns (DoclingDocument, warnings)."""
    conv = _get_converter()
    if path.suffix.lower() != ".pdf":
        result = conv.convert(str(path))
        return result.document, [str(e.error_message) for e in result.errors]

    total = _pdf_page_count(path)
    if max_pages:
        total = min(total, max_pages)
    warnings: list[str] = []
    try:
        from docling_core.types.doc.document import DoclingDocument

        docs = []
        for start, end in _segments(total, SEGMENT_PAGES):
            t = time.time()
            result = conv.convert(str(path), page_range=(start, end))
            docs.append(result.document)
            warnings += [str(e.error_message) for e in result.errors]
            progress(f"  parsed pages {start}-{end}/{total} ({time.time() - t:.0f}s)")
        doc = docs[0] if len(docs) == 1 else DoclingDocument.concatenate(docs)
        return doc, warnings
    except Exception as exc:  # segmented path failed -> single shot
        progress(f"  segmented parse failed ({exc}); falling back to single pass")
        result = conv.convert(str(path), page_range=(1, total))
        return result.document, [str(e.error_message) for e in result.errors]


def parse_file(
    path: Path, cfg: Config, max_pages: int | None = None, progress=lambda msg: None
) -> ParsedDoc:
    from docling_core.types.doc.document import SectionHeaderItem, TableItem

    t0 = time.time()
    doc_id = compute_doc_id(path)

    cache_dir = cfg.resolve_path(cfg.paths.cache_dir) / "parsed"
    # partial parses get their own cache key so they never poison full runs
    suffix = f".p{max_pages}" if max_pages else ""
    json_path = cache_dir / f"{doc_id}{suffix}.json"

    if json_path.exists():
        progress(f"  using cached parse {json_path.name}")
        from docling_core.types.doc.document import DoclingDocument

        doc = DoclingDocument.model_validate(json.loads(json_path.read_text()))
        conv_warnings: list[str] = []
        removed = 0
    else:
        doc, conv_warnings = _convert(path, max_pages, progress)
        removed = _strip_boilerplate(doc, cfg.cleaning.boilerplate_page_frequency)
        cache_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(doc.export_to_dict()), encoding="utf-8")

    num_tables = num_headings = 0
    for item, _level in doc.iterate_items():
        if isinstance(item, TableItem):
            num_tables += 1
        elif isinstance(item, SectionHeaderItem):
            num_headings += 1

    markdown = normalize_markdown(doc.export_to_markdown())

    warnings = conv_warnings[:10]
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
