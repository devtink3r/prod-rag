"""rag CLI — entry points for ingestion, retrieval, eval, and serving."""

import typer
from rich.console import Console

from rag.config import load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def info() -> None:
    """Show resolved configuration."""
    cfg = load_config()
    console.print_json(cfg.model_dump_json())


@app.command()
def parse(
    file: str = typer.Argument(help="Path to a document to parse"),
    preview: bool = typer.Option(True, help="Write cleaned markdown next to cache"),
    max_pages: int = typer.Option(0, help="Parse only the first N pages (0 = all)"),
) -> None:
    """Parse + clean one document and show IR stats (phase 2 debug tool)."""
    from pathlib import Path

    from rag.ingestion.parser import parse_file

    cfg = load_config()
    path = Path(file).resolve()
    if not path.exists():
        console.print(f"[red]Not found: {path}[/red]")
        raise typer.Exit(1)
    limit = f" (first {max_pages} pages)" if max_pages else ""
    console.print(f"Parsing [bold]{path.name}[/bold]{limit} ...")
    doc = parse_file(path, cfg, max_pages=max_pages or None)
    console.print(doc.stats.model_dump())
    if doc.warnings:
        console.print(f"[yellow]{len(doc.warnings)} warnings[/yellow]", doc.warnings[:3])
    if preview:
        out = cfg.resolve_path(cfg.paths.cache_dir) / "preview" / f"{path.stem}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(doc.markdown, encoding="utf-8")
        console.print(f"Cleaned markdown: {out}")
    console.print(f"DoclingDocument cache: {doc.docling_json_path}")


@app.command()
def chunk(
    file: str = typer.Argument(help="Path to a document"),
    max_pages: int = typer.Option(0, help="Parse only first N pages (0 = all)"),
    enrich: bool = typer.Option(False, help="Add LLM contextual summaries (needs API key)"),
    show: int = typer.Option(2, help="Print first N chunks"),
) -> None:
    """Parse (cache-aware) + chunk one document; inspect results (phase 3 debug)."""
    import json as _json
    from pathlib import Path

    from rag.config import load_secrets
    from rag.ingestion.chunker import chunk_document
    from rag.ingestion.enrichment import add_contextual_summaries
    from rag.ingestion.parser import compute_doc_id, parse_file

    cfg = load_config()
    path = Path(file).resolve()
    doc_id = compute_doc_id(path)
    cached = cfg.resolve_path(cfg.paths.cache_dir) / "parsed" / f"{doc_id}.json"
    if cached.exists():
        from rag.ingestion.models import ParsedDoc

        console.print(f"Using cached parse {cached.name}")
        parsed = ParsedDoc(
            doc_id=doc_id, source_path=path, doc_type=path.suffix.lstrip("."),
            title=path.stem, docling_json_path=cached,
        )
    else:
        parsed = parse_file(path, cfg, max_pages=max_pages or None)

    children, parents = chunk_document(parsed, cfg)
    if enrich:
        n = add_contextual_summaries(children, parsed.markdown, cfg, load_secrets())
        console.print(f"Enriched {n} chunks")

    toks = [c.token_count for c in children]
    console.print({
        "children": len(children), "parents": len(parents),
        "avg_tokens": sum(toks) // max(1, len(toks)),
        "max_tokens": max(toks, default=0),
        "table_chunks": sum(1 for c in children if c.element_type == "table"),
        "sections": len({tuple(c.section_path) for c in children}),
    })
    out = cfg.resolve_path(cfg.paths.cache_dir) / "chunks" / f"{doc_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in [*children, *parents]:
            f.write(_json.dumps(c.model_dump(mode="json")) + "\n")
    console.print(f"Chunks written: {out}")
    for c in children[:show]:
        console.rule(f"[cyan]{c.chunk_id} ({c.token_count} tok, p.{c.page_start}-{c.page_end})")
        console.print(" > ".join(c.section_path) or "(no section)", style="dim")
        console.print(c.text[:400] + ("..." if len(c.text) > 400 else ""))


@app.command()
def ingest(
    dry_run: bool = typer.Option(False, help="Report changes without indexing"),
) -> None:
    """Run the ingestion pipeline (phase 2+)."""
    console.print("[yellow]Not implemented yet — arrives in phase 2-4.[/yellow]")
    raise typer.Exit(1)


@app.command()
def query(question: str) -> None:
    """Ask a question against the index (phase 5+)."""
    console.print("[yellow]Not implemented yet — arrives in phase 5-6.[/yellow]")
    raise typer.Exit(1)


@app.command("eval")
def eval_cmd() -> None:
    """Run the evaluation suite (phase 7)."""
    console.print("[yellow]Not implemented yet — arrives in phase 7.[/yellow]")
    raise typer.Exit(1)


@app.command()
def serve() -> None:
    """Start the FastAPI server (phase 6)."""
    console.print("[yellow]Not implemented yet — arrives in phase 6.[/yellow]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
