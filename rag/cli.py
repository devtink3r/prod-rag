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
