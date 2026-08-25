"""rag CLI — entry points for ingestion, retrieval, eval, and serving."""

import warnings

import typer

# docling emits noisy-but-harmless bbox clamping warnings on some PDFs
warnings.filterwarnings("ignore", message=".*outside page bounds.*")
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
    doc = parse_file(path, cfg, max_pages=max_pages or None, progress=console.print)
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
    from rag.ingestion.parser import parse_file

    cfg = load_config()
    path = Path(file).resolve()
    parsed = parse_file(path, cfg, max_pages=max_pages or None, progress=console.print)

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
    out = cfg.resolve_path(cfg.paths.cache_dir) / "chunks" / f"{parsed.doc_id}.jsonl"
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
    max_pages: int = typer.Option(0, help="Parse only first N pages per doc (testing)"),
) -> None:
    """Incrementally ingest data/docs into Qdrant + registry."""
    from rag.config import load_secrets
    from rag.ingestion.pipeline import run_ingest

    report = run_ingest(
        load_config(), load_secrets(), dry_run=dry_run,
        max_pages=max_pages or None, progress=console.print,
    )
    console.print(report)
    if report.failed:
        raise typer.Exit(1)


@app.command()
def query(
    question: str,
    show_text: int = typer.Option(400, help="Chars of each block to print (0 = none)"),
) -> None:
    """Retrieve context for a question (phase 5; grounded answers arrive in phase 6)."""
    from rag.config import load_secrets
    from rag.retrieval.retriever import build_retriever

    retriever = build_retriever(load_config(), load_secrets())
    result = retriever.retrieve(question)
    if result.no_answer:
        console.print(
            f"[yellow]No relevant content found (top score "
            f"{result.top_score:.3f} below floor).[/yellow]"
        )
        raise typer.Exit(0)
    console.print(f"[green]{len(result.blocks)} context blocks[/green] "
                  f"(top score {result.top_score:.3f})\n")
    for i, b in enumerate(result.blocks, 1):
        pages = f"p.{b.page_start}" + (f"-{b.page_end}" if b.page_end != b.page_start else "")
        console.rule(f"[cyan][{i}] score={b.score:.3f} {pages} matches={b.matched_chunks}")
        console.print(" > ".join(b.section_path) or b.title, style="bold")
        if show_text:
            console.print(b.text[:show_text] + ("..." if len(b.text) > show_text else ""))


@app.command()
def ask(
    question: str,
    no_stream: bool = typer.Option(False, help="Wait for the full answer"),
) -> None:
    """Ask a question and get a grounded, cited answer."""
    from rag.config import load_secrets
    from rag.generation.answerer import answer_question, stream_answer
    from rag.generation.llm import OpenRouterLLM
    from rag.retrieval.retriever import build_retriever

    cfg = load_config()
    secrets = load_secrets()
    retriever = build_retriever(cfg, secrets)
    llm = OpenRouterLLM(cfg.llm, secrets)

    if no_stream:
        ans = answer_question(question, retriever, llm, cfg)
        console.print(ans.text)
        sources = ans.sources
    else:
        sources = []
        for event in stream_answer(question, retriever, llm, cfg):
            if event["type"] in ("token", "refusal"):
                print(event["data"], end="", flush=True)
            elif event["type"] == "sources":
                sources = event["data"]
        print()
    if sources:
        console.print("\n[bold]Sources[/bold]")
        for s in sources:
            d = s if isinstance(s, dict) else s.__dict__
            console.print(f"  [{d['n']}] {d['section'] or d['title']} ({d['pages']}, "
                          f"score {d['score']})")


@app.command("eval")
def eval_cmd() -> None:
    """Run the evaluation suite (phase 7)."""
    console.print("[yellow]Not implemented yet — arrives in phase 7.[/yellow]")
    raise typer.Exit(1)


@app.command()
def serve() -> None:
    """Start the FastAPI server (SSE streaming at POST /ask)."""
    import uvicorn

    cfg = load_config()
    uvicorn.run("rag.api.app:app", host=cfg.api.host, port=cfg.api.port)


if __name__ == "__main__":
    app()
