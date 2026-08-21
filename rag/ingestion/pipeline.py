"""Incremental ingestion pipeline.

discover -> plan (new/changed/unchanged/deleted) -> per-doc:
parse -> chunk -> enrich -> embed -> upsert Qdrant + parents -> registry.
Per-document isolation: one bad file never fails the batch.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rag.config import Config, Secrets
from rag.index.registry import RegistryEntry


@dataclass
class Plan:
    new: list[Path] = field(default_factory=list)
    changed: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)  # source_paths gone from disk


def discover(cfg: Config) -> list[Path]:
    root = cfg.resolve_path(cfg.paths.source_dir)
    files: set[Path] = set()
    for pattern in cfg.ingestion.include_patterns:
        files.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(files)


def plan_actions(
    found: dict[str, str],  # source_path -> content_hash
    registry: dict[str, RegistryEntry],
    pipeline_version: int,
) -> Plan:
    plan = Plan()
    for path_str, digest in found.items():
        entry = registry.get(path_str)
        if entry is None:
            plan.new.append(Path(path_str))
        elif (
            entry.content_hash != digest
            or entry.pipeline_version != pipeline_version
            or entry.status != "indexed"
        ):
            plan.changed.append(Path(path_str))
        else:
            plan.unchanged.append(Path(path_str))
    plan.deleted = [p for p in registry if p not in found]
    return plan


@dataclass
class IngestReport:
    indexed: int = 0
    failed: int = 0
    skipped: int = 0
    removed: int = 0
    errors: list[str] = field(default_factory=list)


def run_ingest(
    cfg: Config,
    secrets: Secrets,
    dry_run: bool = False,
    max_pages: int | None = None,
    progress=lambda msg: None,
) -> IngestReport:
    from rag.index.registry import Registry
    from rag.ingestion.parser import compute_doc_id

    registry = Registry(secrets.postgres_dsn, cfg.registry.schema_name)
    files = discover(cfg)
    found = {str(p): compute_doc_id(p) for p in files}
    plan = plan_actions(found, registry.entries(), cfg.ingestion.pipeline_version)

    progress(
        f"plan: {len(plan.new)} new, {len(plan.changed)} changed, "
        f"{len(plan.unchanged)} unchanged, {len(plan.deleted)} deleted"
    )
    report = IngestReport(skipped=len(plan.unchanged))
    if dry_run:
        for p in plan.new:
            progress(f"  would index (new): {p.name}")
        for p in plan.changed:
            progress(f"  would re-index (changed): {p.name}")
        for p in plan.deleted:
            progress(f"  would remove: {p}")
        return report

    from rag.index.embedder import BgeM3Embedder
    from rag.index.qdrant_store import QdrantStore
    from rag.ingestion.chunker import chunk_document
    from rag.ingestion.enrichment import add_contextual_summaries
    from rag.ingestion.parser import parse_file

    store = QdrantStore(cfg.vector_store)
    store.ensure_collection(cfg.embedding.model, cfg.ingestion.pipeline_version)
    embedder = None  # lazy: model load is slow, skip entirely if nothing to do

    old_entries = registry.entries()

    for path in [*plan.new, *plan.changed]:
        old = old_entries.get(str(path))
        try:
            progress(f"parsing {path.name} ...")
            parsed = parse_file(path, cfg, max_pages=max_pages)
            children, parents = chunk_document(parsed, cfg)
            enriched = add_contextual_summaries(children, parsed.markdown, cfg, secrets)
            if enriched:
                progress(f"  enriched {enriched}/{len(children)} chunks")
            if embedder is None:
                progress(f"loading embedding model {cfg.embedding.model} ...")
                embedder = BgeM3Embedder(cfg.embedding)
            progress(f"  embedding {len(children)} chunks ...")
            embeddings = embedder.encode([c.embed_text for c in children])

            if old:  # remove the previous version's points first
                store.delete_doc(old.doc_id)
            store.upsert_chunks(
                children,
                embeddings,
                extra={
                    "source_path": str(path),
                    "title": parsed.title,
                    "doc_type": parsed.doc_type,
                    "pipeline_version": cfg.ingestion.pipeline_version,
                },
            )
            registry.replace_parents(old.doc_id if old else None, parents)
            registry.mark(
                str(path), parsed.doc_id, found[str(path)], "indexed",
                title=parsed.title, doc_type=parsed.doc_type,
                num_pages=parsed.stats.num_pages, chunk_count=len(children),
                pipeline_version=cfg.ingestion.pipeline_version,
                last_indexed_at=datetime.now(timezone.utc),
            )
            report.indexed += 1
            progress(f"  indexed {path.name}: {len(children)} chunks, {len(parents)} parents")
        except Exception as exc:  # per-doc isolation
            report.failed += 1
            report.errors.append(f"{path.name}: {exc}")
            registry.mark(
                str(path), found[str(path)], found[str(path)], "failed", error=str(exc)[:500]
            )
            progress(f"  FAILED {path.name}: {exc}")

    for source_path in plan.deleted:
        entry = old_entries[source_path]
        store.delete_doc(entry.doc_id)
        registry.delete_parents(entry.doc_id)
        registry.delete_doc(source_path)
        report.removed += 1
        progress(f"removed {source_path}")

    return report
