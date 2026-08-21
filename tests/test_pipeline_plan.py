from pathlib import Path

from rag.index.registry import RegistryEntry
from rag.ingestion.pipeline import plan_actions


def entry(path: str, digest: str, status: str = "indexed", version: int = 1) -> RegistryEntry:
    return RegistryEntry(path, f"id-{digest}", digest, status, version)


def test_new_changed_unchanged_deleted():
    found = {"/a.pdf": "h1", "/b.pdf": "h2-new", "/c.pdf": "h3"}
    registry = {
        "/b.pdf": entry("/b.pdf", "h2-old"),
        "/c.pdf": entry("/c.pdf", "h3"),
        "/gone.pdf": entry("/gone.pdf", "h4"),
    }
    plan = plan_actions(found, registry, pipeline_version=1)
    assert plan.new == [Path("/a.pdf")]
    assert plan.changed == [Path("/b.pdf")]
    assert plan.unchanged == [Path("/c.pdf")]
    assert plan.deleted == ["/gone.pdf"]


def test_pipeline_version_bump_forces_reindex():
    found = {"/a.pdf": "h1"}
    registry = {"/a.pdf": entry("/a.pdf", "h1", version=1)}
    assert plan_actions(found, registry, pipeline_version=2).changed == [Path("/a.pdf")]


def test_failed_docs_are_retried():
    found = {"/a.pdf": "h1"}
    registry = {"/a.pdf": entry("/a.pdf", "h1", status="failed")}
    assert plan_actions(found, registry, pipeline_version=1).changed == [Path("/a.pdf")]
