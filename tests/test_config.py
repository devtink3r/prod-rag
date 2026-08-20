from pathlib import Path

from rag.config import PROJECT_ROOT, load_config


def test_config_loads():
    cfg = load_config()
    assert cfg.chunking.target_tokens == 512
    assert cfg.embedding.model == "BAAI/bge-m3"
    assert cfg.retrieval.fused_top_k > cfg.retrieval.rerank_top_n


def test_source_dir_resolves():
    cfg = load_config()
    src = cfg.resolve_path(cfg.paths.source_dir)
    assert src == PROJECT_ROOT / "data" / "docs"
    assert isinstance(src, Path)
