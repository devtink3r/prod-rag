"""Lightweight request tracing: one JSON line per request in
.cache/traces/YYYY-MM-DD.jsonl. Zero infrastructure; grep-able; the schema
maps 1:1 onto Langfuse spans if you later self-host it.
"""

import json
import time
from pathlib import Path

from rag.config import load_config

_ENABLED = True


def record(event: dict) -> None:
    if not _ENABLED:
        return
    try:
        cfg = load_config()
        out_dir = cfg.resolve_path(cfg.paths.cache_dir) / "traces"
        out_dir.mkdir(parents=True, exist_ok=True)
        event.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
        path = out_dir / f"{time.strftime('%Y-%m-%d')}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass  # tracing must never break serving


def tail(n: int = 20) -> list[dict]:
    cfg = load_config()
    out_dir = cfg.resolve_path(cfg.paths.cache_dir) / "traces"
    files = sorted(out_dir.glob("*.jsonl"))
    if not files:
        return []
    lines = files[-1].read_text().strip().split("\n")
    return [json.loads(line) for line in lines[-n:]]
