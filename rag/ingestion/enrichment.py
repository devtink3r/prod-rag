"""Contextual-retrieval enrichment: a cheap LLM writes 1-2 sentences situating
each chunk within its document; prepended to embed_text before embedding.
Fail-soft: any error leaves the chunk unenriched rather than failing ingestion.
"""

import httpx

from rag.config import Config, Secrets
from rag.ingestion.models import Chunk

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_PROMPT = """<document>
{doc_excerpt}
</document>

Here is a chunk from this document:
<chunk>
{chunk}
</chunk>

Write 1-2 short sentences situating this chunk within the overall document
(what rule/section it belongs to, what it governs). Answer with only the
context sentences, nothing else."""


def add_contextual_summaries(
    chunks: list[Chunk],
    doc_markdown: str,
    cfg: Config,
    secrets: Secrets,
    max_chunks: int | None = None,
) -> int:
    """Enrich child chunks in place. Returns number enriched."""
    if not cfg.chunking.contextual_summaries or not secrets.openrouter_api_key:
        return 0
    excerpt = doc_markdown[:8000]
    headers = {"Authorization": f"Bearer {secrets.openrouter_api_key}"}
    enriched = 0
    targets = [c for c in chunks if c.kind == "child"]
    if max_chunks:
        targets = targets[:max_chunks]
    with httpx.Client(timeout=30) as client:
        for chunk in targets:
            try:
                resp = client.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json={
                        "model": cfg.llm.utility_model,
                        "temperature": 0.0,
                        "max_tokens": 150,
                        "messages": [
                            {
                                "role": "user",
                                "content": _PROMPT.format(
                                    doc_excerpt=excerpt, chunk=chunk.text[:4000]
                                ),
                            }
                        ],
                    },
                )
                resp.raise_for_status()
                context = resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                continue
            if context:
                chunk.embed_text = f"{context}\n\n{chunk.embed_text}"
                enriched += 1
    return enriched
