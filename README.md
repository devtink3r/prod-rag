# prod-rag

Open-source enterprise RAG: Docling parsing → structure-aware chunking → BGE-M3 hybrid
embeddings → Qdrant → cross-encoder reranking → grounded generation via OpenRouter.
Built on LlamaIndex.

## Stack

| Layer | Choice |
|---|---|
| Parsing (pdf/docx/pptx/xlsx/html/md + OCR) | Docling |
| Chunking | Structure-aware, parent-child, ~512 tok |
| Embeddings | BGE-M3 (dense + sparse, local) |
| Vector store | Qdrant (hybrid, server-side RRF) |
| Reranker | bge-reranker-v2-m3 |
| LLM | OpenRouter (config-driven) |
| Registry / sessions | Postgres |
| Eval / observability | RAGAS, Langfuse |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv venv && uv sync           # create .venv + install deps
cp .env.example .env         # then fill in your OpenRouter key
docker compose up -d         # qdrant :6333, postgres :5433
```

## Usage

```bash
uv run rag info              # show resolved config
uv run rag ingest            # index documents from data/docs/  (phase 2+)
uv run rag query "..."       # ask a question                   (phase 5+)
uv run rag serve             # FastAPI + SSE                    (phase 6)
uv run rag eval              # evaluation suite                 (phase 7)
uv run pytest                # tests
```

Drop documents into `data/docs/` and run `rag ingest`. Ingestion is incremental —
unchanged files are skipped, changed files are re-indexed, deleted files are removed.

## Configuration

All tunables in `config.yaml` (chunk sizes, models, top-k, score floors).
Secrets only in `.env` (`RAG_OPENROUTER_API_KEY`, `RAG_POSTGRES_DSN`, `RAG_API_KEY`).

## Build phases

1. ✅ Scaffold: uv project, docker-compose, config, CLI skeleton
2. Parsing + cleaning (Docling → Markdown IR)
3. Chunking + enrichment
4. Embedding + Qdrant indexing + incremental registry
5. Retrieval: hybrid search + rerank + parent expansion
6. Generation + citations + FastAPI
7. Eval harness + Langfuse
8. Tuning against eval results
