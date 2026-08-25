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
uv run rag ask "..."         # grounded, cited answer (streaming)
uv run rag eval              # golden-set eval: hit@k, MRR, refusal accuracy
uv run rag eval --generation # + LLM-judged faithfulness/relevance
uv run rag traces            # recent request traces (JSONL in .cache/traces/)
uv run pytest                # tests
```

Drop documents into `data/docs/` and run `rag ingest`. Ingestion is incremental —
unchanged files are skipped, changed files are re-indexed, deleted files are removed.

## Latency

CLI commands (`rag ask`, `rag query`) load models fresh each invocation
(~30-60s on CPU). For interactive use run `rag serve` once — models stay warm
and each request pays only embed + search + rerank + LLM. The per-stage
timing breakdown after each answer shows where time goes; the main knobs are
`retrieval.fused_top_k` and `retrieval.rerank_max_length` (both trade recall
for rerank speed — verify changes with `rag eval`).

## Configuration

All tunables in `config.yaml` (chunk sizes, models, top-k, score floors).
Secrets only in `.env` (`RAG_OPENROUTER_API_KEY`, `RAG_POSTGRES_DSN`, `RAG_API_KEY`).

## Evaluation & observability

Golden set lives in `rag/eval/golden.yaml` — add cases whenever a real question
fails. `rag eval` saves reports to `.cache/eval/` and prints the delta against
the previous run, so every tuning change (chunk size, top-k, model, floor) gets
measured, not guessed. Request traces are JSONL in `.cache/traces/` (schema maps
1:1 onto Langfuse spans if you later self-host Langfuse for a UI).

## Known issues / eval backlog

- Appendix form fields ("22. Percentage share/interest...") trigger the rule-boundary
  detector, producing misleading "Rule N" breadcrumbs for form content (pages ~850+).
  Reranker buries them; revisit during evaluation (suppress rule detection in
  appendix/form regions or require rule-like title grammar).
- Parents/children ratio is ~1.2:1 (many single-child sections); consider merging
  small adjacent parents if generation context feels thin.

## Build phases

1. ✅ Scaffold: uv project, docker-compose, config, CLI skeleton
2. Parsing + cleaning (Docling → Markdown IR)
3. Chunking + enrichment
4. Embedding + Qdrant indexing + incremental registry
5. Retrieval: hybrid search + rerank + parent expansion
6. Generation + citations + FastAPI
7. Eval harness + Langfuse
8. Tuning against eval results
