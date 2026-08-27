"""FastAPI serving layer: POST /ask (SSE streaming), GET /health.

Models load once at startup (lifespan). Auth: X-API-Key header when
RAG_API_KEY is set.
"""

import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag.config import load_config, load_secrets

_state: dict = {}
_ingest = {"running": False, "log": [], "report": None}
_ingest_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from rag.generation.llm import OpenRouterLLM
    from rag.retrieval.retriever import build_retriever

    cfg = load_config()
    secrets = load_secrets()
    _state["cfg"] = cfg
    _state["retriever"] = build_retriever(cfg, secrets)
    _state["llm"] = OpenRouterLLM(cfg.llm, secrets)
    yield
    _state.clear()


def check_api_key(request: Request) -> None:
    expected = load_secrets().api_key
    if expected and request.headers.get("x-api-key") != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


app = FastAPI(title="prod-rag", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    history: list[dict] = Field(default_factory=list, max_length=20)
    stream: bool = True
    answer_model: str | None = Field(None, max_length=120)
    utility_model: str | None = Field(None, max_length=120)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": "retriever" in _state}


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


_models_cache: dict = {"ts": 0.0, "data": []}


@app.get("/models", dependencies=[Depends(check_api_key)])
def models() -> dict:
    """OpenRouter model catalog (cached 1h), for the UI dropdowns."""
    import time as _time

    import httpx

    if _time.time() - _models_cache["ts"] > 3600 or not _models_cache["data"]:
        resp = httpx.get("https://openrouter.ai/api/v1/models", timeout=20)
        resp.raise_for_status()
        out = []
        for m in resp.json().get("data", []):
            pricing = m.get("pricing") or {}
            free = float(pricing.get("prompt") or 0) == 0 and float(
                pricing.get("completion") or 0) == 0
            out.append({
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "free": free,
                "prompt_price": pricing.get("prompt"),
                "context": m.get("context_length"),
            })
        out.sort(key=lambda m: (not m["free"], m["id"]))
        _models_cache.update(ts=_time.time(), data=out)
    return {"models": _models_cache["data"]}


@app.get("/documents", dependencies=[Depends(check_api_key)])
def documents() -> dict:
    docs = _state["retriever"].registry.list_documents()
    try:
        points = _state["retriever"].store.count()
    except Exception:
        points = None
    return {"documents": docs, "index_points": points}


@app.post("/ingest", dependencies=[Depends(check_api_key)])
def trigger_ingest() -> dict:
    with _ingest_lock:
        if _ingest["running"]:
            return {"started": False, "reason": "already running"}
        _ingest.update(running=True, log=[], report=None)

    def _run() -> None:
        from rag.ingestion.pipeline import run_ingest

        try:
            report = run_ingest(load_config(), load_secrets(),
                                progress=lambda m: _ingest["log"].append(str(m)))
            _ingest["report"] = report.__dict__
        except Exception as exc:
            _ingest["log"].append(f"FAILED: {exc}")
        finally:
            _ingest["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return {"started": True}


@app.get("/ingest/status", dependencies=[Depends(check_api_key)])
def ingest_status() -> dict:
    return {"running": _ingest["running"], "log": _ingest["log"][-30:],
            "report": _ingest["report"]}


class FeedbackRequest(BaseModel):
    question: str = Field(max_length=2000)
    answer: str = Field(max_length=20000)
    verdict: str = Field(pattern="^(up|down)$")


@app.post("/feedback", dependencies=[Depends(check_api_key)])
def feedback(req: FeedbackRequest) -> dict:
    from rag.observability.tracer import record

    record({"kind": "feedback", "verdict": req.verdict,
            "question": req.question, "answer": req.answer[:1000]})
    return {"ok": True}


@app.post("/ask", dependencies=[Depends(check_api_key)])
def ask(req: AskRequest):
    from rag.generation.answerer import answer_question, stream_answer

    if not req.stream:
        ans = answer_question(req.question, _state["retriever"], _state["llm"],
                              _state["cfg"], history=req.history,
                              answer_model=req.answer_model,
                              utility_model=req.utility_model)
        return {"answer": ans.text, "no_answer": ans.no_answer,
                "sources": [s.__dict__ for s in ans.sources]}

    def sse():
        try:
            for event in stream_answer(req.question, _state["retriever"], _state["llm"],
                                       _state["cfg"], history=req.history,
                                       answer_model=req.answer_model,
                                       utility_model=req.utility_model):
                yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
        except Exception as exc:  # surface the cause instead of a dead stream
            yield f"event: error\ndata: {json.dumps(str(exc)[:500])}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
