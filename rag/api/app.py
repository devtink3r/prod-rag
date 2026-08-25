"""FastAPI serving layer: POST /ask (SSE streaming), GET /health.

Models load once at startup (lifespan). Auth: X-API-Key header when
RAG_API_KEY is set.
"""

import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.config import load_config, load_secrets

_state: dict = {}


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": "retriever" in _state}


@app.post("/ask", dependencies=[Depends(check_api_key)])
def ask(req: AskRequest):
    from rag.generation.answerer import answer_question, stream_answer

    if not req.stream:
        ans = answer_question(req.question, _state["retriever"], _state["llm"],
                              _state["cfg"], history=req.history)
        return {"answer": ans.text, "no_answer": ans.no_answer,
                "sources": [s.__dict__ for s in ans.sources]}

    def sse():
        for event in stream_answer(req.question, _state["retriever"], _state["llm"],
                                   _state["cfg"], history=req.history):
            yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
