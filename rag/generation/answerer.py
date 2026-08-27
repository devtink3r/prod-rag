"""Answer orchestration: retrieve -> refuse or generate -> validate citations."""

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from rag.config import Config
from rag.generation.prompts import CONDENSE_PROMPT, REFUSAL_TEXT, build_messages

_CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    n: int
    title: str
    section: str
    pages: str
    source_path: str
    score: float
    snippet: str = ""


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    no_answer: bool = False
    timings: dict = field(default_factory=dict)


def condense_question(question: str, history: list[dict], llm, cfg: Config) -> str:
    """Rewrite a follow-up into a standalone question using the cheap model."""
    if not history:
        return question
    lines = [f"{m['role']}: {m['content']}" for m in history[-6:]]
    prompt = CONDENSE_PROMPT.format(history="\n".join(lines), question=question)
    try:
        return llm.complete(
            [{"role": "user", "content": prompt}],
            model=cfg.llm.utility_model, max_tokens=600,
        ).strip()
    except Exception:
        return question  # fail-soft: answer the raw question


def _sources_from_blocks(blocks) -> list[Source]:
    out = []
    for i, b in enumerate(blocks, 1):
        pages = f"p.{b.page_start}" if b.page_start else ""
        if b.page_end and b.page_end != b.page_start:
            pages += f"-{b.page_end}"
        out.append(Source(
            n=i, title=b.title, section=" > ".join(b.section_path),
            pages=pages, source_path=b.source_path, score=round(b.score, 3),
            snippet=b.text[:600],
        ))
    return out


def validate_citations(text: str, num_blocks: int) -> tuple[str, set[int]]:
    """Drop citation markers pointing at nonexistent blocks; return cited set."""
    cited: set[int] = set()

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= num_blocks:
            cited.add(n)
            return m.group(0)
        return ""

    return _CITE_RE.sub(repl, text), cited


def _trace(question, result, answer_text, cited, timings, cfg) -> None:
    from rag.observability.tracer import record

    record({
        "question": question,
        "no_answer": result.no_answer,
        "top_score": round(result.top_score, 3),
        "blocks": len(result.blocks),
        "cited": sorted(cited),
        "answer_chars": len(answer_text),
        "model": cfg.llm.answer_model,
        "timings": timings,
    })


def answer_question(
    question: str, retriever, llm, cfg: Config,
    history: list[dict] | None = None,
) -> Answer:
    t0 = time.time()
    t = time.time()
    q = condense_question(question, history or [], llm, cfg)
    timings = {}
    if history:
        timings["condense_ms"] = int((time.time() - t) * 1000)
    result = retriever.retrieve(q)
    timings.update(result.timings)
    if result.no_answer:
        timings["total_ms"] = int((time.time() - t0) * 1000)
        _trace(q, result, REFUSAL_TEXT, set(), timings, cfg)
        return Answer(text=REFUSAL_TEXT, no_answer=True, timings=timings)
    t = time.time()
    raw = llm.complete(build_messages(q, result.blocks))
    timings["llm_ms"] = int((time.time() - t) * 1000)
    text, cited = validate_citations(raw, len(result.blocks))
    sources = [s for s in _sources_from_blocks(result.blocks) if not cited or s.n in cited]
    timings["total_ms"] = int((time.time() - t0) * 1000)
    _trace(q, result, text, cited, timings, cfg)
    return Answer(text=text, sources=sources, timings=timings)


def stream_answer(
    question: str, retriever, llm, cfg: Config,
    history: list[dict] | None = None,
) -> Iterator[dict]:
    """Yields events: {'type': 'token', 'data': str} then {'type': 'sources', ...}
    or a single {'type': 'refusal'} event."""
    import queue as queue_mod
    import threading

    t0 = time.time()
    t = time.time()
    timings = {}
    if history:
        yield {"type": "stage", "data": "condensing follow-up question"}
        q = condense_question(question, history, llm, cfg)
        timings["condense_ms"] = int((time.time() - t) * 1000)
    else:
        q = question

    # run retrieval in a thread so stage events stream out live
    stage_q: queue_mod.Queue = queue_mod.Queue()
    holder: dict = {}

    def _run() -> None:
        try:
            try:
                holder["result"] = retriever.retrieve(q, on_stage=stage_q.put)
            except TypeError:  # fakes/tests without on_stage support
                holder["result"] = retriever.retrieve(q)
        except Exception as exc:
            holder["error"] = exc
        finally:
            stage_q.put(None)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    while True:
        item = stage_q.get()
        if item is None:
            break
        yield {"type": "stage", "data": item}
    worker.join()
    if "error" in holder:
        raise holder["error"]
    result = holder["result"]
    timings.update(result.timings)

    if result.no_answer:
        timings["total_ms"] = int((time.time() - t0) * 1000)
        _trace(q, result, REFUSAL_TEXT, set(), timings, cfg)
        yield {"type": "refusal", "data": REFUSAL_TEXT}
        yield {"type": "timings", "data": timings}
        return

    yield {"type": "stage", "data": "generating answer"}
    parts: list[str] = []
    t = time.time()
    first_token_ms = None
    for token in llm.stream(build_messages(q, result.blocks)):
        if first_token_ms is None:
            first_token_ms = int((time.time() - t) * 1000)
        parts.append(token)
        yield {"type": "token", "data": token}
    timings["llm_first_token_ms"] = first_token_ms or 0
    timings["llm_stream_ms"] = int((time.time() - t) * 1000)
    _, cited = validate_citations("".join(parts), len(result.blocks))
    sources = [s.__dict__ for s in _sources_from_blocks(result.blocks)
               if not cited or s.n in cited]
    timings["total_ms"] = int((time.time() - t0) * 1000)
    _trace(q, result, "".join(parts), cited, timings, cfg)
    yield {"type": "sources", "data": sources}
    usage = getattr(llm, "last_usage", {}) or {}
    if usage:
        yield {"type": "usage", "data": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost": usage.get("cost"),
        }}
    yield {"type": "timings", "data": timings}
