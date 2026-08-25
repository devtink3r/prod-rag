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


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    no_answer: bool = False


def condense_question(question: str, history: list[dict], llm, cfg: Config) -> str:
    """Rewrite a follow-up into a standalone question using the cheap model."""
    if not history:
        return question
    lines = [f"{m['role']}: {m['content']}" for m in history[-6:]]
    prompt = CONDENSE_PROMPT.format(history="\n".join(lines), question=question)
    try:
        return llm.complete(
            [{"role": "user", "content": prompt}],
            model=cfg.llm.utility_model, max_tokens=200,
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


def _trace(question, result, answer_text, cited, t0, t_retrieve, cfg) -> None:
    from rag.observability.tracer import record

    record({
        "question": question,
        "no_answer": result.no_answer,
        "top_score": round(result.top_score, 3),
        "blocks": len(result.blocks),
        "cited": sorted(cited),
        "answer_chars": len(answer_text),
        "model": cfg.llm.answer_model,
        "retrieve_ms": int(t_retrieve * 1000),
        "total_ms": int((time.time() - t0) * 1000),
    })


def answer_question(
    question: str, retriever, llm, cfg: Config,
    history: list[dict] | None = None,
) -> Answer:
    t0 = time.time()
    q = condense_question(question, history or [], llm, cfg)
    result = retriever.retrieve(q)
    t_retrieve = time.time() - t0
    if result.no_answer:
        _trace(q, result, REFUSAL_TEXT, set(), t0, t_retrieve, cfg)
        return Answer(text=REFUSAL_TEXT, no_answer=True)
    raw = llm.complete(build_messages(q, result.blocks))
    text, cited = validate_citations(raw, len(result.blocks))
    sources = [s for s in _sources_from_blocks(result.blocks) if not cited or s.n in cited]
    _trace(q, result, text, cited, t0, t_retrieve, cfg)
    return Answer(text=text, sources=sources)


def stream_answer(
    question: str, retriever, llm, cfg: Config,
    history: list[dict] | None = None,
) -> Iterator[dict]:
    """Yields events: {'type': 'token', 'data': str} then {'type': 'sources', ...}
    or a single {'type': 'refusal'} event."""
    t0 = time.time()
    q = condense_question(question, history or [], llm, cfg)
    result = retriever.retrieve(q)
    t_retrieve = time.time() - t0
    if result.no_answer:
        _trace(q, result, REFUSAL_TEXT, set(), t0, t_retrieve, cfg)
        yield {"type": "refusal", "data": REFUSAL_TEXT}
        return
    parts: list[str] = []
    for token in llm.stream(build_messages(q, result.blocks)):
        parts.append(token)
        yield {"type": "token", "data": token}
    _, cited = validate_citations("".join(parts), len(result.blocks))
    sources = [s.__dict__ for s in _sources_from_blocks(result.blocks)
               if not cited or s.n in cited]
    _trace(q, result, "".join(parts), cited, t0, t_retrieve, cfg)
    yield {"type": "sources", "data": sources}
