"""Evaluation runner: retrieval metrics always; generation metrics optional.

Retrieval: hit@k (expected rule appears in retrieved blocks), MRR, refusal
accuracy. Generation (needs LLM): keyword recall + LLM-judged faithfulness
and relevance (RAGAS-style, 1-5).
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from rag.config import Config


@dataclass
class EvalCase:
    question: str
    expect_rule: str | None = None
    expect_keywords: list[str] = field(default_factory=list)
    expect_refusal: bool = False


@dataclass
class CaseResult:
    question: str
    hit: bool | None = None       # expected rule retrieved (None = n/a)
    rank: int | None = None       # 1-based rank of first matching block
    refusal_correct: bool | None = None
    keywords_found: float | None = None
    faithfulness: float | None = None
    relevance: float | None = None
    top_score: float = 0.0
    seconds: float = 0.0
    answer: str = ""
    error: str = ""


def load_cases(path: Path) -> list[EvalCase]:
    data = yaml.safe_load(path.read_text())
    return [EvalCase(**c) for c in data["cases"]]


def _rule_rank(blocks, expect_rule: str) -> int | None:
    for i, b in enumerate(blocks, 1):
        if any(p.startswith(expect_rule + ":") or p == expect_rule
               for p in b.section_path):
            return i
    return None


def evaluate_case(case: EvalCase, retriever, llm, cfg: Config,
                  generation: bool) -> CaseResult:
    t0 = time.time()
    result = retriever.retrieve(case.question)
    r = CaseResult(question=case.question, top_score=round(result.top_score, 3))

    if case.expect_refusal:
        r.refusal_correct = result.no_answer
    elif case.expect_rule:
        rank = None if result.no_answer else _rule_rank(result.blocks, case.expect_rule)
        r.hit = rank is not None
        r.rank = rank

    if generation and llm and not case.expect_refusal and not result.no_answer:
        from rag.eval.judge import judge_answer
        from rag.generation.answerer import answer_question

        try:  # generation failures (rate limits etc.) shouldn't kill the run
            ans = answer_question(case.question, retriever, llm, cfg)
            r.answer = ans.text
            if case.expect_keywords:
                found = sum(1 for k in case.expect_keywords if k.lower() in ans.text.lower())
                r.keywords_found = found / len(case.expect_keywords)
            scores = judge_answer(case.question, ans.text, result.blocks, llm, cfg)
            r.faithfulness = scores.get("faithfulness")
            r.relevance = scores.get("relevance")
        except Exception as exc:
            r.error = str(exc)[:200]

    r.seconds = round(time.time() - t0, 1)
    return r


def _avg(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(results: list[CaseResult]) -> dict:
    retrieval = [r for r in results if r.hit is not None]
    refusals = [r for r in results if r.refusal_correct is not None]
    return {
        "cases": len(results),
        "hit_rate": _avg([1.0 if r.hit else 0.0 for r in retrieval]),
        "mrr": _avg([(1.0 / r.rank if r.rank else 0.0) for r in retrieval]),
        "refusal_accuracy": _avg([1.0 if r.refusal_correct else 0.0 for r in refusals]),
        "keyword_recall": _avg([r.keywords_found for r in results]),
        "faithfulness": _avg([r.faithfulness for r in results]),
        "relevance": _avg([r.relevance for r in results]),
        "avg_seconds": _avg([r.seconds for r in results]),
    }


def run_eval(cfg: Config, retriever, llm, generation: bool,
             progress=lambda m: None) -> dict:
    golden = Path(__file__).parent / "golden.yaml"
    cases = load_cases(golden)
    results = []
    for i, case in enumerate(cases, 1):
        progress(f"[{i}/{len(cases)}] {case.question[:70]}")
        results.append(evaluate_case(case, retriever, llm, cfg, generation))

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generation": generation,
        "config": {
            "fused_top_k": cfg.retrieval.fused_top_k,
            "rerank_top_n": cfg.retrieval.rerank_top_n,
            "score_floor": cfg.retrieval.rerank_score_floor,
            "answer_model": cfg.llm.answer_model,
        },
        "summary": summarize(results),
        "results": [asdict(r) for r in results],
    }

    out_dir = cfg.resolve_path(cfg.paths.cache_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = sorted(out_dir.glob("report-*.json"))
    out = out_dir / f"report-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))
    report["report_path"] = str(out)

    if previous:
        prev = json.loads(previous[-1].read_text())["summary"]
        report["delta_vs_previous"] = {
            k: round(report["summary"][k] - prev[k], 3)
            for k in report["summary"]
            if isinstance(report["summary"].get(k), (int, float))
            and isinstance(prev.get(k), (int, float))
        }
    return report
