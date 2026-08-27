from pathlib import Path

from rag.eval.runner import CaseResult, _rule_rank, load_cases, summarize
from rag.retrieval.retriever import ContextBlock


def block(rule):
    return ContextBlock(text="t", score=0.9, doc_id="d", title="T",
                        source_path="/a", section_path=[rule])


def test_resolve_device_fallback():
    from rag.config import resolve_device

    dev, warn = resolve_device("cpu")
    assert dev == "cpu" and warn is None
    dev, warn = resolve_device("auto")
    assert dev in ("cpu", "cuda") and warn is None
    dev, warn = resolve_device("gpu")  # sandbox/CI has no CUDA -> fallback
    if dev == "cpu":
        assert warn and "falling back" in warn


def test_golden_set_loads():
    cases = load_cases(Path(__file__).parent.parent / "rag" / "eval" / "golden.yaml")
    assert len(cases) >= 10
    assert any(c.expect_refusal for c in cases)
    assert all(c.expect_rule or c.expect_refusal for c in cases)


def test_rule_rank_matches_prefix():
    blocks = [block("Rule 3: Dividends"), block("Rule 6: Holding")]
    assert _rule_rank(blocks, "Rule 6") == 2
    assert _rule_rank(blocks, "Rule 3") == 1
    assert _rule_rank(blocks, "Rule 66") is None  # no prefix false-positive


def test_summarize():
    results = [
        CaseResult(question="a", hit=True, rank=1, faithfulness=5, seconds=1),
        CaseResult(question="b", hit=False, rank=None, faithfulness=3, seconds=1),
        CaseResult(question="c", refusal_correct=True, seconds=1),
    ]
    s = summarize(results)
    assert s["hit_rate"] == 0.5
    assert s["mrr"] == 0.5
    assert s["refusal_accuracy"] == 1.0
    assert s["faithfulness"] == 4.0
