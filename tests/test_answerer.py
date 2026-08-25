from rag.config import load_config
from rag.generation.answerer import answer_question, validate_citations
from rag.generation.prompts import REFUSAL_TEXT, format_context
from rag.retrieval.retriever import ContextBlock, RetrievalResult


class FakeRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, q):
        return self.result


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, messages, **kw):
        self.calls.append(messages)
        return self.reply


def block(text, score=0.9):
    return ContextBlock(text=text, score=score, doc_id="d", title="IT Rules",
                        source_path="/a.pdf", section_path=["Rule 6: Holding"],
                        page_start=2, page_end=3)


def test_validate_citations_drops_invalid():
    text, cited = validate_citations("True [1] and fake [7]. Also [2].", num_blocks=2)
    assert text == "True [1] and fake . Also [2]."
    assert cited == {1, 2}


def test_answer_happy_path_filters_sources_to_cited():
    result = RetrievalResult([block("A"), block("B", 0.5)], no_answer=False, top_score=0.9)
    ans = answer_question("q", FakeRetriever(result), FakeLLM("Answer [2] only."),
                          load_config())
    assert ans.text == "Answer [2] only."
    assert [s.n for s in ans.sources] == [2]
    assert ans.sources[0].pages == "p.2-3"


def test_answer_keeps_all_sources_when_none_cited():
    result = RetrievalResult([block("A")], no_answer=False, top_score=0.9)
    ans = answer_question("q", FakeRetriever(result), FakeLLM("No markers here."),
                          load_config())
    assert [s.n for s in ans.sources] == [1]


def test_refusal_path_skips_llm():
    llm = FakeLLM("should never be called")
    result = RetrievalResult([], no_answer=True, top_score=0.01)
    ans = answer_question("q", FakeRetriever(result), llm, load_config())
    assert ans.no_answer and ans.text == REFUSAL_TEXT
    assert llm.calls == []


def test_format_context_numbers_blocks():
    ctx = format_context([block("first text"), block("second text")])
    assert "[1] IT Rules — Rule 6: Holding (p.2-3)" in ctx
    assert "[2]" in ctx and "second text" in ctx
