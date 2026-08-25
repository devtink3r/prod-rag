"""LLM-as-judge for generation quality (RAGAS-style, no extra dependency).

faithfulness: is every claim in the answer supported by the context?
relevance: does the answer actually address the question?
Both 1-5. Fail-soft: returns {} on any error.
"""

import json
import re

from rag.config import Config
from rag.generation.prompts import format_context

_PROMPT = """You are evaluating a RAG system's answer.

Question: {question}

Context the system retrieved:
{context}

Answer the system gave:
{answer}

Score the answer on two axes, each 1-5:
- faithfulness: 5 = every claim is directly supported by the context; \
1 = contains claims not in the context (hallucination).
- relevance: 5 = fully answers the question asked; 1 = off-topic or empty.

Reply with ONLY this JSON: {{"faithfulness": <1-5>, "relevance": <1-5>}}"""


def judge_answer(question: str, answer: str, blocks, llm, cfg: Config) -> dict:
    try:
        raw = llm.complete(
            [{"role": "user", "content": _PROMPT.format(
                question=question,
                context=format_context(blocks)[:12000],
                answer=answer,
            )}],
            model=cfg.llm.utility_model,
            max_tokens=100,
        )
        match = re.search(r"\{[^{}]*\}", raw)
        if not match:
            return {}
        scores = json.loads(match.group(0))
        return {
            k: float(v) for k, v in scores.items()
            if k in ("faithfulness", "relevance") and 1 <= float(v) <= 5
        }
    except Exception:
        return {}
