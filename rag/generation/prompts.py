"""Prompt contract for grounded, cited answers."""

from rag.retrieval.retriever import ContextBlock

SYSTEM_PROMPT = """You are a precise assistant answering questions about documents \
provided as numbered context blocks.

Rules you must follow:
1. Answer ONLY from the context blocks. Never add facts from general knowledge; \
general knowledge may only help you phrase the answer.
2. Cite every claim with the block number in square brackets, e.g. [1] or [2][3]. \
Only cite block numbers that exist in the context.
3. If the context does not contain enough information to answer, say exactly: \
"I couldn't find this in the documents." — optionally noting what related \
information IS present. Do not guess.
4. Be concise. Answer the question asked, nothing more. Use the document's own \
terminology (rule numbers, section references).
5. Quote exact figures, thresholds, dates and form numbers verbatim from the context."""

REFUSAL_TEXT = "I couldn't find this in the documents."

CONDENSE_PROMPT = """Given the conversation history and a follow-up question, rewrite \
the follow-up as a single standalone question that preserves all context needed to \
answer it. Reply with only the rewritten question.

History:
{history}

Follow-up: {question}"""


def format_context(blocks: list[ContextBlock]) -> str:
    parts = []
    for i, b in enumerate(blocks, 1):
        where = " > ".join(b.section_path) or b.title
        pages = f"p.{b.page_start}" if b.page_start else ""
        if b.page_end and b.page_end != b.page_start:
            pages += f"-{b.page_end}"
        parts.append(f"[{i}] {b.title} — {where} ({pages})\n{b.text}")
    return "\n\n---\n\n".join(parts)


def build_messages(question: str, blocks: list[ContextBlock]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context blocks:\n\n{format_context(blocks)}\n\n"
                       f"Question: {question}",
        },
    ]
