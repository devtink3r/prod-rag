"""Cleaning + normalization on the parsed document.

Two levels:
1. Element-level (DoclingDocument): drop boilerplate text items that repeat
   across pages (headers/footers/watermarks the layout model missed).
2. String-level (markdown): unicode fixes, dehyphenation, whitespace.

Structure-aware: table and code content is never touched by string cleanup.
"""

import re
from collections import Counter

import ftfy

_WS_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d+\s*$")


def normalize_text(text: str) -> str:
    """Unicode + whitespace normalization for a block of prose."""
    text = ftfy.fix_text(text, normalization="NFKC")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def normalize_markdown(md: str) -> str:
    """String-level cleanup of exported markdown, skipping tables/code."""
    out: list[str] = []
    in_code = False
    for line in md.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code or stripped.startswith("|"):
            out.append(line)
            continue
        out.append(normalize_text(line) if stripped else "")
    md = "\n".join(out)
    md = _MULTI_BLANK_RE.sub("\n\n", md)
    return md.strip() + "\n"


def is_toc_line(line: str) -> bool:
    return bool(_DOT_LEADER_RE.search(line.strip()))


def find_boilerplate(
    page_texts: dict[int, list[str]], frequency: float
) -> set[str]:
    """Text items repeating on more than `frequency` of pages = boilerplate.

    `page_texts`: page number -> list of normalized text-item strings.
    Returns the set of boilerplate strings (normalized).
    """
    num_pages = len(page_texts)
    if num_pages < 4:
        return set()
    counts: Counter[str] = Counter()
    for texts in page_texts.values():
        for t in set(texts):
            counts[t] += 1
    threshold = max(3, int(num_pages * frequency))
    return {
        t
        for t, c in counts.items()
        if c >= threshold and len(t) < 120 and not t.startswith("#")
    }


def normalize_for_matching(text: str) -> str:
    """Loose normalization so 'Page 3' and 'Page 47' both match."""
    t = normalize_text(text).lower()
    t = re.sub(r"\d+", "#", t)
    return t
