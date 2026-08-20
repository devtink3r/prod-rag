from pathlib import Path

import pytest

from rag.config import load_config
from rag.ingestion.chunker import TokenCounter, rule_title_of

CACHED = sorted((Path(__file__).parent.parent / ".cache" / "parsed").glob("*.json"))


def test_rule_title_detection():
    t = rule_title_of("3. Arrangements for declaration and payment of dividends within India. The arrangements referred to...")
    assert t is not None and t.startswith("Rule 3: Arrangements for declaration")
    assert rule_title_of("(a) some clause text") is None
    assert rule_title_of("2026. The year was") is None  # 4-digit numbers excluded
    assert rule_title_of("12. Definitions. In these rules") == "Rule 12: Definitions"


def test_token_counter_fallback():
    tc = TokenCounter("nonexistent/model")
    assert tc.count("x" * 400) == 100


@pytest.mark.skipif(not CACHED, reason="no cached parse available")
def test_chunk_real_document():
    from rag.ingestion.chunker import chunk_document
    from rag.ingestion.models import ParsedDoc

    cfg = load_config()
    parsed = ParsedDoc(
        doc_id="test", source_path=Path("x.pdf"), doc_type="pdf",
        title="Income-tax Rules 2026", docling_json_path=CACHED[0],
    )
    children, parents = chunk_document(parsed, cfg)
    assert children and parents
    assert all(c.parent_id for c in children)
    parent_ids = {p.chunk_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)
    # size discipline (heuristic counter): children bounded, parents bigger
    assert max(c.token_count for c in children) <= cfg.chunking.max_tokens * 1.3
    # rule detection found rule-level sections
    rule_sections = [c for c in children if any(p.startswith("Rule ") for p in c.section_path)]
    assert len(rule_sections) > 5
    # breadcrumb present in embed text
    assert children[0].embed_text.startswith("Income-tax Rules 2026")
    # tables kept atomic
    assert any(c.element_type == "table" for c in children)
