from rag.ingestion.cleaners import (
    find_boilerplate,
    is_toc_line,
    normalize_for_matching,
    normalize_markdown,
    normalize_text,
)


def test_normalize_text_fixes_unicode_and_hyphenation():
    assert normalize_text("infor-\nmation") == "information"
    assert normalize_text("â€œquoteâ€") == '"quote"'
    assert normalize_text("a   b\t c") == "a b c"
    assert normalize_text("‖Form‖ means") == '"Form" means'


def test_normalize_markdown_preserves_tables_and_code():
    md = "# T\n\n| a   | b |\n|---|---|\n\n```\nx   =   1\n```\n\n\n\ntext   here\n"
    out = normalize_markdown(md)
    assert "| a   | b |" in out
    assert "x   =   1" in out
    assert "text here" in out
    assert "\n\n\n" not in out


def test_toc_line_detection():
    assert is_toc_line("Chapter 2 .......... 41")
    assert not is_toc_line("Normal sentence ending. 41 rules apply")


def test_boilerplate_frequency():
    pages = {
        p: [normalize_for_matching(f"Page {p}"), normalize_for_matching("Income Tax Rules 2026")]
        for p in range(1, 11)
    }
    pages[1].append("unique intro line")
    bp = find_boilerplate(pages, 0.6)
    assert normalize_for_matching("Page 5") in bp
    assert normalize_for_matching("Income Tax Rules 2026") in bp
    assert "unique intro line" not in bp


def test_boilerplate_skips_short_docs():
    pages = {1: ["x"], 2: ["x"]}
    assert find_boilerplate(pages, 0.6) == set()
