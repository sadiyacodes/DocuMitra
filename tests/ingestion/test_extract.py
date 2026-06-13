from __future__ import annotations

from backend.ingestion.extract import (
    ExtractedDocument,
    ExtractionError,
    PageContent,
)


def test_page_content_fields():
    page = PageContent(
        pdf_id="abc123",
        filename="test.pdf",
        page_number=1,
        text="Hello world",
        bbox=(0.0, 0.0, 595.0, 842.0),
        is_ocr=False,
    )
    assert page.pdf_id == "abc123"
    assert page.filename == "test.pdf"
    assert page.page_number == 1
    assert page.text == "Hello world"
    assert page.bbox == (0.0, 0.0, 595.0, 842.0)
    assert page.is_ocr is False


def test_page_content_bbox_none():
    page = PageContent(
        pdf_id="abc123",
        filename="test.pdf",
        page_number=1,
        text="Hello",
        bbox=None,
        is_ocr=False,
    )
    assert page.bbox is None


def test_extracted_document_fields():
    pages = [
        PageContent(
            pdf_id="abc123",
            filename="test.pdf",
            page_number=1,
            text="Hello",
            bbox=None,
            is_ocr=False,
        )
    ]
    doc = ExtractedDocument(pdf_id="abc123", filename="test.pdf", pages=pages)
    assert doc.pdf_id == "abc123"
    assert doc.filename == "test.pdf"
    assert len(doc.pages) == 1


def test_extraction_error_message():
    err = ExtractionError("test.pdf", "something went wrong")
    assert "test.pdf" in str(err)
    assert "something went wrong" in str(err)


from backend.ingestion.extract import _normalize_text


def test_normalize_strips_null_bytes():
    assert _normalize_text("hello\x00world") == "helloworld"


def test_normalize_removes_soft_hyphens():
    assert _normalize_text("super­man") == "superman"


def test_normalize_fixes_fi_ligature():
    assert _normalize_text("ﬁle") == "file"


def test_normalize_fixes_fl_ligature():
    assert _normalize_text("ﬂoor") == "floor"


def test_normalize_fixes_ff_ligature():
    assert _normalize_text("ﬀ") == "ff"


def test_normalize_fixes_ffi_ligature():
    assert _normalize_text("ﬃ") == "ffi"


def test_normalize_fixes_ffl_ligature():
    assert _normalize_text("ﬄ") == "ffl"


def test_normalize_collapses_spaces_and_tabs():
    assert _normalize_text("hello   \t\tworld") == "hello world"


def test_normalize_collapses_excess_newlines():
    result = _normalize_text("hello\n\n\n\nworld")
    assert result == "hello\n\nworld"


def test_normalize_nfc_unicode():
    nfd = "é"  # e + combining acute accent (NFD)
    assert _normalize_text(nfd) == "\xe9"  # é in NFC


def test_normalize_strips_leading_trailing_whitespace():
    assert _normalize_text("  hello  ") == "hello"


from backend.ingestion.extract import MIN_NATIVE_CHARS, _detect_scanned


def test_detect_scanned_empty_string():
    assert _detect_scanned("") is True


def test_detect_scanned_whitespace_only():
    assert _detect_scanned("   \n\t  ") is True


def test_detect_scanned_below_threshold():
    assert _detect_scanned("x" * (MIN_NATIVE_CHARS - 1)) is True


def test_detect_scanned_at_threshold():
    assert _detect_scanned("x" * MIN_NATIVE_CHARS) is False


def test_detect_scanned_above_threshold():
    assert _detect_scanned("x" * (MIN_NATIVE_CHARS + 1)) is False


def test_detect_scanned_native_page():
    long_text = "This is a page with plenty of native text content for extraction." * 3
    assert _detect_scanned(long_text) is False


from backend.ingestion.extract import _strip_headers_footers


def test_strip_headers_footers_empty():
    assert _strip_headers_footers([]) == []


def test_strip_removes_repeated_header_and_footer():
    pages = [
        "HEADER\nPage one content here.\nFOOTER",
        "HEADER\nPage two content here.\nFOOTER",
        "HEADER\nPage three content here.\nFOOTER",
        "HEADER\nPage four content here.\nFOOTER",
        "HEADER\nPage five content here.\nFOOTER",
    ]
    result = _strip_headers_footers(pages)
    for page in result:
        assert "HEADER" not in page
        assert "FOOTER" not in page


def test_strip_keeps_unique_content_lines():
    pages = [
        "HEADER\nUnique line A.\nFOOTER",
        "HEADER\nUnique line B.\nFOOTER",
        "HEADER\nUnique line C.\nFOOTER",
        "HEADER\nUnique line D.\nFOOTER",
        "HEADER\nUnique line E.\nFOOTER",
    ]
    result = _strip_headers_footers(pages)
    for i, page in enumerate(result):
        assert f"Unique line {chr(65 + i)}." in page


def test_strip_does_not_strip_below_threshold():
    # "RARE_HEADER" appears in 2/5 pages = 40%, below 60% threshold
    pages = [
        "RARE_HEADER\nContent A.\nfooter",
        "RARE_HEADER\nContent B.\nfooter",
        "Content C.\nfooter",
        "Content D.\nfooter",
        "Content E.\nfooter",
    ]
    result = _strip_headers_footers(pages)
    assert any("RARE_HEADER" in p for p in result)


def test_strip_single_page_unchanged():
    pages = ["Only one page, nothing repeats, so nothing is stripped."]
    result = _strip_headers_footers(pages)
    assert result == pages
