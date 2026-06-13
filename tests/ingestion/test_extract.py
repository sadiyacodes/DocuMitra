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
