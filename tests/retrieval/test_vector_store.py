from __future__ import annotations

from backend.retrieval.vector_store import RPC_FUNCTION, TOP_K, SearchResult


def test_constants_defined():
    assert TOP_K == 20
    assert RPC_FUNCTION == "match_chunks"


def test_search_result_fields():
    result = SearchResult(
        chunk_id="abc123def456abcd",
        pdf_id="testpdf123456789",
        filename="doc.pdf",
        page_number=3,
        text="Some chunk text here.",
        token_count=4,
        language="en",
        bbox=[0.0, 0.0, 595.0, 842.0],
        similarity=0.92,
    )
    assert result.chunk_id == "abc123def456abcd"
    assert result.pdf_id == "testpdf123456789"
    assert result.filename == "doc.pdf"
    assert result.page_number == 3
    assert result.text == "Some chunk text here."
    assert result.token_count == 4
    assert result.language == "en"
    assert result.bbox == [0.0, 0.0, 595.0, 842.0]
    assert result.similarity == 0.92


def test_search_result_bbox_none():
    result = SearchResult(
        chunk_id="abc123def456abcd",
        pdf_id="testpdf123456789",
        filename="doc.pdf",
        page_number=1,
        text="Text.",
        token_count=1,
        language="en",
        bbox=None,
        similarity=0.85,
    )
    assert result.bbox is None
