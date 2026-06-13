from __future__ import annotations

from backend.ingestion.chunker import Chunk


def test_chunk_fields():
    chunk = Chunk(
        chunk_id="abc123def456abcd",
        pdf_id="deadbeef12345678",
        filename="report.pdf",
        page_number=3,
        text="This is chunk text.",
        token_count=4,
        language="en",
        bbox=(0.0, 0.0, 595.0, 842.0),
    )
    assert chunk.chunk_id == "abc123def456abcd"
    assert chunk.pdf_id == "deadbeef12345678"
    assert chunk.filename == "report.pdf"
    assert chunk.page_number == 3
    assert chunk.text == "This is chunk text."
    assert chunk.token_count == 4
    assert chunk.language == "en"
    assert chunk.bbox == (0.0, 0.0, 595.0, 842.0)


def test_chunk_bbox_none():
    chunk = Chunk(
        chunk_id="abc123def456abcd",
        pdf_id="deadbeef12345678",
        filename="report.pdf",
        page_number=1,
        text="Text.",
        token_count=1,
        language="en",
        bbox=None,
    )
    assert chunk.bbox is None
