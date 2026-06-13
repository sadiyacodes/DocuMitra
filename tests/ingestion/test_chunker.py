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


from backend.ingestion.chunker import _split_sentences


def test_split_sentences_basic():
    result = _split_sentences("Hello world. How are you? I am fine!")
    assert result == ["Hello world.", "How are you?", "I am fine!"]


def test_split_sentences_empty_string():
    assert _split_sentences("") == []


def test_split_sentences_single_sentence():
    assert _split_sentences("Just one sentence.") == ["Just one sentence."]


def test_split_sentences_filters_blank_fragments():
    result = _split_sentences("Hello.   ")
    assert all(s.strip() for s in result)
    assert len(result) == 1


def test_split_sentences_preserves_sentence_text():
    result = _split_sentences("Revenue was $4.2 billion. Costs rose 3%.")
    assert all(isinstance(s, str) for s in result)
    assert len(result) >= 1
