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


from unittest.mock import MagicMock

from backend.ingestion.chunker import _count_tokens


def test_count_tokens_calls_encode_without_special_tokens():
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    result = _count_tokens("hello world test", mock_tokenizer)
    assert result == 5
    mock_tokenizer.encode.assert_called_once_with("hello world test", add_special_tokens=False)


def test_count_tokens_empty_string():
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = []
    assert _count_tokens("", mock_tokenizer) == 0


def test_count_tokens_returns_int():
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [10, 20]
    assert isinstance(_count_tokens("hi", mock_tokenizer), int)


from unittest.mock import patch

from langdetect import LangDetectException

from backend.ingestion.chunker import _detect_language


def test_detect_language_english():
    text = (
        "This is a sufficiently long English sentence for language detection "
        "to work reliably and return the correct language code."
    )
    assert _detect_language(text) == "en"


def test_detect_language_too_short_returns_unknown():
    with patch(
        "backend.ingestion.chunker.langdetect.detect",
        side_effect=LangDetectException(0, ""),
    ):
        result = _detect_language("hi")
    assert result == "unknown"


def test_detect_language_returns_string():
    text = "Enough text for detection to work without raising an exception here."
    assert isinstance(_detect_language(text), str)
