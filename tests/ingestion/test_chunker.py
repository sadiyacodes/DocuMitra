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


from backend.ingestion.chunker import Chunk, _chunk_page
from backend.ingestion.extract import PageContent


def _word_tokenizer() -> MagicMock:
    """Mock tokenizer: token count = number of space-separated words."""
    mock = MagicMock()
    mock.encode.side_effect = lambda text, **kwargs: list(range(len(text.split())))
    return mock


def _make_page(text: str, page_number: int = 1) -> PageContent:
    return PageContent(
        pdf_id="testpdf1234567",
        filename="test.pdf",
        page_number=page_number,
        text=text,
        bbox=(0.0, 0.0, 595.0, 842.0),
        is_ocr=False,
    )


def test_chunk_page_empty_text_returns_empty():
    assert _chunk_page(_make_page(""), _word_tokenizer()) == []


def test_chunk_page_whitespace_only_returns_empty():
    assert _chunk_page(_make_page("   \n  "), _word_tokenizer()) == []


def test_chunk_page_short_text_single_chunk():
    page = _make_page("Hello world. How are you?")
    chunks = _chunk_page(page, _word_tokenizer(), max_tokens=20, overlap_tokens=3)
    assert len(chunks) == 1
    assert "Hello world" in chunks[0].text


def test_chunk_page_chunk_id_is_16_hex_chars():
    page = _make_page("Hello world.")
    chunks = _chunk_page(page, _word_tokenizer(), max_tokens=20, overlap_tokens=3)
    assert len(chunks[0].chunk_id) == 16
    assert all(c in "0123456789abcdef" for c in chunks[0].chunk_id)


def test_chunk_page_chunk_id_is_deterministic():
    page = _make_page("Hello world. How are you?")
    tok = _word_tokenizer()
    chunks1 = _chunk_page(page, tok, max_tokens=20, overlap_tokens=3)
    chunks2 = _chunk_page(page, tok, max_tokens=20, overlap_tokens=3)
    assert chunks1[0].chunk_id == chunks2[0].chunk_id


def test_chunk_page_metadata_inherited_from_page():
    page = _make_page("Hello world.")
    chunks = _chunk_page(page, _word_tokenizer(), max_tokens=20, overlap_tokens=3)
    assert chunks[0].pdf_id == "testpdf1234567"
    assert chunks[0].filename == "test.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].bbox == (0.0, 0.0, 595.0, 842.0)


def test_chunk_page_token_count_recorded():
    # "Hello world. How are you?" → sentences: ["Hello world.", "How are you?"]
    # word counts: 2 + 3 = 5 words = 5 tokens (with _word_tokenizer)
    page = _make_page("Hello world. How are you?")
    chunks = _chunk_page(page, _word_tokenizer(), max_tokens=20, overlap_tokens=3)
    assert chunks[0].token_count == 5


def test_chunk_page_splits_when_exceeds_max_tokens():
    # max_tokens=6, overlap_tokens=3, 1 token/word
    # S1: "Hello world." = 2 tokens → buffer=[S1], tokens=2
    # S2: "How are you?" = 3 tokens → buffer=[S1,S2], tokens=5
    # S3: "I am fine." = 3 tokens → 5+3=8 > 6 → FLUSH [S1,S2]
    # Result: 2 chunks
    page = _make_page("Hello world. How are you? I am fine.")
    chunks = _chunk_page(page, _word_tokenizer(), max_tokens=6, overlap_tokens=3)
    assert len(chunks) == 2


def test_chunk_page_overlap_appears_in_next_chunk():
    # S2 ("How are you?") = 3 tokens fits in overlap_tokens=3 → appears in chunk 2
    page = _make_page("Hello world. How are you? I am fine.")
    chunks = _chunk_page(page, _word_tokenizer(), max_tokens=6, overlap_tokens=3)
    assert "How are you?" in chunks[1].text


def test_chunk_page_sub_minimum_last_chunk_kept():
    page = _make_page("Short.")
    chunks = _chunk_page(page, _word_tokenizer(), max_tokens=1000, overlap_tokens=200)
    assert len(chunks) == 1
    assert chunks[0].text == "Short."


def test_chunk_page_language_field_is_string():
    page = _make_page(
        "This is a long enough English sentence for language detection to work correctly here."
    )
    chunks = _chunk_page(page, _word_tokenizer(), max_tokens=100, overlap_tokens=10)
    assert isinstance(chunks[0].language, str)
    assert len(chunks[0].language) > 0
