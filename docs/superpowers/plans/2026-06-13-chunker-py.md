# chunker.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `backend/ingestion/chunker.py` — split `ExtractedDocument` pages into overlapping token-bounded chunks with language detection and provenance metadata.

**Architecture:** Sentence-aware greedy accumulation, page-confined (no cross-page chunks). Each page's text is split into sentences via regex, then greedily packed into chunks of ≤1000 tokens (BAAI/bge-small-en-v1.5 tokenizer), with a 200-token overlap tail carried into the next chunk. Language detected per chunk via `langdetect` (seed fixed to 0 for determinism).

**Tech Stack:** Python 3.11+, HuggingFace `transformers.AutoTokenizer`, `langdetect`, `sentence-transformers` (already installed), pytest.

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `backend/ingestion/chunker.py` | Modify (was stub) | Full implementation, built up across tasks |
| `tests/ingestion/test_chunker.py` | Create | TDD tests, built up across tasks |

---

## Task 1: Chunk dataclass + module skeleton

**Files:**
- Modify: `backend/ingestion/chunker.py`
- Create: `tests/ingestion/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ingestion/test_chunker.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ingestion/test_chunker.py -v
```
Expected: `ImportError: cannot import name 'Chunk' from 'backend.ingestion.chunker'`

- [ ] **Step 3: Install dependencies**

Run:
```bash
pip install langdetect sentence-transformers
```
Expected: packages install successfully (transformers pulled in by sentence-transformers).

- [ ] **Step 4: Write module skeleton in backend/ingestion/chunker.py**

```python
"""Splits extracted page text into overlapping chunks with metadata."""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

import langdetect
from langdetect import DetectorFactory, LangDetectException
from transformers import AutoTokenizer

from backend.ingestion.extract import ExtractedDocument, PageContent

DetectorFactory.seed = 0  # make language detection deterministic

CHUNK_MAX_TOKENS = 1000
CHUNK_MIN_TOKENS = 500
OVERLAP_TOKENS = 200
TOKENIZER_MODEL = "BAAI/bge-small-en-v1.5"

log = logging.getLogger(__name__)

_tokenizer: AutoTokenizer | None = None


@dataclass
class Chunk:
    chunk_id: str
    pdf_id: str
    filename: str
    page_number: int
    text: str
    token_count: int
    language: str
    bbox: tuple[float, float, float, float] | None
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ingestion/test_chunker.py -v
```
Expected: 2 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add backend/ingestion/chunker.py tests/ingestion/test_chunker.py
git commit -m "feat: add Chunk dataclass and chunker module skeleton"
```

---

## Task 2: _split_sentences

**Files:**
- Modify: `backend/ingestion/chunker.py`
- Modify: `tests/ingestion/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/ingestion/test_chunker.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ingestion/test_chunker.py -k "split_sentences" -v
```
Expected: `ImportError: cannot import name '_split_sentences'`

- [ ] **Step 3: Implement _split_sentences in backend/ingestion/chunker.py**

Add after the `_tokenizer` line:

```python
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text) if s.strip()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ingestion/test_chunker.py -k "split_sentences" -v
```
Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/ingestion/chunker.py tests/ingestion/test_chunker.py
git commit -m "feat: implement _split_sentences"
```

---

## Task 3: _count_tokens

**Files:**
- Modify: `backend/ingestion/chunker.py`
- Modify: `tests/ingestion/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/ingestion/test_chunker.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ingestion/test_chunker.py -k "count_tokens" -v
```
Expected: `ImportError: cannot import name '_count_tokens'`

- [ ] **Step 3: Implement _count_tokens in backend/ingestion/chunker.py**

Add after `_split_sentences`:

```python
def _count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ingestion/test_chunker.py -k "count_tokens" -v
```
Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/ingestion/chunker.py tests/ingestion/test_chunker.py
git commit -m "feat: implement _count_tokens"
```

---

## Task 4: _detect_language

**Files:**
- Modify: `backend/ingestion/chunker.py`
- Modify: `tests/ingestion/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/ingestion/test_chunker.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ingestion/test_chunker.py -k "detect_language" -v
```
Expected: `ImportError: cannot import name '_detect_language'`

- [ ] **Step 3: Implement _detect_language in backend/ingestion/chunker.py**

Add after `_count_tokens`:

```python
def _detect_language(text: str) -> str:
    try:
        return langdetect.detect(text)
    except LangDetectException:
        return "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ingestion/test_chunker.py -k "detect_language" -v
```
Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/ingestion/chunker.py tests/ingestion/test_chunker.py
git commit -m "feat: implement _detect_language"
```

---

## Task 5: _chunk_page

**Files:**
- Modify: `backend/ingestion/chunker.py`
- Modify: `tests/ingestion/test_chunker.py`

The tests use a mock tokenizer where **1 token = 1 space-separated word**, and pass small `max_tokens`/`overlap_tokens` values so tests don't need thousand-word sentences.

- [ ] **Step 1: Write failing tests**

Append to `tests/ingestion/test_chunker.py`:

```python
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
    # chunk text = "Hello world. How are you?" — join with space → 5 words
    page = _make_page("Hello world. How are you?")
    chunks = _chunk_page(page, _word_tokenizer(), max_tokens=20, overlap_tokens=3)
    assert chunks[0].token_count == 5


def test_chunk_page_splits_when_exceeds_max_tokens():
    # max_tokens=6, overlap_tokens=3, 1 token/word
    # S1: "Hello world." = 2 tokens → buffer=[S1], tokens=2
    # S2: "How are you?" = 3 tokens → buffer=[S1,S2], tokens=5
    # S3: "I am fine." = 3 tokens → 5+3=8 > 6 → FLUSH [S1,S2]
    #   overlap: reversed([S1,S2]):
    #     S2 = 3 tokens, 0+3=3 ≤ 3 → overlap=[S2], overlap_tokens=3
    #     S1 = 2 tokens, 3+2=5 > 3 → stop
    #   new buffer = [S2, S3], tokens = 3+3 = 6
    # End → FLUSH [S2, S3]
    # Result: 2 chunks
    page = _make_page("Hello world. How are you? I am fine.")
    chunks = _chunk_page(page, _word_tokenizer(), max_tokens=6, overlap_tokens=3)
    assert len(chunks) == 2


def test_chunk_page_overlap_appears_in_next_chunk():
    # Same scenario as above: S2 ("How are you?") should appear in chunk 2
    page = _make_page("Hello world. How are you? I am fine.")
    chunks = _chunk_page(page, _word_tokenizer(), max_tokens=6, overlap_tokens=3)
    assert "How are you?" in chunks[1].text


def test_chunk_page_sub_minimum_last_chunk_kept():
    # A page with only a short sentence produces one chunk even if below CHUNK_MIN_TOKENS
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ingestion/test_chunker.py -k "chunk_page" -v
```
Expected: `ImportError: cannot import name '_chunk_page'`

- [ ] **Step 3: Implement _chunk_page in backend/ingestion/chunker.py**

Add after `_detect_language`:

```python
def _chunk_page(
    page: PageContent,
    tokenizer: AutoTokenizer,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    if not page.text.strip():
        return []

    sentences = _split_sentences(page.text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens: int = 0
    chunk_index: int = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence, tokenizer)

        if buffer and buffer_tokens + sentence_tokens > max_tokens:
            chunk_text = " ".join(buffer)
            chunk_id = hashlib.sha256(
                f"{page.pdf_id}:{page.page_number}:{chunk_index}".encode()
            ).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                pdf_id=page.pdf_id,
                filename=page.filename,
                page_number=page.page_number,
                text=chunk_text,
                token_count=buffer_tokens,
                language=_detect_language(chunk_text),
                bbox=page.bbox,
            ))
            chunk_index += 1

            overlap: list[str] = []
            overlap_token_count: int = 0
            for s in reversed(buffer):
                s_tokens = _count_tokens(s, tokenizer)
                if overlap_token_count + s_tokens <= overlap_tokens:
                    overlap.insert(0, s)
                    overlap_token_count += s_tokens
                else:
                    break

            buffer = overlap + [sentence]
            buffer_tokens = overlap_token_count + sentence_tokens
        else:
            buffer.append(sentence)
            buffer_tokens += sentence_tokens

    if buffer:
        chunk_text = " ".join(buffer)
        chunk_id = hashlib.sha256(
            f"{page.pdf_id}:{page.page_number}:{chunk_index}".encode()
        ).hexdigest()[:16]
        chunks.append(Chunk(
            chunk_id=chunk_id,
            pdf_id=page.pdf_id,
            filename=page.filename,
            page_number=page.page_number,
            text=chunk_text,
            token_count=buffer_tokens,
            language=_detect_language(chunk_text),
            bbox=page.bbox,
        ))

    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ingestion/test_chunker.py -k "chunk_page" -v
```
Expected: 11 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/ingestion/chunker.py tests/ingestion/test_chunker.py
git commit -m "feat: implement _chunk_page with sentence-aware overlap"
```

---

## Task 6: _get_tokenizer + chunk_document

**Files:**
- Modify: `backend/ingestion/chunker.py`
- Modify: `tests/ingestion/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/ingestion/test_chunker.py`:

```python
from backend.ingestion.chunker import chunk_document
from backend.ingestion.extract import ExtractedDocument, PageContent


def _make_doc(pages: list[PageContent]) -> ExtractedDocument:
    return ExtractedDocument(pdf_id="testpdf1234567", filename="test.pdf", pages=pages)


def test_chunk_document_empty_pages_returns_empty():
    with patch("backend.ingestion.chunker._get_tokenizer", return_value=_word_tokenizer()):
        result = chunk_document(_make_doc([]))
    assert result == []


def test_chunk_document_returns_chunks_from_all_pages():
    pages = [
        _make_page("Hello world. How are you?", page_number=1),
        _make_page("Second page has content too.", page_number=2),
    ]
    with patch("backend.ingestion.chunker._get_tokenizer", return_value=_word_tokenizer()):
        chunks = chunk_document(_make_doc(pages))
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2


def test_chunk_document_skips_empty_page_text():
    pages = [
        _make_page("", page_number=1),
        _make_page("This page has real content.", page_number=2),
    ]
    with patch("backend.ingestion.chunker._get_tokenizer", return_value=_word_tokenizer()):
        chunks = chunk_document(_make_doc(pages))
    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_chunk_document_chunk_ids_are_unique():
    pages = [
        _make_page("First page sentence one. First page sentence two.", page_number=1),
        _make_page("Second page sentence one. Second page sentence two.", page_number=2),
    ]
    with patch("backend.ingestion.chunker._get_tokenizer", return_value=_word_tokenizer()):
        chunks = chunk_document(_make_doc(pages))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_document_returns_list_of_chunk():
    pages = [_make_page("Some text here.")]
    with patch("backend.ingestion.chunker._get_tokenizer", return_value=_word_tokenizer()):
        chunks = chunk_document(_make_doc(pages))
    assert all(isinstance(c, Chunk) for c in chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ingestion/test_chunker.py -k "chunk_document" -v
```
Expected: `ImportError: cannot import name 'chunk_document'`

- [ ] **Step 3: Implement _get_tokenizer and chunk_document in backend/ingestion/chunker.py**

Add after `_chunk_page`:

```python
def _get_tokenizer() -> AutoTokenizer:
    """Lazy singleton: load the bge-small tokenizer once and keep it warm."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    return _tokenizer


def chunk_document(doc: ExtractedDocument) -> list[Chunk]:
    tokenizer = _get_tokenizer()
    chunks: list[Chunk] = []
    for page in doc.pages:
        chunks.extend(_chunk_page(page, tokenizer))
    return chunks
```

- [ ] **Step 4: Run full test suite**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ingestion/test_chunker.py -v
```
Expected: All tests PASSED (≥ 24 tests).

- [ ] **Step 5: Run the combined ingestion test suite to check for regressions**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ -v --tb=short 2>&1 | tail -10
```
Expected: All tests PASSED (44 extract tests + chunker tests, no failures).

- [ ] **Step 6: Commit**

```bash
git add backend/ingestion/chunker.py tests/ingestion/test_chunker.py
git commit -m "feat: implement chunk_document — complete chunking pipeline"
```

---

## Self-Review

**Spec coverage:**
- ✅ `Chunk` dataclass with all fields — Task 1
- ✅ `CHUNK_MAX_TOKENS=1000`, `CHUNK_MIN_TOKENS=500`, `OVERLAP_TOKENS=200` — Task 1
- ✅ `TOKENIZER_MODEL="BAAI/bge-small-en-v1.5"` — Task 1
- ✅ `DetectorFactory.seed = 0` (determinism) — Task 1
- ✅ `_split_sentences` via regex — Task 2
- ✅ `_count_tokens` with explicit tokenizer param — Task 3
- ✅ `_detect_language` with `"unknown"` fallback — Task 4
- ✅ `_chunk_page` with greedy accumulation + overlap + optional params for testability — Task 5
- ✅ Sub-minimum last chunk kept (not discarded) — Task 5, `test_chunk_page_sub_minimum_last_chunk_kept`
- ✅ `_get_tokenizer` lazy singleton — Task 6
- ✅ `chunk_document` public entry point — Task 6
- ✅ Page-confined chunking (no cross-page) — enforced by `_chunk_page` operating on one `PageContent` at a time

**Type consistency:**
- `_chunk_page(page: PageContent, tokenizer: AutoTokenizer, max_tokens: int, overlap_tokens: int) -> list[Chunk]` — consistent in Task 5 implementation and Task 6 call site (uses defaults)
- `chunk_document(doc: ExtractedDocument) -> list[Chunk]` — `ExtractedDocument` imported from `extract`, consistent throughout
- `Chunk.chunk_id` is always `hashlib.sha256(...).hexdigest()[:16]` — 16 hex chars, consistent with test assertions

**No placeholders found.**
