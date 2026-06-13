# DocuMitra extract.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the DocuMitra repo and implement `backend/ingestion/extract.py` — the PDF-to-text extraction stage that feeds `chunker.py`.

**Architecture:** Flat per-page pipeline: native text via PyMuPDF → scanned-page detection → full-page Tesseract OCR if needed → embedded image OCR appended → text normalization → document-level header/footer stripping → returns typed `ExtractedDocument` dataclass. Frontend scaffold is out of scope (separate plan).

**Tech Stack:** Python 3.11+, PyMuPDF (`fitz`), pytesseract, Pillow, pytest, pytest-mock.

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `pyproject.toml` | Create | pytest config + pythonpath |
| `requirements.txt` | Create | prod dependencies |
| `requirements-dev.txt` | Create | test dependencies |
| `backend/__init__.py` | Create | package marker |
| `backend/ingestion/__init__.py` | Create | package marker |
| `backend/ingestion/extract.py` | Create | full implementation (built up across tasks) |
| `backend/ingestion/chunker.py` | Create | stub |
| `backend/ingestion/embed.py` | Create | stub |
| `backend/retrieval/__init__.py` | Create | package marker |
| `backend/retrieval/vector_store.py` | Create | stub |
| `backend/retrieval/reranker.py` | Create | stub |
| `backend/generation/__init__.py` | Create | package marker |
| `backend/generation/llm_client.py` | Create | stub |
| `backend/generation/prompt_templates.py` | Create | stub |
| `backend/eval/__init__.py` | Create | package marker |
| `backend/eval/eval_runner.py` | Create | stub |
| `backend/main.py` | Create | stub |
| `data/pdfs/.gitkeep` | Create | empty dir marker |
| `data/cache/.gitkeep` | Create | empty dir marker |
| `scripts/ingest_all.py` | Create | stub |
| `scripts/cache_demo.py` | Create | stub |
| `tests/__init__.py` | Create | package marker |
| `tests/ingestion/__init__.py` | Create | package marker |
| `tests/ingestion/test_extract.py` | Create | TDD tests (built up across tasks) |

---

## Task 1: Scaffold repo skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: all stub modules and package markers

- [ ] **Step 1: Create pyproject.toml**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 2: Create requirements.txt**

```
pymupdf>=1.24.0
pytesseract>=0.3.10
Pillow>=10.4.0
fastapi>=0.111.0
uvicorn>=0.30.0
sentence-transformers>=3.0.0
supabase>=2.4.0
anthropic>=0.30.0
psycopg2-binary>=2.9.9
langdetect>=1.0.9
```

- [ ] **Step 3: Create requirements-dev.txt**

```
pytest>=8.2.0
pytest-mock>=3.14.0
```

- [ ] **Step 4: Create directory structure and package markers**

Run:
```bash
mkdir -p backend/ingestion backend/retrieval backend/generation backend/eval
mkdir -p data/pdfs data/cache scripts tests/ingestion
touch backend/__init__.py backend/ingestion/__init__.py
touch backend/retrieval/__init__.py backend/generation/__init__.py backend/eval/__init__.py
touch scripts/__init__.py tests/__init__.py tests/ingestion/__init__.py
touch data/pdfs/.gitkeep data/cache/.gitkeep
```

- [ ] **Step 5: Create backend/ingestion/chunker.py**

```python
"""Splits extracted page text into overlapping chunks with metadata."""
from __future__ import annotations
```

- [ ] **Step 6: Create backend/ingestion/embed.py**

```python
"""Encodes text chunks into vectors using BAAI/bge-small-en-v1.5."""
from __future__ import annotations
```

- [ ] **Step 7: Create backend/retrieval/vector_store.py**

```python
"""pgvector-backed top-k retrieval via Supabase."""
from __future__ import annotations
```

- [ ] **Step 8: Create backend/retrieval/reranker.py**

```python
"""Optional cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)."""
from __future__ import annotations
```

- [ ] **Step 9: Create backend/generation/llm_client.py**

```python
"""Anthropic primary + Ollama fallback LLM client."""
from __future__ import annotations
```

- [ ] **Step 10: Create backend/generation/prompt_templates.py**

```python
"""All LLM prompt templates — never inline strings."""
from __future__ import annotations
```

- [ ] **Step 11: Create backend/eval/eval_runner.py**

```python
"""Evaluation: p95 latency, R@k, MRR, citation accuracy, hallucination rate."""
from __future__ import annotations
```

- [ ] **Step 12: Create backend/main.py**

```python
"""FastAPI application entry point."""
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="DocuMitra")
```

- [ ] **Step 13: Create scripts/ingest_all.py**

```python
"""Runs full ingestion pipeline: extract → chunk → embed → store."""
from __future__ import annotations
```

- [ ] **Step 14: Create scripts/cache_demo.py**

```python
"""Pre-computes and caches responses for DEMO_MODE."""
from __future__ import annotations
```

- [ ] **Step 15: Install dev dependencies and verify pytest runs**

Run:
```bash
pip install -r requirements-dev.txt && pytest --collect-only
```
Expected: `no tests ran` (exit 0)

- [ ] **Step 16: Commit**

```bash
git init
git add .
git commit -m "chore: scaffold DocuMitra repo structure"
```

---

## Task 2: Data model — PageContent, ExtractedDocument, ExtractionError

**Files:**
- Create: `backend/ingestion/extract.py` (skeleton + data model)
- Create: `tests/ingestion/test_extract.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ingestion/test_extract.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/ingestion/test_extract.py -v
```
Expected: `ModuleNotFoundError: No module named 'backend.ingestion.extract'`

- [ ] **Step 3: Install prod dependencies**

Run:
```bash
pip install pymupdf pytesseract Pillow
```
Expected: packages install successfully.

- [ ] **Step 4: Create backend/ingestion/extract.py with data model**

```python
"""PDF → per-page text extraction (PyMuPDF + Tesseract)."""
from __future__ import annotations

import hashlib
import io
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

MIN_NATIVE_CHARS = 50
HEADER_LINES = 3
FOOTER_LINES = 3
HEURISTIC_THRESHOLD = 0.60
OCR_DPI = 300

log = logging.getLogger(__name__)


@dataclass
class PageContent:
    pdf_id: str
    filename: str
    page_number: int
    text: str
    bbox: tuple[float, float, float, float] | None
    is_ocr: bool


@dataclass
class ExtractedDocument:
    pdf_id: str
    filename: str
    pages: list[PageContent]


class ExtractionError(Exception):
    def __init__(self, filename: str, cause: Exception | str) -> None:
        self.filename = filename
        self.cause = cause
        super().__init__(f"Failed to extract '{filename}': {cause}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
pytest tests/ingestion/test_extract.py -v
```
Expected: 4 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add backend/ingestion/extract.py tests/ingestion/test_extract.py
git commit -m "feat: add PageContent, ExtractedDocument, ExtractionError data model"
```

---

## Task 3: _normalize_text

**Files:**
- Modify: `backend/ingestion/extract.py` (add `_normalize_text`)
- Modify: `tests/ingestion/test_extract.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/ingestion/test_extract.py`:

```python
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
    nfd = "é"  # e + combining acute accent (NFD)
    assert _normalize_text(nfd) == "\xe9"  # é in NFC


def test_normalize_strips_leading_trailing_whitespace():
    assert _normalize_text("  hello  ") == "hello"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/ingestion/test_extract.py -k "normalize" -v
```
Expected: `ImportError: cannot import name '_normalize_text'`

- [ ] **Step 3: Implement _normalize_text in backend/ingestion/extract.py**

Add after the `log = logging.getLogger(__name__)` line:

```python
def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("­", "")
    for ligature, replacement in (
        ("ﬁ", "fi"),
        ("ﬂ", "fl"),
        ("ﬀ", "ff"),
        ("ﬃ", "ffi"),
        ("ﬄ", "ffl"),
    ):
        text = text.replace(ligature, replacement)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/ingestion/test_extract.py -k "normalize" -v
```
Expected: 11 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/ingestion/extract.py tests/ingestion/test_extract.py
git commit -m "feat: implement _normalize_text"
```

---

## Task 4: _detect_scanned

**Files:**
- Modify: `backend/ingestion/extract.py`
- Modify: `tests/ingestion/test_extract.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/ingestion/test_extract.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/ingestion/test_extract.py -k "detect_scanned" -v
```
Expected: `ImportError: cannot import name '_detect_scanned'`

- [ ] **Step 3: Implement _detect_scanned in backend/ingestion/extract.py**

Add after `_normalize_text`:

```python
def _detect_scanned(text: str) -> bool:
    return len(text.strip()) < MIN_NATIVE_CHARS
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/ingestion/test_extract.py -k "detect_scanned" -v
```
Expected: 6 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/ingestion/extract.py tests/ingestion/test_extract.py
git commit -m "feat: implement _detect_scanned"
```

---

## Task 5: _strip_headers_footers

**Files:**
- Modify: `backend/ingestion/extract.py`
- Modify: `tests/ingestion/test_extract.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/ingestion/test_extract.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/ingestion/test_extract.py -k "strip" -v
```
Expected: `ImportError: cannot import name '_strip_headers_footers'`

- [ ] **Step 3: Implement _strip_headers_footers in backend/ingestion/extract.py**

Add after `_detect_scanned`:

```python
def _strip_headers_footers(pages: list[str]) -> list[str]:
    if not pages:
        return pages

    total = len(pages)
    line_counts: dict[str, int] = {}

    for text in pages:
        lines = text.splitlines()
        candidates = (
            [l.strip() for l in lines[:HEADER_LINES]]
            + [l.strip() for l in lines[-FOOTER_LINES:]]
        )
        for line in set(candidates):
            if line:
                line_counts[line] = line_counts.get(line, 0) + 1

    to_strip = {
        line for line, count in line_counts.items()
        if count / total >= HEURISTIC_THRESHOLD
    }

    if not to_strip:
        return pages

    return [
        "\n".join(l for l in text.splitlines() if l.strip() not in to_strip)
        for text in pages
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/ingestion/test_extract.py -k "strip" -v
```
Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/ingestion/extract.py tests/ingestion/test_extract.py
git commit -m "feat: implement _strip_headers_footers"
```

---

## Task 6: _ocr_page

**Files:**
- Modify: `backend/ingestion/extract.py`
- Modify: `tests/ingestion/test_extract.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/ingestion/test_extract.py`:

```python
from unittest.mock import MagicMock, patch

from backend.ingestion.extract import OCR_DPI, _ocr_page


def test_ocr_page_renders_at_correct_dpi():
    mock_page = MagicMock()
    mock_pixmap = MagicMock()
    mock_pixmap.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    mock_page.get_pixmap.return_value = mock_pixmap

    with patch("backend.ingestion.extract.Image.open"):
        with patch("backend.ingestion.extract.pytesseract.image_to_string", return_value="ocr result"):
            result = _ocr_page(mock_page)

    mock_page.get_pixmap.assert_called_once_with(dpi=OCR_DPI)
    assert result == "ocr result"


def test_ocr_page_passes_png_bytes_to_image_open():
    mock_page = MagicMock()
    mock_pixmap = MagicMock()
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    mock_pixmap.tobytes.return_value = png_bytes
    mock_page.get_pixmap.return_value = mock_pixmap

    with patch("backend.ingestion.extract.io.BytesIO") as mock_bytesio:
        with patch("backend.ingestion.extract.Image.open"):
            with patch("backend.ingestion.extract.pytesseract.image_to_string", return_value="text"):
                _ocr_page(mock_page)

    mock_pixmap.tobytes.assert_called_once_with("png")
    mock_bytesio.assert_called_once_with(png_bytes)


def test_ocr_page_uses_lang_eng():
    mock_page = MagicMock()
    mock_pixmap = MagicMock()
    mock_pixmap.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    mock_page.get_pixmap.return_value = mock_pixmap

    with patch("backend.ingestion.extract.Image.open") as mock_open:
        with patch("backend.ingestion.extract.pytesseract.image_to_string", return_value="text") as mock_tess:
            _ocr_page(mock_page)

    mock_tess.assert_called_once_with(mock_open.return_value, lang="eng")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/ingestion/test_extract.py -k "ocr_page" -v
```
Expected: `ImportError: cannot import name '_ocr_page'`

- [ ] **Step 3: Implement _ocr_page in backend/ingestion/extract.py**

Add after `_strip_headers_footers`:

```python
def _ocr_page(page: fitz.Page) -> str:
    pixmap = page.get_pixmap(dpi=OCR_DPI)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    return pytesseract.image_to_string(image, lang="eng")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/ingestion/test_extract.py -k "ocr_page" -v
```
Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/ingestion/extract.py tests/ingestion/test_extract.py
git commit -m "feat: implement _ocr_page"
```

---

## Task 7: _ocr_images

**Files:**
- Modify: `backend/ingestion/extract.py`
- Modify: `tests/ingestion/test_extract.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/ingestion/test_extract.py`:

```python
import logging

from backend.ingestion.extract import _ocr_images


def test_ocr_images_empty_when_no_images():
    mock_page = MagicMock()
    mock_page.get_images.return_value = []
    assert _ocr_images(mock_page, MagicMock()) == ""


def test_ocr_images_extracts_and_ocrs_single_image():
    mock_page = MagicMock()
    mock_doc = MagicMock()
    mock_page.get_images.return_value = [(42, 0, 0, 0, 0, "", "", 0)]

    mock_pixmap = MagicMock()
    mock_pixmap.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    with patch("backend.ingestion.extract.fitz.Pixmap", return_value=mock_pixmap):
        with patch("backend.ingestion.extract.Image.open"):
            with patch("backend.ingestion.extract.pytesseract.image_to_string", return_value="diagram label"):
                result = _ocr_images(mock_page, mock_doc)

    assert result == "diagram label"


def test_ocr_images_joins_multiple_images_with_newline():
    mock_page = MagicMock()
    mock_doc = MagicMock()
    mock_page.get_images.return_value = [
        (1, 0, 0, 0, 0, "", "", 0),
        (2, 0, 0, 0, 0, "", "", 0),
    ]

    mock_pixmap = MagicMock()
    mock_pixmap.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    with patch("backend.ingestion.extract.fitz.Pixmap", return_value=mock_pixmap):
        with patch("backend.ingestion.extract.Image.open"):
            with patch(
                "backend.ingestion.extract.pytesseract.image_to_string",
                side_effect=["text one", "text two"],
            ):
                result = _ocr_images(mock_page, mock_doc)

    assert result == "text one\ntext two"


def test_ocr_images_skips_failed_image_with_warning(caplog):
    mock_page = MagicMock()
    mock_doc = MagicMock()
    mock_page.get_images.return_value = [(99, 0, 0, 0, 0, "", "", 0)]

    with patch("backend.ingestion.extract.fitz.Pixmap", side_effect=RuntimeError("bad image")):
        with caplog.at_level(logging.WARNING, logger="backend.ingestion.extract"):
            result = _ocr_images(mock_page, mock_doc)

    assert result == ""
    assert "99" in caplog.text


def test_ocr_images_skips_blank_ocr_result():
    mock_page = MagicMock()
    mock_doc = MagicMock()
    mock_page.get_images.return_value = [(10, 0, 0, 0, 0, "", "", 0)]

    mock_pixmap = MagicMock()
    mock_pixmap.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    with patch("backend.ingestion.extract.fitz.Pixmap", return_value=mock_pixmap):
        with patch("backend.ingestion.extract.Image.open"):
            with patch("backend.ingestion.extract.pytesseract.image_to_string", return_value="   "):
                result = _ocr_images(mock_page, mock_doc)

    assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/ingestion/test_extract.py -k "ocr_images" -v
```
Expected: `ImportError: cannot import name '_ocr_images'`

- [ ] **Step 3: Implement _ocr_images in backend/ingestion/extract.py**

Add after `_ocr_page`:

```python
def _ocr_images(page: fitz.Page, doc: fitz.Document) -> str:
    texts: list[str] = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            pixmap = fitz.Pixmap(doc, xref)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image)
            if text.strip():
                texts.append(text.strip())
        except Exception as e:
            log.warning("Failed to OCR image xref=%d: %s", xref, e)
    return "\n".join(texts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/ingestion/test_extract.py -k "ocr_images" -v
```
Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/ingestion/extract.py tests/ingestion/test_extract.py
git commit -m "feat: implement _ocr_images"
```

---

## Task 8: extract_pdf — full integration

**Files:**
- Modify: `backend/ingestion/extract.py`
- Modify: `tests/ingestion/test_extract.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/ingestion/test_extract.py`:

```python
import pytest

from backend.ingestion.extract import extract_pdf


def _make_mock_page(text: str, images: list | None = None) -> MagicMock:
    """Helper: build a fitz.Page mock returning given native text and image list."""
    mock_page = MagicMock()
    mock_page.get_text.return_value = {
        "blocks": [
            {
                "type": 0,
                "lines": [{"spans": [{"text": text}]}],
            }
        ]
    }
    mock_rect = MagicMock()
    mock_rect.x0, mock_rect.y0, mock_rect.x1, mock_rect.y1 = 0.0, 0.0, 595.0, 842.0
    mock_page.rect = mock_rect
    mock_page.get_images.return_value = images or []
    return mock_page


def test_extract_pdf_returns_extracted_document(tmp_path):
    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"fake pdf bytes for hashing")

    mock_page = _make_mock_page(
        "This is page one with enough content to avoid triggering OCR."
    )
    mock_doc = MagicMock()
    mock_doc.__iter__ = lambda self: iter([mock_page])
    mock_doc.is_encrypted = False

    with patch("backend.ingestion.extract.fitz.open", return_value=mock_doc):
        result = extract_pdf(fake_pdf)

    assert isinstance(result, ExtractedDocument)
    assert result.filename == "sample.pdf"
    assert len(result.pages) == 1


def test_extract_pdf_pdf_id_is_16_hex_chars(tmp_path):
    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"deterministic content")

    mock_page = _make_mock_page(
        "Content long enough to avoid triggering OCR path in this test."
    )
    mock_doc = MagicMock()
    mock_doc.__iter__ = lambda self: iter([mock_page])
    mock_doc.is_encrypted = False

    with patch("backend.ingestion.extract.fitz.open", return_value=mock_doc):
        result = extract_pdf(fake_pdf)

    assert len(result.pdf_id) == 16
    assert all(c in "0123456789abcdef" for c in result.pdf_id)


def test_extract_pdf_pdf_id_is_deterministic(tmp_path):
    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"deterministic content")

    mock_page = _make_mock_page(
        "Content long enough to avoid triggering OCR path in this test."
    )
    mock_doc = MagicMock()
    mock_doc.__iter__ = lambda self: iter([mock_page])
    mock_doc.is_encrypted = False

    with patch("backend.ingestion.extract.fitz.open", return_value=mock_doc):
        r1 = extract_pdf(fake_pdf)
        r2 = extract_pdf(fake_pdf)

    assert r1.pdf_id == r2.pdf_id


def test_extract_pdf_page_numbers_are_one_indexed(tmp_path):
    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"bytes")

    pages = [
        _make_mock_page("Page one has enough content to be treated as native text."),
        _make_mock_page("Page two has enough content to be treated as native text."),
    ]
    mock_doc = MagicMock()
    mock_doc.__iter__ = lambda self: iter(pages)
    mock_doc.is_encrypted = False

    with patch("backend.ingestion.extract.fitz.open", return_value=mock_doc):
        result = extract_pdf(fake_pdf)

    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2


def test_extract_pdf_scanned_page_sets_is_ocr_true(tmp_path):
    fake_pdf = tmp_path / "scanned.pdf"
    fake_pdf.write_bytes(b"bytes")

    mock_page = _make_mock_page("")  # empty native text → triggers OCR
    mock_doc = MagicMock()
    mock_doc.__iter__ = lambda self: iter([mock_page])
    mock_doc.is_encrypted = False

    with patch("backend.ingestion.extract.fitz.open", return_value=mock_doc):
        with patch(
            "backend.ingestion.extract._ocr_page",
            return_value="scanned page content with sufficient text here",
        ):
            result = extract_pdf(fake_pdf)

    assert result.pages[0].is_ocr is True


def test_extract_pdf_native_page_sets_is_ocr_false(tmp_path):
    fake_pdf = tmp_path / "native.pdf"
    fake_pdf.write_bytes(b"bytes")

    mock_page = _make_mock_page(
        "This page has plenty of native text content to avoid OCR entirely."
    )
    mock_doc = MagicMock()
    mock_doc.__iter__ = lambda self: iter([mock_page])
    mock_doc.is_encrypted = False

    with patch("backend.ingestion.extract.fitz.open", return_value=mock_doc):
        result = extract_pdf(fake_pdf)

    assert result.pages[0].is_ocr is False


def test_extract_pdf_raises_on_corrupt_pdf(tmp_path):
    fake_pdf = tmp_path / "corrupt.pdf"
    fake_pdf.write_bytes(b"not a pdf")

    with patch(
        "backend.ingestion.extract.fitz.open",
        side_effect=RuntimeError("bad pdf"),
    ):
        with pytest.raises(ExtractionError) as exc_info:
            extract_pdf(fake_pdf)

    assert "corrupt.pdf" in str(exc_info.value)


def test_extract_pdf_raises_on_encrypted_pdf(tmp_path):
    fake_pdf = tmp_path / "locked.pdf"
    fake_pdf.write_bytes(b"bytes")

    mock_doc = MagicMock()
    mock_doc.is_encrypted = True

    with patch("backend.ingestion.extract.fitz.open", return_value=mock_doc):
        with pytest.raises(ExtractionError) as exc_info:
            extract_pdf(fake_pdf)

    assert "locked.pdf" in str(exc_info.value)


def test_extract_pdf_empty_pdf_returns_empty_pages(tmp_path):
    fake_pdf = tmp_path / "empty.pdf"
    fake_pdf.write_bytes(b"bytes")

    mock_doc = MagicMock()
    mock_doc.__iter__ = lambda self: iter([])
    mock_doc.is_encrypted = False

    with patch("backend.ingestion.extract.fitz.open", return_value=mock_doc):
        result = extract_pdf(fake_pdf)

    assert result.pages == []


def test_extract_pdf_bbox_populated_from_page_rect(tmp_path):
    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"bytes")

    mock_page = _make_mock_page(
        "Enough native text content to avoid triggering OCR path here."
    )
    mock_doc = MagicMock()
    mock_doc.__iter__ = lambda self: iter([mock_page])
    mock_doc.is_encrypted = False

    with patch("backend.ingestion.extract.fitz.open", return_value=mock_doc):
        result = extract_pdf(fake_pdf)

    assert result.pages[0].bbox == (0.0, 0.0, 595.0, 842.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/ingestion/test_extract.py -k "extract_pdf" -v
```
Expected: `ImportError: cannot import name 'extract_pdf'`

- [ ] **Step 3: Implement extract_pdf in backend/ingestion/extract.py**

Add after `_ocr_images`:

```python
def extract_pdf(path: Path) -> ExtractedDocument:
    file_bytes = path.read_bytes()
    pdf_id = hashlib.sha256(file_bytes).hexdigest()[:16]
    filename = path.name

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise ExtractionError(filename, e)

    if doc.is_encrypted:
        raise ExtractionError(filename, "PDF is password-protected")

    raw_texts: list[str] = []
    bboxes: list[tuple[float, float, float, float]] = []
    is_ocr_flags: list[bool] = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        native_text = "\n".join(
            span["text"]
            for block in blocks
            if block["type"] == 0
            for line in block["lines"]
            for span in line["spans"]
        )

        rect = page.rect
        bboxes.append((rect.x0, rect.y0, rect.x1, rect.y1))

        is_ocr = _detect_scanned(native_text)
        is_ocr_flags.append(is_ocr)

        if is_ocr:
            try:
                page_text = _ocr_page(page)
            except pytesseract.TesseractNotFoundError as e:
                raise ExtractionError(
                    filename,
                    f"Tesseract not installed: {e}. Install with: brew install tesseract",
                )
        else:
            page_text = native_text

        image_text = _ocr_images(page, doc)
        if image_text:
            page_text = page_text + "\n" + image_text

        raw_texts.append(page_text)

    stripped_texts = _strip_headers_footers(raw_texts)

    pages = [
        PageContent(
            pdf_id=pdf_id,
            filename=filename,
            page_number=i + 1,
            text=_normalize_text(text),
            bbox=bbox,
            is_ocr=is_ocr,
        )
        for i, (text, bbox, is_ocr) in enumerate(zip(stripped_texts, bboxes, is_ocr_flags))
    ]

    return ExtractedDocument(pdf_id=pdf_id, filename=filename, pages=pages)
```

- [ ] **Step 4: Run the full test suite**

Run:
```bash
pytest tests/ingestion/test_extract.py -v
```
Expected: All tests PASSED (≥40 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/ingestion/extract.py tests/ingestion/test_extract.py
git commit -m "feat: implement extract_pdf — complete PDF to ExtractedDocument pipeline"
```

---

## Out of Scope

- Frontend scaffold (Next.js) — separate plan
- `chunker.py`, `embed.py`, `vector_store.py` implementations — separate plans
- Language detection (per-chunk, belongs in `chunker.py`)
