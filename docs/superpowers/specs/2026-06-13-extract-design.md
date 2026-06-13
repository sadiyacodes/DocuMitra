# extract.py Design Spec
**Date:** 2026-06-13
**Scope:** `backend/ingestion/extract.py` — PDF → per-page text with metadata, ready for `chunker.py`

---

## Context

DocuMitra ingests 10+ PDFs (200+ pages each) into a pgvector store for RAG. `extract.py` is the first stage: it converts raw PDF files into structured per-page text objects that `chunker.py` can split and embed. It must handle both native-text PDFs and scanned (image-only) PDFs, extract text from embedded diagrams/images, and strip repeated headers/footers before the text reaches the chunker.

---

## Data Model

```python
@dataclass
class PageContent:
    pdf_id: str          # SHA-256 of file bytes, first 16 hex chars
    filename: str
    page_number: int     # 1-indexed
    text: str            # normalized, header/footer stripped
    bbox: tuple[float, float, float, float] | None  # page bounding box (x0,y0,x1,y1)
    is_ocr: bool         # True if full-page OCR was used for this page

@dataclass
class ExtractedDocument:
    pdf_id: str
    filename: str
    pages: list[PageContent]
```

`pdf_id` is deterministic: `sha256(file_bytes).hexdigest()[:16]`. No clock or random dependency.

---

## Public Interface

```python
from backend.ingestion.extract import extract_pdf, ExtractedDocument, PageContent, ExtractionError
```

Only these four names are public. All internal functions are prefixed with `_`.

**Entry point:**
```python
def extract_pdf(path: Path) -> ExtractedDocument: ...
```

---

## Pipeline (Approach A — flat per-page)

Each PDF goes through these steps in order:

### Step 1: Native text extraction
- `fitz.open(path)` (PyMuPDF)
- Per page: `page.get_text("dict")` → collect text blocks with bboxes
- Concatenate block text into raw page string
- Page-level bbox from `page.rect`

### Step 2: Scanned-page detection
- If `len(raw_text.strip()) < MIN_NATIVE_CHARS` (constant: `50`), page is flagged as scanned
- Threshold chosen to catch pages with only a stray artifact character from PyMuPDF

### Step 3: Full-page OCR (scanned pages only)
- `page.get_pixmap(dpi=300)` → PIL Image
- `pytesseract.image_to_string(image, lang="eng")` → replaces raw page text
- `is_ocr = True` on the resulting `PageContent`

### Step 4: Embedded image OCR (all pages)
- `page.get_images(full=True)` → list of image references
- For each image: extract via `fitz.Pixmap`, convert to PIL Image
- `pytesseract.image_to_string(image)` → append to page text with `\n` separator (regardless of whether the page itself was OCR'd in Step 3)
- Per-image failures: log warning, skip — do not abort the page

### Step 5: Text normalization
Applied to every page's text string:
- Strip null bytes (`\x00`)
- Unicode NFC normalization (`unicodedata.normalize`)
- Collapse repeated whitespace (spaces, tabs, multiple newlines → single newline)
- Fix common OCR ligature artifacts: `ﬁ→fi`, `ﬂ→fl`, `ﬀ→ff`, `ﬃ→ffi`, `ﬄ→ffl`
- Remove soft hyphens (`­`)

### Step 6: Header/footer stripping (document-level post-pass)
- Collect the first `HEADER_LINES = 3` and last `FOOTER_LINES = 3` lines from each page
- A line is a header/footer candidate if it appears (exact match, after stripping) in ≥ `HEURISTIC_THRESHOLD = 0.60` (60%) of pages
- Remove all matching lines from the top/bottom of every page

### Step 7: Assemble ExtractedDocument
- Wrap all `PageContent` objects
- `pdf_id = sha256(path.read_bytes()).hexdigest()[:16]`

---

## Error Handling

| Situation | Behavior |
|-----------|----------|
| Corrupt or password-protected PDF | Raise `ExtractionError(filename, cause)` |
| Tesseract not installed | Raise `ExtractionError` with install instructions |
| Per-image OCR failure | `logging.warning(...)`, skip image, continue |
| Empty PDF (0 pages) | Return `ExtractedDocument` with `pages=[]`, no exception |

`ExtractionError` is a custom exception defined in `extract.py`:
```python
class ExtractionError(Exception):
    def __init__(self, filename: str, cause: Exception | str) -> None: ...
```

---

## Constants (all named, never magic numbers)

```python
MIN_NATIVE_CHARS = 50       # below this → page treated as scanned
HEADER_LINES = 3            # lines checked at top of each page
FOOTER_LINES = 3            # lines checked at bottom of each page
HEURISTIC_THRESHOLD = 0.60  # fraction of pages a line must appear in to be stripped
OCR_DPI = 300               # render resolution for full-page OCR
```

---

## Private Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `_detect_scanned` | `(text: str) -> bool` | Returns True if page needs OCR |
| `_ocr_page` | `(page: fitz.Page) -> str` | Full-page Tesseract OCR |
| `_ocr_images` | `(page: fitz.Page, doc: fitz.Document) -> str` | Extract + OCR embedded images |
| `_normalize_text` | `(text: str) -> str` | Unicode fix, whitespace, ligatures |
| `_strip_headers_footers` | `(pages: list[str]) -> list[str]` | Document-level header/footer removal |

All private functions accept plain Python types so they can be tested without a real PDF.

---

## Dependencies

```
pymupdf          # fitz
pytesseract      # Tesseract wrapper
Pillow           # PIL Image
```

Tesseract binary must be installed separately (`brew install tesseract` / `apt-get install tesseract-ocr`).

---

## Out of Scope for extract.py

- Chunking (→ `chunker.py`)
- Language detection (→ `chunker.py`, per-chunk)
- Embedding (→ `embed.py`)
- Storing to pgvector (→ `vector_store.py`)
