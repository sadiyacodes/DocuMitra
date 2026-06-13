# chunker.py Design Spec
**Date:** 2026-06-13
**Scope:** `backend/ingestion/chunker.py` — `ExtractedDocument` → `list[Chunk]`, feeding `embed.py`

---

## Context

DocuMitra ingests 10+ PDFs (200+ pages each). `chunker.py` is the second ingestion stage: it receives the per-page text produced by `extract.py` and splits it into overlapping token windows that `embed.py` will encode into vectors. Each chunk carries full provenance metadata for citations.

---

## Decisions

- **Tokenizer:** HuggingFace `AutoTokenizer` for `BAAI/bge-small-en-v1.5` — exact token counts matching the embedding model. `sentence-transformers` already in `requirements.txt`, so no new dependency.
- **Page-confined chunking:** Chunks never cross page boundaries. `page_number` is always unambiguous, preserving citation accuracy.
- **Algorithm:** Sentence-aware greedy accumulation — chunks end on sentence boundaries for readable citations and better retrieval quality.

---

## Data Model

```python
@dataclass
class Chunk:
    chunk_id: str      # sha256(f"{pdf_id}:{page_number}:{chunk_index}")[:16]
    pdf_id: str
    filename: str
    page_number: int
    text: str
    token_count: int
    language: str      # ISO 639-1 e.g. "en"; "unknown" if detection fails
    bbox: tuple[float, float, float, float] | None  # inherited from PageContent
```

`chunk_id` is deterministic: same PDF bytes + same chunking params + same chunk index always produces the same id. This allows `embed.py` to skip re-embedding already-stored chunks.

---

## Public Interface

```python
from backend.ingestion.chunker import chunk_document, Chunk
```

Only `chunk_document` and `Chunk` are public. All helpers are private (`_` prefix).

**Entry point:**
```python
def chunk_document(doc: ExtractedDocument) -> list[Chunk]: ...
```

---

## Constants

```python
CHUNK_MAX_TOKENS = 1000        # flush chunk when next sentence would exceed this
CHUNK_MIN_TOKENS = 500         # informational; sub-minimum chunks are kept, not discarded
OVERLAP_TOKENS   = 200         # ~20% overlap; tail sentences carried into next chunk
TOKENIZER_MODEL  = "BAAI/bge-small-en-v1.5"
```

---

## Pipeline

### chunk_document
Iterates `doc.pages`, calls `_chunk_page(page)` for each, concatenates results.

### _chunk_page(page: PageContent) -> list[Chunk]
Per-page chunking:

1. **Sentence splitting** — `re.split(r'(?<=[.!?])\s+', page.text)` → `list[str]`. Abbreviation false-splits are acceptable. Empty strings filtered out.

2. **Greedy accumulation** — iterate sentences, call `_count_tokens(sentence, tokenizer)` for each, accumulate into a buffer. When adding the next sentence would exceed `CHUNK_MAX_TOKENS`, flush the buffer as a chunk.

3. **Overlap** — before flushing, compute the overlap tail: walk the buffer from the end, accumulating sentences until their combined token count reaches `OVERLAP_TOKENS`. Carry those sentences as the start of the next buffer.

4. **Final flush** — after iterating all sentences, flush the remaining buffer as the last chunk (even if below `CHUNK_MIN_TOKENS`).

5. **Empty page** — if `page.text.strip()` is empty, return `[]`.

6. **Language detection** — `langdetect.detect(chunk_text)` per chunk. Catches `LangDetectException`, defaults to `"unknown"`.

7. **chunk_id** — `sha256(f"{page.pdf_id}:{page.page_number}:{i}".encode()).hexdigest()[:16]` where `i` is the 0-indexed chunk position within the page.

### _get_tokenizer() -> AutoTokenizer
Lazy singleton. Loads `AutoTokenizer.from_pretrained(TOKENIZER_MODEL)` on first call, caches in module-level `_tokenizer`. Keeps model warm across calls.

### _count_tokens(text: str, tokenizer: AutoTokenizer) -> int
`len(tokenizer.encode(text, add_special_tokens=False))`

Tokenizer passed explicitly so unit tests can inject a real or mock tokenizer without touching the global.

### _split_sentences(text: str) -> list[str]
`[s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]`

### _detect_language(text: str) -> str
```python
try:
    return langdetect.detect(text)
except LangDetectException:
    return "unknown"
```

---

## Error Handling

| Situation | Behavior |
|-----------|----------|
| Empty page text | `_chunk_page` returns `[]`; `chunk_document` skips silently |
| `LangDetectException` (chunk too short) | `language = "unknown"`, no exception |
| Tokenizer download failure on first use | `RuntimeError` propagates to ingestion boundary |

---

## Private Functions Summary

| Function | Signature | Purpose |
|----------|-----------|---------|
| `_get_tokenizer` | `() -> AutoTokenizer` | Lazy singleton tokenizer |
| `_count_tokens` | `(text: str, tokenizer: AutoTokenizer) -> int` | Token count via bge tokenizer |
| `_split_sentences` | `(text: str) -> list[str]` | Regex sentence splitter |
| `_detect_language` | `(text: str) -> str` | langdetect wrapper, defaults "unknown" |
| `_chunk_page` | `(page: PageContent, tokenizer: AutoTokenizer) -> list[Chunk]` | Per-page chunking |

---

## Dependencies

```
sentence-transformers>=3.0.0   # already in requirements.txt; pulls in transformers + tokenizer
langdetect>=1.0.9              # already in requirements.txt
```

No new dependencies required.

---

## Out of Scope

- Embedding (→ `embed.py`)
- Storing to pgvector (→ `vector_store.py`)
- Header/footer stripping (already done in `extract.py`)
- Text normalization (already done in `extract.py`)
