# vector_store.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `backend/retrieval/vector_store.py` — embed a query string, run a cosine HNSW search via Supabase RPC, and return typed `SearchResult` objects with similarity scores.

**Architecture:** Single public function `search(query, client, k=20, model=None) -> list[SearchResult]`. Internally: load the bge-small-en-v1.5 model singleton from `embed.py`, encode the query with `normalize_embeddings=True`, call the `match_chunks` Postgres RPC via the Supabase client, and deserialize rows into `SearchResult` dataclasses.

**Tech Stack:** Python 3.11+, `sentence-transformers` (shared singleton from `embed.py`), `supabase` Python client, `numpy`, pytest.

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `backend/retrieval/vector_store.py` | Modify (was stub) | Full implementation, built across 2 tasks |
| `tests/retrieval/__init__.py` | Create | Package marker for retrieval tests |
| `tests/retrieval/test_vector_store.py` | Create | TDD tests, built across 2 tasks |

---

## Task 1: SearchResult dataclass + module skeleton

**Files:**
- Modify: `backend/retrieval/vector_store.py`
- Create: `tests/retrieval/__init__.py`
- Create: `tests/retrieval/test_vector_store.py`

- [ ] **Step 1: Create test package marker**

Run:
```bash
touch /Users/sadiya/projects/DocuMitra/tests/retrieval/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/retrieval/test_vector_store.py`:

```python
from __future__ import annotations

from backend.ingestion.embed import TABLE_NAME
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/retrieval/test_vector_store.py -v
```
Expected: `ImportError: cannot import name 'SearchResult'`

- [ ] **Step 4: Replace backend/retrieval/vector_store.py with skeleton**

```python
"""pgvector-backed top-k retrieval via Supabase."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import Client

from backend.ingestion.embed import _get_model

TOP_K = 20
RPC_FUNCTION = "match_chunks"

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id:    str
    pdf_id:      str
    filename:    str
    page_number: int
    text:        str
    token_count: int
    language:    str
    bbox:        list[float] | None
    similarity:  float
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/retrieval/test_vector_store.py -v
```
Expected: 3 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add backend/retrieval/vector_store.py tests/retrieval/__init__.py tests/retrieval/test_vector_store.py
git commit -m "feat: add SearchResult dataclass and vector_store skeleton"
```

---

## Task 2: search function

**Files:**
- Modify: `backend/retrieval/vector_store.py`
- Modify: `tests/retrieval/test_vector_store.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/retrieval/test_vector_store.py`:

```python
import numpy as np
from unittest.mock import MagicMock, patch

from backend.retrieval.vector_store import RPC_FUNCTION, TOP_K, SearchResult, search


def _make_row(similarity: float = 0.92) -> dict:
    return {
        "chunk_id":    "abc123def456abcd",
        "pdf_id":      "testpdf123456789",
        "filename":    "doc.pdf",
        "page_number": 3,
        "text":        "Some chunk text here.",
        "token_count": 4,
        "language":    "en",
        "bbox":        [0.0, 0.0, 595.0, 842.0],
        "similarity":  similarity,
    }


def _make_model() -> MagicMock:
    mock = MagicMock()
    mock.encode.return_value = np.zeros(384)
    return mock


def _make_client(rows: list[dict] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.rpc.return_value.execute.return_value.data = rows or []
    return mock


def test_search_returns_empty_list_when_no_results():
    result = search("query", _make_client([]), model=_make_model())
    assert result == []


def test_search_maps_row_to_search_result():
    row = _make_row()
    results = search("query", _make_client([row]), model=_make_model())
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, SearchResult)
    assert r.chunk_id == row["chunk_id"]
    assert r.pdf_id == row["pdf_id"]
    assert r.filename == row["filename"]
    assert r.page_number == row["page_number"]
    assert r.text == row["text"]
    assert r.token_count == row["token_count"]
    assert r.language == row["language"]
    assert r.bbox == row["bbox"]
    assert r.similarity == row["similarity"]


def test_search_result_bbox_none_preserved():
    row = _make_row()
    row["bbox"] = None
    results = search("query", _make_client([row]), model=_make_model())
    assert results[0].bbox is None


def test_search_encodes_query_with_normalize_embeddings():
    mock_model = _make_model()
    search("what is revenue?", _make_client(), model=mock_model)
    mock_model.encode.assert_called_once_with("what is revenue?", normalize_embeddings=True)


def test_search_calls_rpc_with_correct_function():
    mock_client = _make_client()
    search("query", mock_client, model=_make_model())
    assert mock_client.rpc.call_args[0][0] == RPC_FUNCTION


def test_search_passes_vector_as_list_to_rpc():
    mock_model = _make_model()
    mock_model.encode.return_value = np.ones(384) * 0.5
    mock_client = _make_client()
    search("query", mock_client, model=mock_model)
    rpc_kwargs = mock_client.rpc.call_args[0][1]
    assert isinstance(rpc_kwargs["query_embedding"], list)
    assert len(rpc_kwargs["query_embedding"]) == 384
    assert all(isinstance(v, float) for v in rpc_kwargs["query_embedding"])


def test_search_passes_k_as_match_count():
    mock_client = _make_client()
    search("query", mock_client, k=5, model=_make_model())
    rpc_kwargs = mock_client.rpc.call_args[0][1]
    assert rpc_kwargs["match_count"] == 5


def test_search_default_k_is_top_k():
    mock_client = _make_client()
    search("query", mock_client, model=_make_model())
    rpc_kwargs = mock_client.rpc.call_args[0][1]
    assert rpc_kwargs["match_count"] == TOP_K


def test_search_uses_lazy_singleton_when_model_not_provided():
    mock_model = _make_model()
    mock_client = _make_client()
    with patch("backend.retrieval.vector_store._get_model", return_value=mock_model):
        search("query", mock_client)
    mock_model.encode.assert_called_once()


def test_search_returns_multiple_results_in_order():
    rows = [_make_row(similarity=0.95), _make_row(similarity=0.80)]
    results = search("query", _make_client(rows), model=_make_model())
    assert len(results) == 2
    assert results[0].similarity == 0.95
    assert results[1].similarity == 0.80
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/retrieval/test_vector_store.py -k "search" -v
```
Expected: `ImportError: cannot import name 'search'`

- [ ] **Step 3: Implement search in backend/retrieval/vector_store.py**

Append after the `SearchResult` dataclass:

```python
def search(
    query: str,
    client: Client,
    k: int = TOP_K,
    model: SentenceTransformer | None = None,
) -> list[SearchResult]:
    """Embed query and return top-k chunks by cosine similarity."""
    if model is None:
        model = _get_model()

    vector = model.encode(query, normalize_embeddings=True)

    resp = client.rpc(
        RPC_FUNCTION,
        {"query_embedding": vector.tolist(), "match_count": k},
    ).execute()

    return [
        SearchResult(
            chunk_id=row["chunk_id"],
            pdf_id=row["pdf_id"],
            filename=row["filename"],
            page_number=row["page_number"],
            text=row["text"],
            token_count=row["token_count"],
            language=row["language"],
            bbox=row["bbox"],
            similarity=row["similarity"],
        )
        for row in resp.data
    ]
```

- [ ] **Step 4: Run full test suite**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/retrieval/test_vector_store.py -v
```
Expected: All tests PASSED (≥ 14 tests).

- [ ] **Step 5: Run combined suite to verify no regressions**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ -q --tb=short 2>&1 | tail -5
```
Expected: all tests pass (101 existing + vector_store tests).

- [ ] **Step 6: Commit**

```bash
git add backend/retrieval/vector_store.py tests/retrieval/test_vector_store.py
git commit -m "feat: implement search — embed query and return top-k SearchResult via pgvector RPC"
```

---

## Self-Review

**Spec coverage:**
- ✅ `SearchResult` dataclass with all fields — Task 1
- ✅ `TOP_K=20`, `RPC_FUNCTION="match_chunks"` — Task 1
- ✅ `_get_model()` imported from `embed.py` (shared singleton) — Task 1
- ✅ `search(query, client, k=TOP_K, model=None) -> list[SearchResult]` — Task 2
- ✅ `model.encode(query, normalize_embeddings=True)` — Task 2
- ✅ `vector.tolist()` passed to RPC — Task 2
- ✅ `match_count=k` passed to RPC — Task 2
- ✅ Rows deserialized to `SearchResult` — Task 2
- ✅ Empty result → `[]` — Task 2
- ✅ `bbox` is `list[float] | None` (not tuple) — Task 1 dataclass, Task 2 row pass-through

**Type consistency:**
- `SearchResult.bbox: list[float] | None` — populated as `row["bbox"]` which is a JSONB list or null ✅
- `vector.tolist()` produces `list[float]` matching pgvector expectation ✅
- `model.encode(query, ...)` single string → shape `(384,)` ndarray ✅

**No placeholders found.**
