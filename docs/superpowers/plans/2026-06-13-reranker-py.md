# reranker.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `backend/retrieval/reranker.py` — an optional cross-encoder reranker that re-scores and truncates `list[SearchResult]` from top-20 to top-5, with a passthrough when disabled.

**Architecture:** Single public function `rerank(query, results, top_k=5, model=None, enabled=True) -> list[SearchResult]`. When disabled or results empty, returns `results[:top_k]` immediately without loading the model. When enabled, calls `CrossEncoder.predict([(query, text), ...])`, sorts descending by score, and returns top `top_k`.

**Tech Stack:** Python 3.11+, `sentence-transformers` (`CrossEncoder`), `numpy`, pytest.

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `backend/retrieval/reranker.py` | Modify (was stub) | Full implementation, built across 2 tasks |
| `tests/retrieval/test_reranker.py` | Create | TDD tests, built across 2 tasks |

---

## Task 1: Module skeleton + _get_model

**Files:**
- Modify: `backend/retrieval/reranker.py`
- Create: `tests/retrieval/test_reranker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/retrieval/test_reranker.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.retrieval.reranker import RERANK_MODEL, TOP_K_RERANK, _get_model


def test_constants_defined():
    assert RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert TOP_K_RERANK == 5


def test_get_model_loads_cross_encoder():
    with patch("backend.retrieval.reranker.CrossEncoder") as mock_ce:
        mock_ce.return_value = MagicMock()
        import backend.retrieval.reranker as reranker_mod
        reranker_mod._model = None
        result = _get_model()
    mock_ce.assert_called_once_with(RERANK_MODEL)
    assert result is mock_ce.return_value


def test_get_model_singleton_cached():
    with patch("backend.retrieval.reranker.CrossEncoder") as mock_ce:
        mock_ce.return_value = MagicMock()
        import backend.retrieval.reranker as reranker_mod
        reranker_mod._model = None
        r1 = _get_model()
        r2 = _get_model()
    assert r1 is r2
    mock_ce.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/retrieval/test_reranker.py -v
```
Expected: `ImportError: cannot import name 'RERANK_MODEL'`

- [ ] **Step 3: Replace backend/retrieval/reranker.py with skeleton**

```python
"""Optional cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)."""
from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import CrossEncoder

from backend.retrieval.vector_store import SearchResult

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANK = 5

log = logging.getLogger(__name__)

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Lazy singleton: load cross-encoder once and keep it warm."""
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/retrieval/test_reranker.py -v
```
Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/retrieval/reranker.py tests/retrieval/test_reranker.py
git commit -m "feat: add reranker skeleton with _get_model singleton"
```

---

## Task 2: rerank function

**Files:**
- Modify: `backend/retrieval/reranker.py`
- Modify: `tests/retrieval/test_reranker.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/retrieval/test_reranker.py`:

```python
import numpy as np

from backend.retrieval.reranker import TOP_K_RERANK, rerank
from backend.retrieval.vector_store import SearchResult


def _make_result(idx: int = 0, text: str = "chunk text") -> SearchResult:
    return SearchResult(
        chunk_id=f"chunk{idx:016d}",
        pdf_id="testpdf123456789",
        filename="doc.pdf",
        page_number=idx + 1,
        text=text,
        token_count=5,
        language="en",
        bbox=None,
        similarity=max(0.0, 0.9 - idx * 0.1),
    )


def _make_model(scores: list[float] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.predict.return_value = np.array(scores or [0.7])
    return mock


def test_rerank_disabled_returns_first_top_k():
    results = [_make_result(i) for i in range(10)]
    output = rerank("query", results, top_k=3, enabled=False)
    assert len(output) == 3
    assert output == results[:3]


def test_rerank_disabled_does_not_call_model():
    mock_model = _make_model([0.5])
    results = [_make_result(0)]
    rerank("query", results, top_k=1, model=mock_model, enabled=False)
    mock_model.predict.assert_not_called()


def test_rerank_empty_results_returns_empty():
    output = rerank("query", [], model=_make_model())
    assert output == []


def test_rerank_empty_results_does_not_call_model():
    mock_model = _make_model()
    rerank("query", [], model=mock_model)
    mock_model.predict.assert_not_called()


def test_rerank_sorts_by_score_descending():
    low = _make_result(0, text="low relevance")
    high = _make_result(1, text="high relevance")
    mock_model = _make_model([0.3, 0.9])
    output = rerank("query", [low, high], top_k=2, model=mock_model)
    assert output[0].text == "high relevance"
    assert output[1].text == "low relevance"


def test_rerank_truncates_to_top_k():
    results = [_make_result(i) for i in range(5)]
    mock_model = _make_model([0.5, 0.4, 0.3, 0.2, 0.1])
    output = rerank("query", results, top_k=3, model=mock_model)
    assert len(output) == 3


def test_rerank_passes_query_text_pairs_to_predict():
    result = _make_result(0, text="specific chunk text")
    mock_model = _make_model([0.7])
    rerank("my query", [result], top_k=1, model=mock_model)
    mock_model.predict.assert_called_once_with([("my query", "specific chunk text")])


def test_rerank_top_k_larger_than_results_returns_all_sorted():
    results = [_make_result(i) for i in range(3)]
    mock_model = _make_model([0.3, 0.9, 0.5])
    output = rerank("query", results, top_k=10, model=mock_model)
    assert len(output) == 3
    assert output[0].similarity == results[1].similarity  # highest score first


def test_rerank_default_top_k_is_top_k_rerank():
    results = [_make_result(i) for i in range(10)]
    mock_model = _make_model([float(i) * 0.1 for i in range(10)])
    output = rerank("query", results, model=mock_model)
    assert len(output) == TOP_K_RERANK


def test_rerank_uses_lazy_singleton_when_model_not_provided():
    results = [_make_result(0)]
    mock_model = _make_model([0.7])
    with patch("backend.retrieval.reranker._get_model", return_value=mock_model):
        rerank("query", results)
    mock_model.predict.assert_called_once()


def test_rerank_returns_list_of_search_result():
    results = [_make_result(0)]
    mock_model = _make_model([0.8])
    output = rerank("query", results, top_k=1, model=mock_model)
    assert all(isinstance(r, SearchResult) for r in output)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/retrieval/test_reranker.py -k "rerank" -v
```
Expected: `ImportError: cannot import name 'rerank'`

- [ ] **Step 3: Implement rerank in backend/retrieval/reranker.py**

Append after `_get_model`:

```python
def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = TOP_K_RERANK,
    model: CrossEncoder | None = None,
    enabled: bool = True,
) -> list[SearchResult]:
    """Rerank results with cross-encoder; when disabled returns results[:top_k]."""
    if not enabled or not results:
        return results[:top_k]

    if model is None:
        model = _get_model()

    scores = model.predict([(query, r.text) for r in results])
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_k]]
```

- [ ] **Step 4: Run full test suite**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/retrieval/test_reranker.py -v
```
Expected: All tests PASSED (≥ 14 tests).

- [ ] **Step 5: Run combined suite to verify no regressions**

Run:
```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ -q --tb=short 2>&1 | tail -5
```
Expected: all tests pass (114 existing + reranker tests).

- [ ] **Step 6: Commit**

```bash
git add backend/retrieval/reranker.py tests/retrieval/test_reranker.py
git commit -m "feat: implement rerank — cross-encoder reranker with enabled toggle"
```

---

## Self-Review

**Spec coverage:**
- ✅ `RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"`, `TOP_K_RERANK=5` — Task 1
- ✅ `_get_model()` lazy singleton — Task 1
- ✅ `rerank(query, results, top_k=TOP_K_RERANK, model=None, enabled=True) -> list[SearchResult]` — Task 2
- ✅ `enabled=False` → return `results[:top_k]`, no model load — Task 2
- ✅ `results` empty → return `[]`, no model load — Task 2
- ✅ `model.predict([(query, r.text) for r in results])` — Task 2
- ✅ Sort descending by score, truncate to `top_k` — Task 2
- ✅ `top_k >= len(results)` safe — Task 2 (Python slice handles it)
- ✅ Model injected via optional param (None → singleton) — Task 2

**Type consistency:**
- `rerank` takes `list[SearchResult]` (imported from `vector_store`) and returns `list[SearchResult]` ✅
- `scores` is `np.ndarray`, zipped with `results` — `zip(scores, results)` works on ndarray ✅
- `_get_model() -> CrossEncoder` — used as `model.predict(...)` ✅

**No placeholders found.**
