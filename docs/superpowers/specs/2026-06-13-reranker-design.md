# reranker.py Design Spec
**Date:** 2026-06-13
**Scope:** `backend/retrieval/reranker.py` — optional cross-encoder reranker, top-20 → top-5

---

## Context

DocuMitra retrieval stage 5 (optional). `reranker.py` receives the top-20 `SearchResult` objects from `vector_store.search()`, scores each `(query, chunk_text)` pair with `cross-encoder/ms-marco-MiniLM-L-6-v2`, re-sorts by score, and returns the top-5. When disabled (via `enabled=False`), it truncates to `top_k` without loading the model — satisfying the CLAUDE.md requirement to "toggle off if it breaks the latency budget."

---

## Decisions

- **Toggle:** `enabled: bool = True` parameter on `rerank()`. When False, returns `results[:top_k]` with no model load.
- **Return type:** `list[SearchResult]` — same type as the input; no new dataclass needed.
- **Disabled passthrough:** Returns `results[:top_k]` (not the full list) so the caller always receives ≤ `top_k` results whether reranking is on or off.

---

## Constants

```python
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANK = 5
```

---

## Public Interface

```python
from backend.retrieval.reranker import rerank
```

Only `rerank` is public. `_get_model` is private.

**Entry point:**
```python
def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = TOP_K_RERANK,
    model: CrossEncoder | None = None,
    enabled: bool = True,
) -> list[SearchResult]:
    """Rerank results with cross-encoder; when disabled returns results[:top_k]."""
```

`model` defaults to the lazy singleton but can be injected for tests.

---

## Pipeline

### rerank

1. **Early exits** — `if not enabled or not results: return results[:top_k]`. No model load in either case.

2. **Load model** — `model = _get_model()` if not provided. Lazy singleton: loads `CrossEncoder(RERANK_MODEL)` on first call (~500ms), instant thereafter.

3. **Score** — `scores = model.predict([(query, r.text) for r in results])` → `np.ndarray` of shape `(len(results),)`. Higher score = more relevant to query.

4. **Sort & truncate** — `sorted(zip(scores, results), key=lambda x: x[0], reverse=True)` → return `[r for _, r in ranked[:top_k]]`.

### _get_model() -> CrossEncoder
Lazy singleton. Loads `CrossEncoder(RERANK_MODEL)` on first call, caches in module-level `_model`. Separate from `embed.py`'s `SentenceTransformer` singleton — different model class.

---

## Error Handling

| Situation | Behavior |
|-----------|----------|
| `enabled=False` | Return `results[:top_k]`, no model load |
| `results` is empty list | Return `[]`, no model load |
| `top_k >= len(results)` | Return all results re-sorted (Python slice is safe) |
| Model download fails | `RuntimeError` propagates to caller |
| `model.predict()` raises | Exception propagates — not caught here |

---

## Dependencies

```
sentence-transformers>=3.0.0   # already in requirements.txt; CrossEncoder is included
numpy                          # transitive dep; already available
```

No new dependencies required. `CrossEncoder` is part of the `sentence-transformers` package alongside `SentenceTransformer`.

---

## Out of Scope

- Scoring threshold filtering (only truncating by rank, not by score value)
- LLM generation (→ `llm_client.py`)
- Retrieval (→ `vector_store.py`)
