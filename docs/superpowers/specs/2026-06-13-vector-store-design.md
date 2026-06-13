# vector_store.py Design Spec
**Date:** 2026-06-13
**Scope:** `backend/retrieval/vector_store.py` — query text → top-k `SearchResult` objects via Supabase pgvector HNSW

---

## Context

DocuMitra retrieval stage. `vector_store.py` receives a raw query string, embeds it using the same `BAAI/bge-small-en-v1.5` model that produced the stored chunk vectors, and retrieves the top-k most similar chunks from Supabase pgvector via a cosine HNSW scan. Results feed the optional reranker and then the LLM generation stage.

---

## Decisions

- **Query embedding:** Done internally via `_get_model()` imported from `embed.py` — same singleton, model loaded only once per process.
- **Return type:** Typed `SearchResult` dataclass — consistent with the project's typed pattern (Chunk, ExtractedDocument, etc.).
- **Retrieval mechanism:** Supabase RPC (`match_chunks` Postgres function) — pgvector executes the HNSW scan entirely in Postgres, returning rows pre-sorted by similarity descending.

---

## Prerequisites (SQL migration — run once, out of scope for this module)

```sql
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(384),
    match_count int DEFAULT 20
)
RETURNS TABLE (
    chunk_id    text,
    pdf_id      text,
    filename    text,
    page_number int,
    text        text,
    token_count int,
    language    text,
    bbox        jsonb,
    similarity  float
)
LANGUAGE sql STABLE AS $$
    SELECT
        chunk_id, pdf_id, filename, page_number, text,
        token_count, language, bbox,
        1 - (embedding <=> query_embedding) AS similarity
    FROM chunks
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
```

---

## Data Model

```python
@dataclass
class SearchResult:
    chunk_id:    str
    pdf_id:      str
    filename:    str
    page_number: int
    text:        str
    token_count: int
    language:    str
    bbox:        list[float] | None  # JSONB array from Supabase, or null
    similarity:  float               # cosine similarity, 0.0–1.0
```

`bbox` is `list[float] | None` (deserialized from Supabase JSONB), distinct from `Chunk.bbox` which is `tuple[float,...] | None`.

---

## Constants

```python
TOP_K        = 20
RPC_FUNCTION = "match_chunks"
```

---

## Public Interface

```python
from backend.retrieval.vector_store import search, SearchResult
```

Only `search` and `SearchResult` are public.

**Entry point:**
```python
def search(
    query: str,
    client: Client,
    k: int = TOP_K,
    model: SentenceTransformer | None = None,
) -> list[SearchResult]:
    """Embed query and return top-k chunks by cosine similarity."""
```

`model` defaults to the lazy singleton from `embed.py` but can be injected for tests.

---

## Pipeline

### search

1. **Load model** — `model = _get_model()` if not provided. No-op after first ingestion call since the singleton is already warm.

2. **Embed query** — `vector = model.encode(query, normalize_embeddings=True)` → `np.ndarray` of shape `(384,)`. Single string, no batch dimension.

3. **RPC call** — `client.rpc(RPC_FUNCTION, {"query_embedding": vector.tolist(), "match_count": k}).execute()`. pgvector uses the HNSW index for cosine nearest-neighbour search entirely in Postgres.

4. **Deserialize** — map each row in `resp.data` to a `SearchResult`. `similarity = 1 - cosine_distance`, in `[0, 1]`.

5. **Return** — `list[SearchResult]`, length ≤ k.

---

## Error Handling

| Situation | Behavior |
|-----------|----------|
| `resp.data` empty (no chunks in DB) | Returns `[]`, no exception |
| `k=0` | Returns `[]` — `LIMIT 0` is valid SQL |
| Supabase unavailable / auth failure | `PostgrestAPIError` propagates to caller |
| `match_chunks` RPC not found (migration not run) | `PostgrestAPIError` propagates with clear Supabase message |
| Model not loaded | `_get_model()` loads transparently on first call |

---

## Dependencies

```
sentence-transformers>=3.0.0   # already in requirements.txt (via embed.py)
supabase>=2.4.0                # already in requirements.txt
numpy                          # transitive dep of sentence-transformers
```

Imports `_get_model` from `backend.ingestion.embed` to share the model singleton.

---

## Out of Scope

- Reranking (→ `reranker.py`)
- LLM generation (→ `llm_client.py`)
- Ingestion / embedding (→ `embed.py`)
- SQL migration to create `match_chunks` function
