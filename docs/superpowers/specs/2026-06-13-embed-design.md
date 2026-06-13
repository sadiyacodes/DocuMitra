# embed.py Design Spec
**Date:** 2026-06-13
**Scope:** `backend/ingestion/embed.py` — `list[Chunk]` → embeddings → pgvector (Supabase)

---

## Context

DocuMitra ingestion stage 3. `embed.py` receives chunks from `chunker.py`, encodes each chunk's text into a 384-dimensional vector using `BAAI/bge-small-en-v1.5`, and upserts the vectors + metadata into a Supabase pgvector table. Already-stored chunks (identified by `chunk_id`) are skipped to satisfy the "never re-embed on every run" requirement.

---

## Decisions

- **Skip strategy:** Check pgvector by `chunk_id` before encoding. Single source of truth — no local file cache to keep in sync.
- **Pipeline shape:** Single public function `embed_chunks` (encoding + storage coupled). Matches how `ingest_all.py` calls it; `SentenceTransformer.encode()` is batch-optimized so coupling is an asset, not a liability.
- **Model:** `BAAI/bge-small-en-v1.5` via `sentence-transformers`. Already in `requirements.txt`.

---

## Supabase Table Schema

`embed.py` assumes this table exists. Schema migrations are run once separately (out of scope for this module).

```sql
CREATE TABLE chunks (
    chunk_id    TEXT PRIMARY KEY,
    pdf_id      TEXT NOT NULL,
    filename    TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    text        TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    language    TEXT NOT NULL,
    bbox        JSONB,
    embedding   vector(384) NOT NULL
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

`bbox` is stored as a JSON array `[x0, y0, x1, y1]` or SQL `NULL`.

---

## Constants

```python
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM   = 384
TABLE_NAME      = "chunks"
BATCH_SIZE      = 64
```

---

## Public Interface

```python
from backend.ingestion.embed import embed_chunks
```

Only `embed_chunks` is public. All helpers are private (`_` prefix).

**Entry point:**
```python
def embed_chunks(
    chunks: list[Chunk],
    client: Client,
    model: SentenceTransformer | None = None,
) -> int:
    """Encode new chunks and upsert to Supabase. Returns count of newly stored chunks."""
```

`model` defaults to the lazy singleton but can be injected for tests.

---

## Pipeline

### embed_chunks

1. **Early exit** — if `chunks` is empty, return `0` immediately (no model load, no I/O).

2. **Filter already-stored** — `_fetch_existing_ids({c.chunk_id for c in chunks}, client)` returns a `set[str]` of chunk_ids already in pgvector. Build `new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]`.

3. **Early exit** — if `new_chunks` is empty, return `0`.

4. **Batch encode** — `model.encode([c.text for c in new_chunks], batch_size=BATCH_SIZE, normalize_embeddings=True)` → `np.ndarray` of shape `(len(new_chunks), 384)`. `normalize_embeddings=True` required for cosine similarity correctness with bge-small-en-v1.5.

5. **Upsert** — `_upsert_rows(new_chunks, vectors, client)`.

6. **Return** — `len(new_chunks)`.

### _get_model() -> SentenceTransformer
Lazy singleton. Loads `SentenceTransformer(EMBEDDING_MODEL)` on first call, caches in module-level `_model`.

### _fetch_existing_ids(ids: set[str], client: Client) -> set[str]
Queries Supabase:
```python
resp = client.table(TABLE_NAME).select("chunk_id").in_("chunk_id", list(ids)).execute()
return {row["chunk_id"] for row in resp.data}
```
Returns empty set if no rows match.

### _upsert_rows(chunks: list[Chunk], vectors: np.ndarray, client: Client) -> None
Builds row dicts and upserts:
```python
rows = [
    {
        "chunk_id":    c.chunk_id,
        "pdf_id":      c.pdf_id,
        "filename":    c.filename,
        "page_number": c.page_number,
        "text":        c.text,
        "token_count": c.token_count,
        "language":    c.language,
        "bbox":        list(c.bbox) if c.bbox else None,
        "embedding":   vectors[i].tolist(),
    }
    for i, c in enumerate(chunks)
]
client.table(TABLE_NAME).upsert(rows).execute()
```

---

## Error Handling

| Situation | Behavior |
|-----------|----------|
| `chunks` is empty list | Return `0` immediately, no model load |
| All chunks already in pgvector | Return `0` after filter, no model load |
| Model download fails on first use | `RuntimeError` propagates to ingestion boundary |
| Supabase unavailable / bad credentials | `supabase.PostgrestAPIError` propagates to ingestion boundary |
| `_fetch_existing_ids` returns empty set (fresh table) | All chunks treated as new — normal path |

---

## Private Functions Summary

| Function | Signature | Purpose |
|----------|-----------|---------|
| `_get_model` | `() -> SentenceTransformer` | Lazy singleton for bge model |
| `_fetch_existing_ids` | `(ids: set[str], client: Client) -> set[str]` | Query pgvector for known chunk_ids |
| `_upsert_rows` | `(chunks: list[Chunk], vectors: np.ndarray, client: Client) -> None` | Batch upsert to Supabase |

---

## Dependencies

```
sentence-transformers>=3.0.0   # already in requirements.txt
supabase>=2.4.0                # already in requirements.txt
numpy                          # transitive dep of sentence-transformers
```

No new dependencies required.

---

## Out of Scope

- Table/index creation (run migrations separately)
- Retrieval (→ `vector_store.py`)
- Re-ranking (→ `reranker.py`)
- Query embedding (→ `vector_store.py`)
