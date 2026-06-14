# Enterprise RAG Additions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing DocuMitra RAG pipeline to support multi-format ingestion (CSV, JSON), JWT-based RBAC, query-aware routing, similarity threshold filtering, and structured source attribution in the SSE stream.

**Architecture:** A source-agnostic `Chunk` dataclass (adding `source_id`, `source_type`, `access_roles`) becomes the single contract across ingestion, embedding, and retrieval. JWT auth lives in a dedicated `backend/auth/` module and enforces RBAC at the `search()` call via a Supabase RPC filter. The frontend gains a login page, passes Bearer tokens, and renders a sources panel per answer.

**Tech Stack:** FastAPI, Supabase pgvector, PyJWT, passlib[bcrypt], sentence-transformers, Next.js 14 + TypeScript

**Execution order is strictly sequential** — each phase modifies types relied on by the next.

---

## File Map

**Modified:**
- `backend/ingestion/chunker.py` — add `source_id`, `source_type`, `access_roles` to `Chunk`; `chunk_document` gains `access_roles` param
- `backend/ingestion/embed.py` — `_upsert_rows` adds new fields to row dicts
- `backend/retrieval/vector_store.py` — `SearchResult` updated; `search()` gains `user_role`, `min_similarity`, `source_types` params
- `backend/main.py` — auth router, RBAC on all endpoints, CSV/JSON endpoints, SSE sources event
- `frontend/lib/api.ts` — `StreamEvent` union type, named-event SSE parsing, Bearer token
- `frontend/hooks/useStreamingChat.ts` — `Message.sources`, token forwarding, sources event handler
- `frontend/components/chat/MessageList.tsx` — sources panel per assistant message
- `requirements.txt` — add `PyJWT`, `passlib[bcrypt]`

**Created:**
- `backend/auth/__init__.py`
- `backend/auth/models.py` — `User` dataclass, `get_user()`, `load_users()`
- `backend/auth/jwt_utils.py` — `create_token`, `decode_token`
- `backend/auth/dependencies.py` — `get_current_user` FastAPI dependency
- `backend/auth/router.py` — `POST /auth/login`
- `backend/ingestion/ingest_csv.py` — CSV → `list[Chunk]`
- `backend/ingestion/ingest_json.py` — JSON → `list[Chunk]`
- `backend/retrieval/router.py` — keyword-based source-type routing
- `frontend/lib/auth.ts` — `login`, `logout`, `getToken`
- `frontend/app/login/page.tsx` — login UI
- `scripts/seed_users.py` — generates `data/users.json` with bcrypt hashes
- `scripts/generate_synthetic_data.py` — generates CSV + JSON datasets
- `data/users.json` (output of seed script)
- `data/employees.csv` (output of synthetic data script)
- `data/audit_logs.json` (output of synthetic data script)
- `data/access_policies.json`
- `tests/auth/__init__.py`
- `tests/auth/test_jwt_utils.py`
- `tests/auth/test_auth_router.py`
- `tests/ingestion/test_ingest_csv.py`
- `tests/ingestion/test_ingest_json.py`
- `tests/retrieval/test_router.py`

**Supabase SQL (run in Supabase dashboard → SQL Editor):**
- `scripts/migration_enterprise.sql`

---

## Task 1: Supabase Schema Migration

**Files:**
- Create: `scripts/migration_enterprise.sql`

- [ ] **Step 1: Write the migration SQL**

Create `scripts/migration_enterprise.sql`:

```sql
-- 1. Add new columns
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_id    text;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_type  text NOT NULL DEFAULT 'pdf';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS access_roles text[] NOT NULL DEFAULT '{}';

-- 2. Backfill source_id from existing pdf_id values
UPDATE chunks SET source_id = pdf_id WHERE source_id IS NULL;

-- 3. Make source_id required
ALTER TABLE chunks ALTER COLUMN source_id SET NOT NULL;

-- 4. Drop the old match_chunks function (signature change requires DROP first)
DROP FUNCTION IF EXISTS match_chunks(vector, int);

-- 5. Create updated match_chunks RPC
CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding vector(384),
  match_count     int,
  user_role       text    DEFAULT NULL,
  min_similarity  float   DEFAULT 0.4
)
RETURNS TABLE (
  chunk_id     text,
  source_id    text,
  source_type  text,
  filename     text,
  page_number  int,
  text         text,
  token_count  int,
  language     text,
  bbox         float[],
  access_roles text[],
  similarity   float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    chunk_id,
    source_id,
    source_type,
    filename,
    page_number,
    text,
    token_count,
    language,
    bbox,
    access_roles,
    1 - (embedding <=> query_embedding) AS similarity
  FROM chunks
  WHERE
    (user_role IS NULL OR access_roles @> ARRAY[user_role] OR access_roles = '{}')
    AND (1 - (embedding <=> query_embedding)) >= min_similarity
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;
```

- [ ] **Step 2: Run migration in Supabase**

Open Supabase dashboard → SQL Editor → paste the file contents → Run.
Verify: `SELECT column_name FROM information_schema.columns WHERE table_name = 'chunks';` should show `source_id`, `source_type`, `access_roles`.

- [ ] **Step 3: Commit**

```bash
git add scripts/migration_enterprise.sql
git commit -m "feat: supabase schema migration — source_id, source_type, access_roles, updated match_chunks RPC"
```

---

## Task 2: Update Chunk Dataclass and chunk_document

**Files:**
- Modify: `backend/ingestion/chunker.py`
- Modify: `tests/ingestion/test_chunker.py`

- [ ] **Step 1: Find all tests currently asserting `chunk.pdf_id`**

```bash
grep -n "pdf_id" tests/ingestion/test_chunker.py
```

Note the line numbers — those assertions will need updating.

- [ ] **Step 2: Update failing test assertions first (TDD)**

In `tests/ingestion/test_chunker.py`, find every line that references `chunk.pdf_id` and update:
- `chunk.pdf_id` → `chunk.source_id`
- Add `assert chunk.source_type == "pdf"` alongside each `source_id` assertion
- Add `assert chunk.access_roles == []` to at least one test

Also add these two new tests at the bottom of the file:

```python
def test_chunk_document_passes_access_roles():
    doc = make_doc("Hello world. " * 50)  # use the same fixture/helper already in the test file
    chunks = chunk_document(doc, access_roles=["hr", "admin"])
    assert all(c.access_roles == ["hr", "admin"] for c in chunks)


def test_chunk_has_source_type_pdf():
    doc = make_doc("Hello world. " * 50)
    chunks = chunk_document(doc)
    assert all(c.source_type == "pdf" for c in chunks)
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd /Users/sadiya/projects/DocuMitra
python -m pytest tests/ingestion/test_chunker.py -v 2>&1 | tail -20
```

Expected: multiple failures on `AttributeError: 'Chunk' object has no attribute 'source_id'`.

- [ ] **Step 4: Update the Chunk dataclass in chunker.py**

Replace the `Chunk` dataclass:

```python
@dataclass
class Chunk:
    chunk_id:     str
    source_id:    str   # stable identifier for the source document (was pdf_id)
    source_type:  str   # "pdf" | "csv" | "json"
    filename:     str
    page_number:  int
    text:         str
    token_count:  int
    language:     str
    bbox:         tuple[float, float, float, float] | None
    access_roles: list[str]
```

- [ ] **Step 5: Update _chunk_page to produce new Chunk fields**

In `_chunk_page`, every `Chunk(...)` construction currently passes `pdf_id=page.pdf_id`. Change both occurrences (mid-page flush and end-of-page flush) to:

```python
Chunk(
    chunk_id=chunk_id,
    source_id=page.pdf_id,   # pdf_id is still valid for PDFs
    source_type="pdf",
    filename=page.filename,
    page_number=page.page_number,
    text=chunk_text,
    token_count=buffer_tokens,
    language=_detect_language(chunk_text),
    bbox=page.bbox,
    access_roles=access_roles,
)
```

- [ ] **Step 6: Add access_roles param to _chunk_page and chunk_document**

Update `_chunk_page` signature:

```python
def _chunk_page(
    page: PageContent,
    tokenizer: AutoTokenizer,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    access_roles: list[str] | None = None,
) -> list[Chunk]:
    if access_roles is None:
        access_roles = []
    ...
```

Update `chunk_document` signature:

```python
def chunk_document(
    doc: ExtractedDocument,
    access_roles: list[str] | None = None,
) -> list[Chunk]:
    """Split all pages of an extracted document into overlapping chunks."""
    tokenizer = _get_tokenizer()
    chunks: list[Chunk] = []
    for page in doc.pages:
        chunks.extend(_chunk_page(page, tokenizer, access_roles=access_roles or []))
    return chunks
```

- [ ] **Step 7: Run tests to confirm they pass**

```bash
python -m pytest tests/ingestion/test_chunker.py -v 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add backend/ingestion/chunker.py tests/ingestion/test_chunker.py
git commit -m "feat: add source_id, source_type, access_roles to Chunk dataclass; chunk_document accepts access_roles param"
```

---

## Task 3: Update embed.py Row Dicts

**Files:**
- Modify: `backend/ingestion/embed.py`
- Modify: `tests/ingestion/test_embed.py`

- [ ] **Step 1: Find upsert row assertions in tests**

```bash
grep -n "pdf_id\|upsert\|rows" tests/ingestion/test_embed.py
```

- [ ] **Step 2: Update test assertions first**

Find any test that checks the structure of rows passed to `client.table(...).upsert()`. Update the expected dict keys:
- Remove `"pdf_id"` key assertions
- Add `"source_id"`, `"source_type"`, `"access_roles"` key assertions

Add this new test if not already present:

```python
def test_upsert_rows_include_source_fields():
    from backend.ingestion.chunker import Chunk
    import numpy as np
    from unittest.mock import MagicMock, patch

    chunk = Chunk(
        chunk_id="abc123",
        source_id="src-01",
        source_type="csv",
        filename="test.csv",
        page_number=1,
        text="hello world",
        token_count=2,
        language="en",
        bbox=None,
        access_roles=["admin", "hr"],
    )
    vectors = np.zeros((1, 384))
    client = MagicMock()

    with patch("backend.ingestion.embed._fetch_existing_ids", return_value=set()):
        from backend.ingestion.embed import _upsert_rows
        _upsert_rows([chunk], vectors, client)

    call_args = client.table.return_value.upsert.call_args[0][0]
    row = call_args[0]
    assert row["source_id"] == "src-01"
    assert row["source_type"] == "csv"
    assert row["access_roles"] == ["admin", "hr"]
    assert "pdf_id" not in row
```

- [ ] **Step 3: Run test to confirm it fails**

```bash
python -m pytest tests/ingestion/test_embed.py::test_upsert_rows_include_source_fields -v
```

Expected: FAIL — `"source_id" not in row` or similar.

- [ ] **Step 4: Update _upsert_rows in embed.py**

Find the `rows = [...]` list comprehension in `_upsert_rows` and replace it:

```python
rows = [
    {
        "chunk_id":     c.chunk_id,
        "source_id":    c.source_id,
        "source_type":  c.source_type,
        "filename":     c.filename,
        "page_number":  c.page_number,
        "text":         c.text,
        "token_count":  c.token_count,
        "language":     c.language,
        "bbox":         list(c.bbox) if c.bbox is not None else None,
        "embedding":    vectors[i].tolist(),
        "access_roles": c.access_roles,
    }
    for i, c in enumerate(chunks)
]
```

- [ ] **Step 5: Run all embed tests**

```bash
python -m pytest tests/ingestion/test_embed.py -v 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backend/ingestion/embed.py tests/ingestion/test_embed.py
git commit -m "feat: embed.py upserts source_id, source_type, access_roles columns"
```

---

## Task 4: Update SearchResult and search()

**Files:**
- Modify: `backend/retrieval/vector_store.py`
- Modify: `tests/retrieval/test_vector_store.py`

- [ ] **Step 1: Update test mock data and assertions**

In `tests/retrieval/test_vector_store.py`, find the mock RPC response dict(s) and add the new fields. Typical mock row currently has `pdf_id` — update every fake row dict:

```python
MOCK_ROW = {
    "chunk_id":     "abc123",
    "source_id":    "src-001",     # was pdf_id
    "source_type":  "pdf",          # new
    "filename":     "test.pdf",
    "page_number":  1,
    "text":         "sample text",
    "token_count":  10,
    "language":     "en",
    "bbox":         None,
    "access_roles": ["admin"],      # new
    "similarity":   0.85,
}
```

Update all assertions from `result.pdf_id` → `result.source_id`, and add:

```python
def test_search_result_has_source_fields():
    # Use the existing mock_client fixture in the test file
    # (or replicate the mock setup already used by other tests)
    result = search("test query", mock_client)
    assert result[0].source_id == "src-001"
    assert result[0].source_type == "pdf"
    assert result[0].access_roles == ["admin"]


def test_search_passes_user_role_and_min_similarity_to_rpc():
    search("query", mock_client, user_role="hr", min_similarity=0.6)
    call_kwargs = mock_client.rpc.call_args[0][1]
    assert call_kwargs["user_role"] == "hr"
    assert call_kwargs["min_similarity"] == 0.6


def test_search_filters_by_source_types():
    # Mock returns two results: one pdf, one csv
    # When source_types=["pdf"], only the pdf result should be returned
    ...  # adapt to the existing mock pattern in the file
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/retrieval/test_vector_store.py -v 2>&1 | tail -20
```

Expected: failures on missing attributes / wrong RPC args.

- [ ] **Step 3: Update SearchResult dataclass**

```python
@dataclass
class SearchResult:
    chunk_id:     str
    source_id:    str
    source_type:  str
    filename:     str
    page_number:  int
    text:         str
    token_count:  int
    language:     str
    bbox:         list[float] | None
    access_roles: list[str]
    similarity:   float
```

- [ ] **Step 4: Update search() signature and RPC call**

```python
def search(
    query: str,
    client: Client,
    k: int = TOP_K,
    model: SentenceTransformer | None = None,
    user_role: str | None = None,
    min_similarity: float = 0.4,
    source_types: list[str] | None = None,
) -> list[SearchResult]:
    """Embed query and return top-k chunks by cosine similarity, filtered by role and source type."""
    if model is None:
        model = _get_model()

    vector = model.encode(query, normalize_embeddings=True)
    fetch_k = k * 3 if source_types else k  # over-fetch to compensate for type filtering

    resp = client.rpc(
        RPC_FUNCTION,
        {
            "query_embedding": vector.tolist(),
            "match_count": fetch_k,
            "user_role": user_role,
            "min_similarity": min_similarity,
        },
    ).execute()

    results = [
        SearchResult(
            chunk_id=row["chunk_id"],
            source_id=row["source_id"],
            source_type=row["source_type"],
            filename=row["filename"],
            page_number=row["page_number"],
            text=row["text"],
            token_count=row["token_count"],
            language=row["language"],
            bbox=row["bbox"],
            access_roles=row["access_roles"],
            similarity=row["similarity"],
        )
        for row in resp.data
    ]

    if source_types:
        results = [r for r in results if r.source_type in source_types]

    log.debug("search returned %d results (k=%d, role=%s, types=%s)", len(results[:k]), k, user_role, source_types)
    return results[:k]
```

- [ ] **Step 5: Run all vector_store tests**

```bash
python -m pytest tests/retrieval/test_vector_store.py -v 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backend/retrieval/vector_store.py tests/retrieval/test_vector_store.py
git commit -m "feat: SearchResult adds source_id/source_type/access_roles; search() supports user_role, min_similarity, source_types"
```

---

## Task 5: Install Auth Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add packages to requirements.txt**

Open `requirements.txt` and add these two lines in the appropriate section (after existing backend deps):

```
PyJWT==2.9.0
passlib[bcrypt]==1.7.4
```

- [ ] **Step 2: Install**

```bash
pip install PyJWT==2.9.0 "passlib[bcrypt]==1.7.4"
```

Expected output ends with `Successfully installed ...` — no errors.

- [ ] **Step 3: Verify import**

```bash
python -c "import jwt; import passlib; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add PyJWT and passlib[bcrypt] for JWT auth"
```

---

## Task 6: JWT Auth Module

**Files:**
- Create: `backend/auth/__init__.py`
- Create: `backend/auth/models.py`
- Create: `backend/auth/jwt_utils.py`
- Create: `backend/auth/dependencies.py`
- Create: `backend/auth/router.py`
- Create: `tests/auth/__init__.py`
- Create: `tests/auth/test_jwt_utils.py`
- Create: `tests/auth/test_auth_router.py`

- [ ] **Step 1: Write JWT utils tests**

Create `tests/auth/__init__.py` (empty) and `tests/auth/test_jwt_utils.py`:

```python
"""Tests for JWT token creation and decoding."""
import pytest
import jwt as pyjwt
import backend.auth.jwt_utils as jwt_utils


def test_create_and_decode_roundtrip():
    token = jwt_utils.create_token("alice", "admin")
    payload = jwt_utils.decode_token(token)
    assert payload["sub"] == "alice"
    assert payload["role"] == "admin"


def test_expired_token_raises():
    original = jwt_utils.TOKEN_EXPIRE_MINUTES
    jwt_utils.TOKEN_EXPIRE_MINUTES = -1
    token = jwt_utils.create_token("bob", "hr")
    jwt_utils.TOKEN_EXPIRE_MINUTES = original
    with pytest.raises(pyjwt.ExpiredSignatureError):
        jwt_utils.decode_token(token)


def test_tampered_token_raises():
    token = jwt_utils.create_token("alice", "admin")
    bad = token[:-4] + "xxxx"
    with pytest.raises(pyjwt.InvalidTokenError):
        jwt_utils.decode_token(bad)


def test_token_is_string():
    token = jwt_utils.create_token("carol", "finance")
    assert isinstance(token, str)
    assert len(token) > 20
```

- [ ] **Step 2: Write auth router tests**

Create `tests/auth/test_auth_router.py`:

```python
"""Tests for POST /auth/login."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.auth.models import User


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


def _mock_user():
    return User(username="alice", hashed_password="$hashed$", role="admin")


def test_login_success_returns_token(client):
    with patch("backend.auth.router.get_user", return_value=_mock_user()), \
         patch("backend.auth.router._pwd_ctx.verify", return_value=True):
        resp = client.post("/auth/login", data={"username": "alice", "password": "admin123"})
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_login_wrong_password_returns_401(client):
    with patch("backend.auth.router.get_user", return_value=_mock_user()), \
         patch("backend.auth.router._pwd_ctx.verify", return_value=False):
        resp = client.post("/auth/login", data={"username": "alice", "password": "wrong"})
    assert resp.status_code == 401


def test_login_unknown_user_returns_401(client):
    with patch("backend.auth.router.get_user", return_value=None):
        resp = client.post("/auth/login", data={"username": "nobody", "password": "x"})
    assert resp.status_code == 401
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
python -m pytest tests/auth/ -v 2>&1 | tail -15
```

Expected: `ImportError` or `ModuleNotFoundError` for `backend.auth`.

- [ ] **Step 4: Create backend/auth/__init__.py**

```bash
touch backend/auth/__init__.py
```

- [ ] **Step 5: Create backend/auth/models.py**

```python
"""User model and in-file user store (loaded from data/users.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROLES = {"admin", "hr", "finance", "engineering"}

_USERS_FILE = Path(__file__).parent.parent.parent / "data" / "users.json"


@dataclass
class User:
    username: str
    hashed_password: str
    role: str


def load_users() -> dict[str, User]:
    """Load users from data/users.json. Returns empty dict if file absent (pre-seed)."""
    if not _USERS_FILE.exists():
        return {}
    data = json.loads(_USERS_FILE.read_text(encoding="utf-8"))
    return {u["username"]: User(**u) for u in data}


def get_user(username: str) -> User | None:
    return load_users().get(username)
```

- [ ] **Step 6: Create backend/auth/jwt_utils.py**

```python
"""JWT token creation and decoding."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import jwt

SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-in-prod")
ALGORITHM: str = "HS256"
TOKEN_EXPIRE_MINUTES: int = 480


def create_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
```

- [ ] **Step 7: Create backend/auth/dependencies.py**

```python
"""FastAPI dependency for authenticated requests."""
from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from backend.auth.jwt_utils import decode_token
from backend.auth.models import User, get_user

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def get_current_user(token: str | None = Depends(oauth2_scheme)) -> User:
    if token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_token(token)
        username: str = payload["sub"]
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user
```

- [ ] **Step 8: Create backend/auth/router.py**

```python
"""Authentication endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext

from backend.auth.jwt_utils import create_token
from backend.auth.models import get_user

router = APIRouter()
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form.username)
    if not user or not _pwd_ctx.verify(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user.username, user.role)
    return {"access_token": token, "token_type": "bearer"}
```

- [ ] **Step 9: Run auth tests**

```bash
python -m pytest tests/auth/ -v 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add backend/auth/ tests/auth/
git commit -m "feat: JWT auth module — User model, create/decode token, login endpoint, get_current_user dependency"
```

---

## Task 7: Wire Auth into main.py and Update All Endpoints

**Files:**
- Modify: `backend/main.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Update test_main.py to provide auth**

In `tests/test_main.py`, every test that calls `/query`, `/ingest`, or `/chunks` now requires a valid JWT. Add a fixture that patches `get_current_user` to bypass auth in unit tests:

```python
import pytest
from unittest.mock import patch
from backend.auth.models import User

@pytest.fixture(autouse=True)
def mock_auth():
    """Bypass JWT auth in all main.py tests."""
    user = User(username="alice", hashed_password="x", role="admin")
    with patch("backend.main.get_current_user", return_value=user):
        yield
```

Also update any test that calls `/query` and checks the SSE stream: it should now also expect an `event: sources` line before the text data. For example:

```python
def test_query_emits_sources_event(client, mock_search, mock_generate):
    # mock_search already returns a list[SearchResult] — keep existing mock
    resp = client.post("/query", json={"query": "test", "rerank": False})
    raw = resp.text
    assert "event: sources" in raw
    assert "data: [" in raw  # sources JSON array
```

- [ ] **Step 2: Run tests to confirm failures**

```bash
python -m pytest tests/test_main.py -v 2>&1 | tail -20
```

Expected: failures on 401 Unauthorized and missing `event: sources`.

- [ ] **Step 3: Update main.py**

Replace the entire file content with the updated version below. Key changes:
- Include auth router
- `IngestResponse.pdf_id` → `source_id`
- Add `Form` and `uuid` imports
- Add `roles` form field to `/ingest`
- Add `user_role` from JWT to `/query` and `/chunks`
- `_sse_generator` emits `event: sources` before text
- Add `/ingest/csv` and `/ingest/json` endpoints

```python
"""FastAPI application entry point."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase import Client, create_client

from backend.auth.dependencies import get_current_user
from backend.auth.models import User
from backend.auth.router import router as auth_router
from backend.generation.llm_client import generate
from backend.ingestion.chunker import chunk_document
from backend.ingestion.embed import embed_chunks
from backend.ingestion.extract import ExtractionError, extract_pdf
from backend.ingestion.ingest_csv import ingest_csv
from backend.ingestion.ingest_json import ingest_json
from backend.retrieval.reranker import rerank
from backend.retrieval.router import route_query
from backend.retrieval.vector_store import search

log = logging.getLogger(__name__)

app = FastAPI(title="DocuMitra")
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def _get_supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def get_supabase() -> Client:
    return _get_supabase()


class QueryRequest(BaseModel):
    query: str
    rerank: bool = True


def _sse_generator(
    query: str,
    client: Client,
    rerank_enabled: bool,
    user_role: str,
) -> Iterator[str]:
    source_types = route_query(query)
    results = search(query, client, user_role=user_role, source_types=source_types)
    if rerank_enabled:
        results = rerank(query, results)

    sources_payload = [
        {
            "filename": r.filename,
            "page": r.page_number,
            "similarity": round(r.similarity, 3),
            "source_type": r.source_type,
        }
        for r in results
    ]
    yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"

    for chunk in generate(query, results):
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/query")
def query_endpoint(
    req: QueryRequest,
    client: Client = Depends(get_supabase),
    current_user: User = Depends(get_current_user),
):
    return StreamingResponse(
        _sse_generator(req.query, client, req.rerank, current_user.role),
        media_type="text/event-stream",
    )


class IngestResponse(BaseModel):
    source_id: str
    filename: str
    chunks_added: int


def _parse_roles(roles: str) -> list[str]:
    return [r.strip() for r in roles.split(",") if r.strip()]


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    file: UploadFile,
    roles: str = Form(default=""),
    client: Client = Depends(get_supabase),
    current_user: User = Depends(get_current_user),
) -> IngestResponse:
    access_roles = _parse_roles(roles)
    content = await file.read()
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        doc = extract_pdf(tmp_path, filename=file.filename or tmp_path.name)
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)
    chunks = chunk_document(doc, access_roles=access_roles)
    added = embed_chunks(chunks, client)
    return IngestResponse(source_id=doc.pdf_id, filename=doc.filename, chunks_added=added)


@app.post("/ingest/csv", response_model=IngestResponse)
async def ingest_csv_endpoint(
    file: UploadFile,
    roles: str = Form(default=""),
    client: Client = Depends(get_supabase),
    current_user: User = Depends(get_current_user),
) -> IngestResponse:
    access_roles = _parse_roles(roles)
    content = await file.read()
    filename = file.filename or "upload.csv"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        source_id = uuid.uuid4().hex[:16]
        chunks = ingest_csv(tmp_path, source_id=source_id, filename=filename, access_roles=access_roles)
    finally:
        tmp_path.unlink(missing_ok=True)
    added = embed_chunks(chunks, client)
    return IngestResponse(source_id=source_id, filename=filename, chunks_added=added)


@app.post("/ingest/json", response_model=IngestResponse)
async def ingest_json_endpoint(
    file: UploadFile,
    roles: str = Form(default=""),
    client: Client = Depends(get_supabase),
    current_user: User = Depends(get_current_user),
) -> IngestResponse:
    access_roles = _parse_roles(roles)
    content = await file.read()
    filename = file.filename or "upload.json"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        source_id = uuid.uuid4().hex[:16]
        chunks = ingest_json(tmp_path, source_id=source_id, filename=filename, access_roles=access_roles)
    finally:
        tmp_path.unlink(missing_ok=True)
    added = embed_chunks(chunks, client)
    return IngestResponse(source_id=source_id, filename=filename, chunks_added=added)


@app.get("/chunks")
def chunks_endpoint(
    query: str,
    k: int = 5,
    client: Client = Depends(get_supabase),
    current_user: User = Depends(get_current_user),
) -> dict:
    results = search(query, client, k=k, user_role=current_user.role)
    return {"results": [dataclasses.asdict(r) for r in results]}
```

- [ ] **Step 4: Run all main tests**

```bash
python -m pytest tests/test_main.py -v 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py tests/test_main.py
git commit -m "feat: wire JWT auth into all endpoints; add /ingest/csv, /ingest/json; SSE emits sources event; query routes by source type"
```

---

## Task 8: CSV Ingestion Module

**Files:**
- Create: `backend/ingestion/ingest_csv.py`
- Create: `tests/ingestion/test_ingest_csv.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ingestion/test_ingest_csv.py`:

```python
"""Tests for CSV → Chunk ingestion."""
import csv
from pathlib import Path

import pytest

from backend.ingestion.ingest_csv import ingest_csv


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "employees.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "department", "salary"])
        writer.writeheader()
        for i in range(10):
            writer.writerow({"id": i, "name": f"Person {i}", "department": "Engineering", "salary": 80000 + i * 1000})
    return path


def test_ingest_csv_returns_at_least_one_chunk(csv_file):
    chunks = ingest_csv(csv_file, source_id="csv-001", filename="employees.csv", access_roles=["hr"])
    assert len(chunks) >= 1


def test_ingest_csv_source_fields(csv_file):
    chunks = ingest_csv(csv_file, source_id="csv-001", filename="employees.csv", access_roles=["hr", "admin"])
    for c in chunks:
        assert c.source_id == "csv-001"
        assert c.source_type == "csv"
        assert c.filename == "employees.csv"
        assert "hr" in c.access_roles
        assert "admin" in c.access_roles


def test_ingest_csv_chunk_text_contains_column_names(csv_file):
    chunks = ingest_csv(csv_file, source_id="s1", filename="test.csv", access_roles=[])
    combined = " ".join(c.text for c in chunks).lower()
    assert "name" in combined
    assert "department" in combined
    assert "salary" in combined


def test_ingest_csv_unique_chunk_ids(csv_file):
    chunks = ingest_csv(csv_file, source_id="s2", filename="test.csv", access_roles=[])
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_ingest_csv_empty_file_returns_no_chunks(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("id,name\n", encoding="utf-8")
    chunks = ingest_csv(path, source_id="s3", filename="empty.csv", access_roles=[])
    assert chunks == []


def test_ingest_csv_positive_token_counts(csv_file):
    chunks = ingest_csv(csv_file, source_id="s4", filename="test.csv", access_roles=[])
    assert all(c.token_count > 0 for c in chunks)
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/ingestion/test_ingest_csv.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'backend.ingestion.ingest_csv'`.

- [ ] **Step 3: Implement ingest_csv.py**

Create `backend/ingestion/ingest_csv.py`:

```python
"""Ingest CSV files into the chunk pipeline."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from backend.ingestion.chunker import (
    CHUNK_MAX_TOKENS,
    Chunk,
    _count_tokens,
    _detect_language,
    _get_tokenizer,
)


def _row_to_text(row: dict[str, str]) -> str:
    return " | ".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())


def ingest_csv(
    path: Path,
    source_id: str,
    filename: str,
    access_roles: list[str],
) -> list[Chunk]:
    """Read a CSV file and return a list of token-bounded Chunks."""
    tokenizer = _get_tokenizer()

    with open(path, newline="", encoding="utf-8") as f:
        row_texts = [_row_to_text(r) for r in csv.DictReader(f) if any(r.values())]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    chunk_index = 0
    record_start = 0

    for i, row_text in enumerate(row_texts):
        row_tokens = _count_tokens(row_text, tokenizer)
        if buffer_tokens + row_tokens > CHUNK_MAX_TOKENS and buffer:
            text = "\n".join(buffer)
            chunk_id = hashlib.sha256(f"{source_id}:{chunk_index}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source_id=source_id,
                source_type="csv",
                filename=filename,
                page_number=record_start + 1,
                text=text,
                token_count=buffer_tokens,
                language=_detect_language(text),
                bbox=None,
                access_roles=list(access_roles),
            ))
            buffer = []
            buffer_tokens = 0
            chunk_index += 1
            record_start = i

        buffer.append(row_text)
        buffer_tokens += row_tokens

    if buffer:
        text = "\n".join(buffer)
        chunk_id = hashlib.sha256(f"{source_id}:{chunk_index}".encode()).hexdigest()[:16]
        chunks.append(Chunk(
            chunk_id=chunk_id,
            source_id=source_id,
            source_type="csv",
            filename=filename,
            page_number=record_start + 1,
            text=text,
            token_count=buffer_tokens,
            language=_detect_language(text),
            bbox=None,
            access_roles=list(access_roles),
        ))

    return chunks
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ingestion/test_ingest_csv.py -v 2>&1 | tail -10
```

Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add backend/ingestion/ingest_csv.py tests/ingestion/test_ingest_csv.py
git commit -m "feat: CSV ingestion — ingest_csv() converts rows to token-bounded Chunks with RBAC roles"
```

---

## Task 9: JSON Ingestion Module

**Files:**
- Create: `backend/ingestion/ingest_json.py`
- Create: `tests/ingestion/test_ingest_json.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ingestion/test_ingest_json.py`:

```python
"""Tests for JSON log → Chunk ingestion."""
import json
from pathlib import Path

import pytest

from backend.ingestion.ingest_json import ingest_json


@pytest.fixture
def json_file(tmp_path: Path) -> Path:
    path = tmp_path / "audit.json"
    records = [
        {"timestamp": f"2024-01-15T10:{i:02d}:00Z", "user": f"user{i}", "action": "login", "status": "success"}
        for i in range(20)
    ]
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_ingest_json_returns_at_least_one_chunk(json_file):
    chunks = ingest_json(json_file, source_id="json-001", filename="audit.json", access_roles=["admin"])
    assert len(chunks) >= 1


def test_ingest_json_source_fields(json_file):
    chunks = ingest_json(json_file, source_id="json-001", filename="audit.json", access_roles=["admin"])
    for c in chunks:
        assert c.source_id == "json-001"
        assert c.source_type == "json"
        assert c.filename == "audit.json"
        assert "admin" in c.access_roles


def test_ingest_json_chunk_text_contains_field_names(json_file):
    chunks = ingest_json(json_file, source_id="s1", filename="audit.json", access_roles=[])
    combined = " ".join(c.text for c in chunks).lower()
    assert "timestamp" in combined
    assert "action" in combined


def test_ingest_json_unique_chunk_ids(json_file):
    chunks = ingest_json(json_file, source_id="s2", filename="audit.json", access_roles=[])
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_ingest_json_single_dict(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"version": "1.0", "env": "prod"}), encoding="utf-8")
    chunks = ingest_json(path, source_id="s3", filename="config.json", access_roles=[])
    assert len(chunks) >= 1


def test_ingest_json_empty_list(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    chunks = ingest_json(path, source_id="s4", filename="empty.json", access_roles=[])
    assert chunks == []
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/ingestion/test_ingest_json.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement ingest_json.py**

Create `backend/ingestion/ingest_json.py`:

```python
"""Ingest JSON log files into the chunk pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.ingestion.chunker import (
    CHUNK_MAX_TOKENS,
    Chunk,
    _count_tokens,
    _detect_language,
    _get_tokenizer,
)


def _record_to_text(record: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in record.items())


def ingest_json(
    path: Path,
    source_id: str,
    filename: str,
    access_roles: list[str],
) -> list[Chunk]:
    """Read a JSON file (list of records or single dict) and return token-bounded Chunks."""
    tokenizer = _get_tokenizer()

    raw = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict] = raw if isinstance(raw, list) else [raw]
    if not records:
        return []

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    chunk_index = 0
    record_start = 0

    for i, record in enumerate(records):
        rec_text = _record_to_text(record)
        rec_tokens = _count_tokens(rec_text, tokenizer)
        if buffer_tokens + rec_tokens > CHUNK_MAX_TOKENS and buffer:
            text = "\n---\n".join(buffer)
            chunk_id = hashlib.sha256(f"{source_id}:{chunk_index}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source_id=source_id,
                source_type="json",
                filename=filename,
                page_number=record_start + 1,
                text=text,
                token_count=buffer_tokens,
                language=_detect_language(text),
                bbox=None,
                access_roles=list(access_roles),
            ))
            buffer = []
            buffer_tokens = 0
            chunk_index += 1
            record_start = i

        buffer.append(rec_text)
        buffer_tokens += rec_tokens

    if buffer:
        text = "\n---\n".join(buffer)
        chunk_id = hashlib.sha256(f"{source_id}:{chunk_index}".encode()).hexdigest()[:16]
        chunks.append(Chunk(
            chunk_id=chunk_id,
            source_id=source_id,
            source_type="json",
            filename=filename,
            page_number=record_start + 1,
            text=text,
            token_count=buffer_tokens,
            language=_detect_language(text),
            bbox=None,
            access_roles=list(access_roles),
        ))

    return chunks
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ingestion/test_ingest_json.py -v 2>&1 | tail -10
```

Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add backend/ingestion/ingest_json.py tests/ingestion/test_ingest_json.py
git commit -m "feat: JSON log ingestion — ingest_json() converts records to token-bounded Chunks with RBAC roles"
```

---

## Task 10: Query-Aware Router

**Files:**
- Create: `backend/retrieval/router.py`
- Create: `tests/retrieval/test_router.py`

- [ ] **Step 1: Write failing tests**

Create `tests/retrieval/test_router.py`:

```python
"""Tests for keyword-based query routing."""
from backend.retrieval.router import route_query


def test_audit_log_query_routes_to_json():
    result = route_query("show me audit logs for failed login events")
    assert "json" in result


def test_employee_salary_query_routes_to_csv():
    result = route_query("what is the salary of employees in engineering department")
    assert "csv" in result


def test_policy_query_routes_to_pdf():
    result = route_query("what does the compliance policy say about data retention")
    assert "pdf" in result


def test_ambiguous_query_returns_all_three_types():
    result = route_query("give me a summary of everything")
    assert set(result) == {"pdf", "csv", "json"}


def test_result_has_no_duplicates():
    result = route_query("employee compliance policy audit log")
    assert len(result) == len(set(result))


def test_result_only_contains_valid_source_types():
    result = route_query("any query at all")
    assert all(t in {"pdf", "csv", "json"} for t in result)


def test_empty_query_returns_all_types():
    result = route_query("")
    assert set(result) == {"pdf", "csv", "json"}
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/retrieval/test_router.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement router.py**

Create `backend/retrieval/router.py`:

```python
"""Keyword-based query routing to data source types."""
from __future__ import annotations

_SOURCE_HINTS: dict[str, list[str]] = {
    "csv": [
        "employee", "salary", "department", "budget", "headcount",
        "hire", "payroll", "staff", "record", "workforce",
    ],
    "json": [
        "log", "audit", "alert", "event", "error", "trace",
        "timestamp", "incident", "access", "login", "logout",
    ],
    "pdf": [
        "policy", "compliance", "regulation", "guideline", "chapter",
        "section", "document", "handbook", "procedure", "clause",
    ],
}

ALL_TYPES = ["pdf", "csv", "json"]


def route_query(query: str) -> list[str]:
    """Return source types to search based on query keywords.

    Returns all types when the query is ambiguous or empty.
    Never returns duplicate types.
    """
    q = query.lower()
    scores = {src: sum(1 for kw in hints if kw in q) for src, hints in _SOURCE_HINTS.items()}
    best = max(scores.values())
    if best == 0:
        return list(ALL_TYPES)
    return [src for src, score in scores.items() if score == best]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/retrieval/test_router.py -v 2>&1 | tail -10
```

Expected: all 7 pass.

- [ ] **Step 5: Commit**

```bash
git add backend/retrieval/router.py tests/retrieval/test_router.py
git commit -m "feat: keyword-based query router — routes to pdf/csv/json source types, falls back to all types when ambiguous"
```

---

## Task 11: Seed Script and Synthetic Dataset

**Files:**
- Create: `scripts/seed_users.py`
- Create: `scripts/generate_synthetic_data.py`
- Create: `data/access_policies.json`

- [ ] **Step 1: Create seed_users.py**

```python
#!/usr/bin/env python3
"""Generate data/users.json with bcrypt-hashed passwords."""
import json
from pathlib import Path
from passlib.context import CryptContext

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

USERS = [
    {"username": "alice", "password": "admin123",   "role": "admin"},
    {"username": "bob",   "password": "hr123",      "role": "hr"},
    {"username": "carol", "password": "finance123", "role": "finance"},
    {"username": "dave",  "password": "eng123",     "role": "engineering"},
]

out = [
    {
        "username": u["username"],
        "hashed_password": _pwd_ctx.hash(u["password"]),
        "role": u["role"],
    }
    for u in USERS
]

out_path = Path(__file__).parent.parent / "data" / "users.json"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Written {len(out)} users to {out_path}")
for u in USERS:
    print(f"  {u['username']} / {u['password']}  →  role: {u['role']}")
```

- [ ] **Step 2: Run seed script**

```bash
python scripts/seed_users.py
```

Expected output:
```
Written 4 users to .../data/users.json
  alice / admin123  →  role: admin
  bob / hr123  →  role: hr
  carol / finance123  →  role: finance
  dave / eng123  →  role: engineering
```

- [ ] **Step 3: Create generate_synthetic_data.py**

```python
#!/usr/bin/env python3
"""Generate synthetic enterprise CSV and JSON datasets."""
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

out_dir = Path(__file__).parent.parent / "data"
out_dir.mkdir(exist_ok=True)

# --- employees.csv (hr + admin access) ---
departments = ["Engineering", "HR", "Finance", "Marketing", "Legal"]
employees = [
    {
        "id": i + 1,
        "name": f"Employee {i+1:03d}",
        "department": departments[i % len(departments)],
        "salary": random.randint(60000, 130000),
        "manager": f"Manager {(i // 5) + 1:02d}",
        "hire_date": (datetime(2018, 1, 1) + timedelta(days=random.randint(0, 2000))).strftime("%Y-%m-%d"),
        "status": "active" if i % 10 != 0 else "inactive",
    }
    for i in range(50)
]

csv_path = out_dir / "employees.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=employees[0].keys())
    writer.writeheader()
    writer.writerows(employees)
print(f"Written {len(employees)} rows to {csv_path}")

# --- audit_logs.json (admin only) ---
actions = ["login", "logout", "view_report", "export_data", "modify_record", "delete_record", "access_denied"]
logs = [
    {
        "id": i + 1,
        "timestamp": (datetime(2024, 1, 1) + timedelta(hours=i * 3)).isoformat() + "Z",
        "user": f"user{(i % 10) + 1:02d}",
        "action": actions[i % len(actions)],
        "resource": f"document_{random.randint(1, 20):03d}",
        "ip": f"192.168.1.{random.randint(1, 254)}",
        "status": "success" if i % 7 != 0 else "denied",
        "duration_ms": random.randint(10, 500),
    }
    for i in range(80)
]

json_path = out_dir / "audit_logs.json"
json_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
print(f"Written {len(logs)} records to {json_path}")
```

- [ ] **Step 4: Run data generation script**

```bash
python scripts/generate_synthetic_data.py
```

Expected:
```
Written 50 rows to .../data/employees.csv
Written 80 records to .../data/audit_logs.json
```

- [ ] **Step 5: Create access_policies.json**

Create `data/access_policies.json`:

```json
{
  "employees.csv": ["admin", "hr"],
  "audit_logs.json": ["admin"],
  "policy_handbook.pdf": ["admin", "hr", "engineering", "finance"],
  "financial_report.pdf": ["admin", "finance"],
  "engineering_specs.pdf": ["admin", "engineering"]
}
```

- [ ] **Step 6: Commit**

```bash
git add scripts/seed_users.py scripts/generate_synthetic_data.py data/access_policies.json
git add data/users.json data/employees.csv data/audit_logs.json
git commit -m "feat: synthetic enterprise dataset — 4 roles, 50 employees CSV, 80 audit log records JSON, access policies"
```

---

## Task 12: Frontend — Auth Library and Login Page

**Files:**
- Create: `frontend/lib/auth.ts`
- Create: `frontend/app/login/page.tsx`

- [ ] **Step 1: Create frontend/lib/auth.ts**

```typescript
const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
const TOKEN_KEY = 'documitra_token'

export function getToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

export async function login(username: string, password: string): Promise<void> {
  const body = new URLSearchParams({ username, password, grant_type: 'password' })
  const res = await fetch(`${BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  })
  if (!res.ok) throw new Error('Invalid credentials')
  const { access_token } = (await res.json()) as { access_token: string }
  setToken(access_token)
}

export function logout(): void {
  clearToken()
}
```

- [ ] **Step 2: Create frontend/app/login/page.tsx**

```tsx
'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { login } from '@/lib/auth'

export default function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(username, password)
      router.push('/chat')
    } catch {
      setError('Invalid username or password')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <form
        onSubmit={handleSubmit}
        className="bg-white p-8 rounded-xl shadow-sm border border-gray-200 w-80 flex flex-col gap-4"
      >
        <h1 className="text-xl font-semibold text-gray-900">DocuMitra</h1>
        <p className="text-sm text-gray-500">Sign in to access your enterprise documents</p>
        {error && <p className="text-sm text-red-500 bg-red-50 rounded px-3 py-2">{error}</p>}
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={e => setUsername(e.target.value)}
          className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          required
          autoComplete="username"
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          required
          autoComplete="current-password"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 text-white rounded-lg px-4 py-2 text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Signing in…' : 'Sign in'}
        </button>
        <p className="text-xs text-gray-400 text-center">
          Test accounts: alice/admin123 · bob/hr123 · carol/finance123 · dave/eng123
        </p>
      </form>
    </div>
  )
}
```

- [ ] **Step 3: Verify pages compile**

```bash
cd /Users/sadiya/projects/DocuMitra/frontend && npx tsc --noEmit 2>&1 | head -20
```

Expected: no errors related to the new files.

- [ ] **Step 4: Commit**

```bash
git add frontend/lib/auth.ts frontend/app/login/page.tsx
git commit -m "feat: login page and auth token helpers (getToken, login, logout)"
```

---

## Task 13: Frontend — SSE Update and Sources Panel

**Files:**
- Modify: `frontend/lib/api.ts`
- Modify: `frontend/hooks/useStreamingChat.ts`
- Modify: `frontend/components/chat/MessageList.tsx`

- [ ] **Step 1: Update frontend/lib/api.ts**

Replace the `SearchResult` interface and `streamQuery` function. Keep `ingestFile` and `getChunks` as-is, but add the Bearer token param to both as well.

```typescript
const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

export interface SearchResult {
  chunk_id: string
  source_id: string       // renamed from pdf_id
  source_type: string     // "pdf" | "csv" | "json"
  filename: string
  page_number: number
  text: string
  token_count: number
  language: string
  bbox: number[] | null
  access_roles: string[]
  similarity: number
}

export interface SourceResult {
  filename: string
  page: number
  similarity: number
  source_type: string
}

export interface IngestResponse {
  source_id: string       // renamed from pdf_id
  filename: string
  chunks_added: number
}

export interface ChunksResponse {
  results: SearchResult[]
}

export type StreamEvent =
  | { type: 'text'; content: string }
  | { type: 'sources'; sources: SourceResult[] }

async function assertOk(res: Response): Promise<void> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body.detail) detail = String(body.detail)
    } catch {}
    throw new ApiError(res.status, detail)
  }
}

function authHeader(token?: string | null): HeadersInit {
  const h: HeadersInit = { 'Content-Type': 'application/json' }
  if (token) h['Authorization'] = `Bearer ${token}`
  return h
}

export async function* streamQuery(
  query: string,
  rerank = true,
  token?: string | null,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: authHeader(token),
    body: JSON.stringify({ query, rerank }),
  })
  await assertOk(res)

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() ?? ''
    for (const part of parts) {
      const lines = part.split('\n')
      let eventType = 'message'
      let data = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) eventType = line.slice(7).trim()
        else if (line.startsWith('data: ')) data = line.slice(6)
      }
      if (eventType === 'sources') {
        try { yield { type: 'sources', sources: JSON.parse(data) as SourceResult[] } } catch {}
      } else if (data === '[DONE]') {
        return
      } else if (data) {
        try { yield { type: 'text', content: JSON.parse(data) as string } } catch {}
      }
    }
  }
}

export async function ingestFile(file: File, roles: string[] = [], token?: string | null): Promise<IngestResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('roles', roles.join(','))
  const headers: HeadersInit = {}
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${BASE}/ingest`, { method: 'POST', headers, body: form })
  await assertOk(res)
  return res.json()
}

export async function getChunks(query: string, k: number, token?: string | null): Promise<ChunksResponse> {
  const params = new URLSearchParams({ query, k: String(k) })
  const headers: HeadersInit = {}
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${BASE}/chunks?${params}`, { headers })
  await assertOk(res)
  return res.json()
}
```

- [ ] **Step 2: Update useStreamingChat.ts**

```typescript
'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { streamQuery, type SourceResult } from '@/lib/api'
import { getToken } from '@/lib/auth'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceResult[]
  timestamp: number
}

const STORAGE_KEY = 'documitra_chat_history'

export function useStreamingChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const skipNextSave = useRef(false)
  const isStreamingRef = useRef(false)

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) setMessages(JSON.parse(stored))
    } catch {}
  }, [])

  useEffect(() => {
    if (skipNextSave.current) {
      skipNextSave.current = false
      return
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages))
  }, [messages])

  const sendMessage = useCallback(async (query: string) => {
    if (isStreamingRef.current) return

    isStreamingRef.current = true
    setIsStreaming(true)

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      timestamp: Date.now(),
    }
    const assistantId = crypto.randomUUID()
    const assistantMsg: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    }

    setMessages(prev => [...prev, userMsg, assistantMsg])

    try {
      for await (const event of streamQuery(query, true, getToken())) {
        if (event.type === 'sources') {
          setMessages(prev =>
            prev.map(m => m.id === assistantId ? { ...m, sources: event.sources } : m),
          )
        } else if (event.type === 'text') {
          setMessages(prev =>
            prev.map(m => m.id === assistantId ? { ...m, content: m.content + event.content } : m),
          )
        }
      }
    } finally {
      isStreamingRef.current = false
      setIsStreaming(false)
    }
  }, [])

  const clearHistory = useCallback(() => {
    skipNextSave.current = true
    setMessages([])
    localStorage.removeItem(STORAGE_KEY)
  }, [])

  return { messages, isStreaming, sendMessage, clearHistory }
}
```

- [ ] **Step 3: Add sources panel to MessageList.tsx**

Read the current `MessageList.tsx` file first, then find the assistant message rendering block and add a sources panel beneath it:

```tsx
{msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
  <div className="mt-2 flex flex-wrap gap-1.5">
    {msg.sources.map((s, i) => (
      <span
        key={i}
        title={`${s.filename} · p.${s.page} · ${s.source_type} · ${(s.similarity * 100).toFixed(0)}% match`}
        className="inline-flex items-center gap-1 text-xs bg-gray-100 text-gray-600 rounded-full px-2 py-0.5 border border-gray-200"
      >
        <span className="font-medium">{s.filename}</span>
        <span className="text-gray-400">p.{s.page}</span>
        <span className="text-blue-500">{(s.similarity * 100).toFixed(0)}%</span>
      </span>
    ))}
  </div>
)}
```

- [ ] **Step 4: Type-check the frontend**

```bash
cd /Users/sadiya/projects/DocuMitra/frontend && npx tsc --noEmit 2>&1 | head -30
```

Expected: no errors.

- [ ] **Step 5: Run frontend tests**

```bash
cd /Users/sadiya/projects/DocuMitra/frontend && npm test -- --passWithNoTests 2>&1 | tail -15
```

Expected: no failures (update any tests that used `pdf_id` → `source_id` or the old `streamQuery` signature).

- [ ] **Step 6: Commit**

```bash
git add frontend/lib/api.ts frontend/hooks/useStreamingChat.ts frontend/components/chat/MessageList.tsx
git commit -m "feat: SSE sources panel — parse named sources event, show filename/page/confidence chips per answer"
```

---

## Task 14: Full Backend Test Suite Verification

**Files:** no changes — verification only

- [ ] **Step 1: Run the full backend test suite**

```bash
cd /Users/sadiya/projects/DocuMitra && python -m pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests pass. If any fail, fix before proceeding.

- [ ] **Step 2: Run a quick smoke test against a live server (optional, requires env vars)**

```bash
# Terminal 1 — start server
uvicorn backend.main:app --reload

# Terminal 2 — test login and query
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -d "username=alice&password=admin123" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "show audit logs"}' | head -5
```

Expected: SSE response starting with `event: sources\ndata: [...]`.

- [ ] **Step 3: Commit final verification note**

```bash
git commit --allow-empty -m "chore: full test suite green after enterprise RAG additions"
```

---

## Self-Review Checklist

- [x] **Multi-format ingestion** — CSV (`ingest_csv.py`) and JSON (`ingest_json.py`), both with endpoints in `main.py`
- [x] **RBAC enforcement** — `access_roles[]` on chunks, JWT auth on all endpoints, RPC filters by role
- [x] **Similarity threshold** — `min_similarity=0.4` default in `search()`; passed to RPC
- [x] **Query-aware routing** — `router.py` keyword-based; wired into `_sse_generator`
- [x] **User auth layer** — JWT login endpoint, `get_current_user` dependency, `data/users.json`
- [x] **`roles` param on `/ingest`** — Form field, passed to `chunk_document`
- [x] **Source-agnostic schema** — `Chunk.source_id`/`source_type`, `SearchResult` updated, `embed.py` rows updated
- [x] **Structured sources SSE event** — `event: sources\ndata: [...]` emitted before text stream
- [x] **Confidence indicators in frontend** — similarity %, filename, page in chip badges
- [x] **Synthetic dataset** — `employees.csv`, `audit_logs.json`, `users.json`, `access_policies.json`
- [x] **Existing tests updated** — `test_chunker.py`, `test_embed.py`, `test_vector_store.py`, `test_main.py` all need `pdf_id` → `source_id` updates
