"""Encodes text chunks into vectors using BAAI/bge-small-en-v1.5."""
from __future__ import annotations

import logging
import threading

import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import Client

from backend.ingestion.chunker import Chunk

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
TABLE_NAME = "chunks"
BATCH_SIZE = 64
ID_BATCH_SIZE = 500

log = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    """Lazy singleton: load bge-small-en-v1.5 once and keep it warm."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _fetch_existing_ids(ids: set[str], client: Client) -> set[str]:
    """Return the subset of ids already present in pgvector, batched to avoid URL limits."""
    if not ids:
        return set()
    id_list = list(ids)
    existing: set[str] = set()
    for start in range(0, len(id_list), ID_BATCH_SIZE):
        batch = id_list[start : start + ID_BATCH_SIZE]
        resp = (
            client.table(TABLE_NAME)
            .select("chunk_id")
            .in_("chunk_id", batch)
            .execute()
        )
        existing.update(row["chunk_id"] for row in resp.data)
    return existing


def _upsert_rows(
    chunks: list[Chunk],
    vectors: np.ndarray,
    client: Client,
) -> None:
    """Build row dicts and upsert to Supabase in batches."""
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
    for start in range(0, len(rows), BATCH_SIZE):
        client.table(TABLE_NAME).upsert(rows[start : start + BATCH_SIZE]).execute()


def embed_chunks(
    chunks: list[Chunk],
    client: Client,
    model: SentenceTransformer | None = None,
) -> int:
    """Encode new chunks and upsert to Supabase. Returns count of newly stored chunks."""
    if not chunks:
        return 0

    if model is None:
        model = _get_model()

    existing_ids = _fetch_existing_ids({c.chunk_id for c in chunks}, client)
    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

    if not new_chunks:
        return 0

    vectors = model.encode(
        [c.text for c in new_chunks],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
    )
    _upsert_rows(new_chunks, vectors, client)
    return len(new_chunks)
