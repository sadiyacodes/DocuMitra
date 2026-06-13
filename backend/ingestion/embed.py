"""Encodes text chunks into vectors using BAAI/bge-small-en-v1.5."""
from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import Client

from backend.ingestion.chunker import Chunk

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
TABLE_NAME = "chunks"
BATCH_SIZE = 64

log = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy singleton: load bge-small-en-v1.5 once and keep it warm."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _fetch_existing_ids(ids: set[str], client: Client) -> set[str]:
    """Return the subset of ids already present in pgvector."""
    if not ids:
        return set()
    resp = client.table(TABLE_NAME).select("chunk_id").in_("chunk_id", list(ids)).execute()
    return {row["chunk_id"] for row in resp.data}
