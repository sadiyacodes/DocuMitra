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
