"""pgvector-backed top-k retrieval via Supabase."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from supabase import Client

from backend.ingestion.embed import _get_model

TOP_K = 20
RPC_FUNCTION = "match_chunks"

log = logging.getLogger(__name__)


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
    fetch_k = k * 3 if source_types else k

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
