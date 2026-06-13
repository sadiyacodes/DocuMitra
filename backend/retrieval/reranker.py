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
