"""Optional cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)."""
from __future__ import annotations

import logging
import threading

from sentence_transformers import CrossEncoder

from backend.retrieval.vector_store import SearchResult

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANK = 5

log = logging.getLogger(__name__)

_model: CrossEncoder | None = None
_model_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    """Lazy singleton: load cross-encoder once and keep it warm."""
    global _model
    if _model is None:
        with _model_lock:
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
    """Rerank results with cross-encoder; when disabled returns results[:top_k] without loading the model."""
    if not enabled or not results:
        return results[:top_k]

    if model is None:
        model = _get_model()

    log.debug("reranking %d results → top %d with %s", len(results), top_k, RERANK_MODEL)
    scores = model.predict([(query, r.text) for r in results])
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_k]]
