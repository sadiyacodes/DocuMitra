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
