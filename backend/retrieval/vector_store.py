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
