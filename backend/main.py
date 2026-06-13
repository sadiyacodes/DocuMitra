"""FastAPI application entry point."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase import Client, create_client

from backend.generation.llm_client import generate
from backend.ingestion.chunker import chunk_document
from backend.ingestion.embed import embed_chunks
from backend.ingestion.extract import ExtractionError, extract_pdf
from backend.retrieval.reranker import rerank
from backend.retrieval.vector_store import search

log = logging.getLogger(__name__)

app = FastAPI(title="DocuMitra")


@lru_cache
def _get_supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def get_supabase() -> Client:
    return _get_supabase()
