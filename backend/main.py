"""FastAPI application entry point."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from collections.abc import Iterator
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


class QueryRequest(BaseModel):
    query: str
    rerank: bool = True


def _sse_generator(query: str, client: Client, rerank_enabled: bool) -> Iterator[str]:
    results = search(query, client)
    if rerank_enabled:
        results = rerank(query, results)
    for chunk in generate(query, results):
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/query")
def query_endpoint(req: QueryRequest, client: Client = Depends(get_supabase)):
    return StreamingResponse(
        _sse_generator(req.query, client, req.rerank),
        media_type="text/event-stream",
    )
