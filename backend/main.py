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

from dotenv import load_dotenv

load_dotenv()  # load .env from project root before any os.environ reads

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class IngestResponse(BaseModel):
    pdf_id: str
    filename: str
    chunks_added: int


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    file: UploadFile,
    client: Client = Depends(get_supabase),
) -> IngestResponse:
    content = await file.read()
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        doc = extract_pdf(tmp_path, filename=file.filename or tmp_path.name)
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)
    chunks = chunk_document(doc)
    added = embed_chunks(chunks, client)
    return IngestResponse(pdf_id=doc.pdf_id, filename=doc.filename, chunks_added=added)


@app.get("/chunks")
def chunks_endpoint(
    query: str,
    k: int = 5,
    client: Client = Depends(get_supabase),
) -> dict:
    results = search(query, client, k=k)
    return {"results": [dataclasses.asdict(r) for r in results]}
