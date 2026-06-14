"""FastAPI application entry point."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # load .env from project root before any os.environ reads

from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase import Client, create_client

from backend.auth.dependencies import get_current_user
from backend.auth.models import User
from backend.auth.router import router as auth_router
from backend.generation.llm_client import generate
from backend.ingestion.chunker import chunk_document
from backend.ingestion.embed import embed_chunks
from backend.ingestion.extract import ExtractionError, extract_pdf
from backend.ingestion.ingest_csv import ingest_csv
from backend.ingestion.ingest_json import ingest_json
from backend.retrieval.reranker import rerank
from backend.retrieval.router import route_query
from backend.retrieval.vector_store import search

log = logging.getLogger(__name__)

app = FastAPI(title="DocuMitra")
app.include_router(auth_router)

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


def _sse_generator(
    query: str,
    client: Client,
    rerank_enabled: bool,
    user_role: str,
) -> Iterator[str]:
    source_types = route_query(query)
    results = search(query, client, user_role=user_role, source_types=source_types)
    if rerank_enabled:
        results = rerank(query, results)

    sources_payload = [
        {
            "filename": r.filename,
            "page": r.page_number,
            "similarity": round(r.similarity, 3),
            "source_type": r.source_type,
        }
        for r in results
    ]
    yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"

    for chunk in generate(query, results):
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/query")
def query_endpoint(
    req: QueryRequest,
    client: Client = Depends(get_supabase),
    current_user: User = Depends(get_current_user),
):
    return StreamingResponse(
        _sse_generator(req.query, client, req.rerank, current_user.role),
        media_type="text/event-stream",
    )


class IngestResponse(BaseModel):
    source_id: str
    filename: str
    chunks_added: int


def _parse_roles(roles: str) -> list[str]:
    return [r.strip() for r in roles.split(",") if r.strip()]


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    file: UploadFile,
    roles: str = Form(default=""),
    client: Client = Depends(get_supabase),
    current_user: User = Depends(get_current_user),
) -> IngestResponse:
    access_roles = _parse_roles(roles)
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
    chunks = chunk_document(doc, access_roles=access_roles)
    added = embed_chunks(chunks, client)
    return IngestResponse(source_id=doc.pdf_id, filename=doc.filename, chunks_added=added)


@app.post("/ingest/csv", response_model=IngestResponse)
async def ingest_csv_endpoint(
    file: UploadFile,
    roles: str = Form(default=""),
    client: Client = Depends(get_supabase),
    current_user: User = Depends(get_current_user),
) -> IngestResponse:
    access_roles = _parse_roles(roles)
    content = await file.read()
    filename = file.filename or "upload.csv"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        source_id = uuid.uuid4().hex[:16]
        chunks = ingest_csv(tmp_path, source_id=source_id, filename=filename, access_roles=access_roles)
    finally:
        tmp_path.unlink(missing_ok=True)
    added = embed_chunks(chunks, client)
    return IngestResponse(source_id=source_id, filename=filename, chunks_added=added)


@app.post("/ingest/json", response_model=IngestResponse)
async def ingest_json_endpoint(
    file: UploadFile,
    roles: str = Form(default=""),
    client: Client = Depends(get_supabase),
    current_user: User = Depends(get_current_user),
) -> IngestResponse:
    access_roles = _parse_roles(roles)
    content = await file.read()
    filename = file.filename or "upload.json"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        source_id = uuid.uuid4().hex[:16]
        chunks = ingest_json(tmp_path, source_id=source_id, filename=filename, access_roles=access_roles)
    finally:
        tmp_path.unlink(missing_ok=True)
    added = embed_chunks(chunks, client)
    return IngestResponse(source_id=source_id, filename=filename, chunks_added=added)


@app.get("/chunks")
def chunks_endpoint(
    query: str,
    k: int = 5,
    client: Client = Depends(get_supabase),
    current_user: User = Depends(get_current_user),
) -> dict:
    results = search(query, client, k=k, user_role=current_user.role)
    return {"results": [dataclasses.asdict(r) for r in results]}
