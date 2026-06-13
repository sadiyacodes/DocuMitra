from __future__ import annotations

import dataclasses
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.ingestion.extract import ExtractionError
from backend.main import app, get_supabase
from backend.retrieval.vector_store import SearchResult


mock_supabase = MagicMock()
app.dependency_overrides[get_supabase] = lambda: mock_supabase
client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_mocks():
    mock_supabase.reset_mock()


def _make_result(
    filename: str = "doc.pdf",
    page: int = 1,
    text: str = "Sample text.",
) -> SearchResult:
    return SearchResult(
        chunk_id="abc123def456abcd",
        pdf_id="pdf123",
        filename=filename,
        page_number=page,
        text=text,
        token_count=3,
        language="en",
        bbox=None,
        similarity=0.9,
    )


def test_get_supabase_calls_create_client_with_env_vars():
    from backend.main import _get_supabase

    _get_supabase.cache_clear()
    with patch("backend.main.create_client") as mock_create:
        with patch.dict(
            "os.environ",
            {"SUPABASE_URL": "http://sb.local", "SUPABASE_KEY": "anon-key"},
        ):
            _get_supabase()
    mock_create.assert_called_once_with("http://sb.local", "anon-key")
    _get_supabase.cache_clear()


# ── POST /query ─────────────────────────────────────────────────────────────

def test_query_returns_200_with_sse_content_type():
    with patch("backend.main.search", return_value=[_make_result()]):
        with patch("backend.main.rerank", return_value=[_make_result()]):
            with patch("backend.main.generate", return_value=iter(["answer"])):
                response = client.post("/query", json={"query": "test"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_query_streams_sse_chunks():
    chunks = ["Hello ", "world"]
    with patch("backend.main.search", return_value=[_make_result()]):
        with patch("backend.main.rerank", return_value=[_make_result()]):
            with patch("backend.main.generate", return_value=iter(chunks)):
                response = client.post("/query", json={"query": "test"})
    events = [e for e in response.text.split("\n\n") if e.startswith("data: ")]
    data_values = [e[len("data: "):] for e in events]
    assert json.dumps("Hello ") in data_values
    assert json.dumps("world") in data_values


def test_query_sends_done_sentinel():
    with patch("backend.main.search", return_value=[_make_result()]):
        with patch("backend.main.rerank", return_value=[_make_result()]):
            with patch("backend.main.generate", return_value=iter(["text"])):
                response = client.post("/query", json={"query": "test"})
    events = [e for e in response.text.split("\n\n") if e.startswith("data: ")]
    data_values = [e[len("data: "):] for e in events]
    assert "[DONE]" in data_values


def test_query_calls_rerank_when_enabled():
    fake = [_make_result()]
    with patch("backend.main.search", return_value=fake):
        with patch("backend.main.rerank", return_value=fake) as mock_r:
            with patch("backend.main.generate", return_value=iter(["x"])):
                client.post("/query", json={"query": "hello"})
    mock_r.assert_called_once_with("hello", fake)


def test_query_skips_rerank_when_disabled():
    with patch("backend.main.search", return_value=[_make_result()]):
        with patch("backend.main.rerank") as mock_r:
            with patch("backend.main.generate", return_value=iter(["x"])):
                client.post("/query", json={"query": "test", "rerank": False})
    mock_r.assert_not_called()


# ── POST /ingest ─────────────────────────────────────────────────────────────

def _make_upload(
    filename: str = "test.pdf",
    content: bytes = b"PDF content",
) -> tuple:
    return ("file", (filename, io.BytesIO(content), "application/pdf"))


def test_ingest_returns_pdf_id_filename_and_count():
    fake_doc = MagicMock()
    fake_doc.pdf_id = "pdf_abc123"
    fake_doc.filename = "test.pdf"

    with patch("backend.main.extract_pdf", return_value=fake_doc):
        with patch("backend.main.chunk_document", return_value=[MagicMock()]):
            with patch("backend.main.embed_chunks", return_value=7):
                response = client.post("/ingest", files=[_make_upload()])

    assert response.status_code == 200
    data = response.json()
    assert data["pdf_id"] == "pdf_abc123"
    assert data["filename"] == "test.pdf"
    assert data["chunks_added"] == 7


def test_ingest_422_on_extraction_error():
    with patch(
        "backend.main.extract_pdf",
        side_effect=ExtractionError("bad.pdf", "corrupt file"),
    ):
        response = client.post("/ingest", files=[_make_upload("bad.pdf", b"garbage")])

    assert response.status_code == 422
    assert "bad.pdf" in response.json()["detail"]


def test_ingest_calls_full_pipeline_in_order():
    fake_doc = MagicMock()
    fake_doc.pdf_id = "p1"
    fake_doc.filename = "doc.pdf"
    fake_chunks = [MagicMock(), MagicMock()]

    with patch("backend.main.extract_pdf", return_value=fake_doc) as mock_extract:
        with patch("backend.main.chunk_document", return_value=fake_chunks) as mock_chunk:
            with patch("backend.main.embed_chunks", return_value=2) as mock_embed:
                client.post("/ingest", files=[_make_upload()])

    mock_extract.assert_called_once()
    mock_chunk.assert_called_once_with(fake_doc)
    mock_embed.assert_called_once_with(fake_chunks, mock_supabase)
