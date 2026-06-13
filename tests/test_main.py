from __future__ import annotations

import dataclasses
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

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
