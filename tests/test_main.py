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
