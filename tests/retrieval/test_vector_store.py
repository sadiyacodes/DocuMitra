from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock, patch

from backend.retrieval.vector_store import RPC_FUNCTION, TOP_K, SearchResult, search


def test_constants_defined():
    assert TOP_K == 20
    assert RPC_FUNCTION == "match_chunks"


def test_search_result_fields():
    result = SearchResult(
        chunk_id="abc123def456abcd",
        pdf_id="testpdf123456789",
        filename="doc.pdf",
        page_number=3,
        text="Some chunk text here.",
        token_count=4,
        language="en",
        bbox=[0.0, 0.0, 595.0, 842.0],
        similarity=0.92,
    )
    assert result.chunk_id == "abc123def456abcd"
    assert result.pdf_id == "testpdf123456789"
    assert result.filename == "doc.pdf"
    assert result.page_number == 3
    assert result.text == "Some chunk text here."
    assert result.token_count == 4
    assert result.language == "en"
    assert result.bbox == [0.0, 0.0, 595.0, 842.0]
    assert result.similarity == 0.92


def test_search_result_bbox_none():
    result = SearchResult(
        chunk_id="abc123def456abcd",
        pdf_id="testpdf123456789",
        filename="doc.pdf",
        page_number=1,
        text="Text.",
        token_count=1,
        language="en",
        bbox=None,
        similarity=0.85,
    )
    assert result.bbox is None


def _make_row(similarity: float = 0.92) -> dict:
    return {
        "chunk_id":    "abc123def456abcd",
        "pdf_id":      "testpdf123456789",
        "filename":    "doc.pdf",
        "page_number": 3,
        "text":        "Some chunk text here.",
        "token_count": 4,
        "language":    "en",
        "bbox":        [0.0, 0.0, 595.0, 842.0],
        "similarity":  similarity,
    }


def _make_model() -> MagicMock:
    mock = MagicMock()
    mock.encode.return_value = np.zeros(384)
    return mock


def _make_client(rows: list[dict] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.rpc.return_value.execute.return_value.data = rows or []
    return mock


def test_search_returns_empty_list_when_no_results():
    result = search("query", _make_client([]), model=_make_model())
    assert result == []


def test_search_maps_row_to_search_result():
    row = _make_row()
    results = search("query", _make_client([row]), model=_make_model())
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, SearchResult)
    assert r.chunk_id == row["chunk_id"]
    assert r.pdf_id == row["pdf_id"]
    assert r.filename == row["filename"]
    assert r.page_number == row["page_number"]
    assert r.text == row["text"]
    assert r.token_count == row["token_count"]
    assert r.language == row["language"]
    assert r.bbox == row["bbox"]
    assert r.similarity == row["similarity"]


def test_search_result_bbox_none_preserved():
    row = _make_row()
    row["bbox"] = None
    results = search("query", _make_client([row]), model=_make_model())
    assert results[0].bbox is None


def test_search_encodes_query_with_normalize_embeddings():
    mock_model = _make_model()
    search("what is revenue?", _make_client(), model=mock_model)
    mock_model.encode.assert_called_once_with("what is revenue?", normalize_embeddings=True)


def test_search_calls_rpc_with_correct_function():
    mock_client = _make_client()
    search("query", mock_client, model=_make_model())
    assert mock_client.rpc.call_args[0][0] == RPC_FUNCTION


def test_search_passes_vector_as_list_to_rpc():
    mock_model = _make_model()
    mock_model.encode.return_value = np.ones(384) * 0.5
    mock_client = _make_client()
    search("query", mock_client, model=mock_model)
    rpc_kwargs = mock_client.rpc.call_args[0][1]
    assert isinstance(rpc_kwargs["query_embedding"], list)
    assert len(rpc_kwargs["query_embedding"]) == 384
    assert all(isinstance(v, float) for v in rpc_kwargs["query_embedding"])


def test_search_passes_k_as_match_count():
    mock_client = _make_client()
    search("query", mock_client, k=5, model=_make_model())
    rpc_kwargs = mock_client.rpc.call_args[0][1]
    assert rpc_kwargs["match_count"] == 5


def test_search_default_k_is_top_k():
    mock_client = _make_client()
    search("query", mock_client, model=_make_model())
    rpc_kwargs = mock_client.rpc.call_args[0][1]
    assert rpc_kwargs["match_count"] == TOP_K


def test_search_uses_lazy_singleton_when_model_not_provided():
    mock_model = _make_model()
    mock_client = _make_client()
    with patch("backend.retrieval.vector_store._get_model", return_value=mock_model):
        search("query", mock_client)
    mock_model.encode.assert_called_once()


def test_search_returns_multiple_results_in_order():
    rows = [_make_row(similarity=0.95), _make_row(similarity=0.80)]
    results = search("query", _make_client(rows), model=_make_model())
    assert len(results) == 2
    assert results[0].similarity == 0.95
    assert results[1].similarity == 0.80
