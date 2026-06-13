from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.retrieval.reranker import RERANK_MODEL, TOP_K_RERANK, _get_model


def test_constants_defined():
    assert RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert TOP_K_RERANK == 5


def test_get_model_loads_cross_encoder():
    with patch("backend.retrieval.reranker.CrossEncoder") as mock_ce:
        mock_ce.return_value = MagicMock()
        import backend.retrieval.reranker as reranker_mod
        reranker_mod._model = None
        result = _get_model()
    mock_ce.assert_called_once_with(RERANK_MODEL)
    assert result is mock_ce.return_value


def test_get_model_singleton_cached():
    with patch("backend.retrieval.reranker.CrossEncoder") as mock_ce:
        mock_ce.return_value = MagicMock()
        import backend.retrieval.reranker as reranker_mod
        reranker_mod._model = None
        r1 = _get_model()
        r2 = _get_model()
    assert r1 is r2
    mock_ce.assert_called_once()
