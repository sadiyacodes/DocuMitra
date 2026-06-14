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


import numpy as np

from backend.retrieval.reranker import rerank
from backend.retrieval.vector_store import SearchResult


def _make_result(idx: int = 0, text: str = "chunk text") -> SearchResult:
    return SearchResult(
        chunk_id=f"chunk{idx:016d}",
        source_id="testpdf123456789",
        source_type="pdf",
        filename="doc.pdf",
        page_number=idx + 1,
        text=text,
        token_count=5,
        language="en",
        bbox=None,
        access_roles=[],
        similarity=max(0.0, 0.9 - idx * 0.1),
    )


def _make_model(scores: list[float] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.predict.return_value = np.array(scores or [0.7])
    return mock


def test_rerank_disabled_returns_first_top_k():
    results = [_make_result(i) for i in range(10)]
    output = rerank("query", results, top_k=3, enabled=False)
    assert len(output) == 3
    assert output == results[:3]


def test_rerank_disabled_does_not_call_model():
    mock_model = _make_model([0.5])
    results = [_make_result(0)]
    rerank("query", results, top_k=1, model=mock_model, enabled=False)
    mock_model.predict.assert_not_called()


def test_rerank_empty_results_returns_empty():
    output = rerank("query", [], model=_make_model())
    assert output == []


def test_rerank_empty_results_does_not_call_model():
    mock_model = _make_model()
    rerank("query", [], model=mock_model)
    mock_model.predict.assert_not_called()


def test_rerank_sorts_by_score_descending():
    low = _make_result(0, text="low relevance")
    high = _make_result(1, text="high relevance")
    mock_model = _make_model([0.3, 0.9])
    output = rerank("query", [low, high], top_k=2, model=mock_model)
    assert output[0].text == "high relevance"
    assert output[1].text == "low relevance"


def test_rerank_truncates_to_top_k():
    results = [_make_result(i) for i in range(5)]
    mock_model = _make_model([0.5, 0.4, 0.3, 0.2, 0.1])
    output = rerank("query", results, top_k=3, model=mock_model)
    assert len(output) == 3


def test_rerank_passes_query_text_pairs_to_predict():
    result = _make_result(0, text="specific chunk text")
    mock_model = _make_model([0.7])
    rerank("my query", [result], top_k=1, model=mock_model)
    mock_model.predict.assert_called_once_with([("my query", "specific chunk text")])


def test_rerank_top_k_larger_than_results_returns_all_sorted():
    results = [_make_result(i) for i in range(3)]
    mock_model = _make_model([0.3, 0.9, 0.5])
    output = rerank("query", results, top_k=10, model=mock_model)
    assert len(output) == 3
    assert output[0].similarity == results[1].similarity


def test_rerank_default_top_k_is_top_k_rerank():
    results = [_make_result(i) for i in range(10)]
    mock_model = _make_model([float(i) * 0.1 for i in range(10)])
    output = rerank("query", results, model=mock_model)
    assert len(output) == TOP_K_RERANK


def test_rerank_uses_lazy_singleton_when_model_not_provided():
    results = [_make_result(0)]
    mock_model = _make_model([0.7])
    with patch("backend.retrieval.reranker._get_model", return_value=mock_model):
        rerank("query", results)
    mock_model.predict.assert_called_once()


def test_rerank_returns_list_of_search_result():
    results = [_make_result(0)]
    mock_model = _make_model([0.8])
    output = rerank("query", results, top_k=1, model=mock_model)
    assert all(isinstance(r, SearchResult) for r in output)
