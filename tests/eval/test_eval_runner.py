from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from backend.eval.eval_runner import (
    EvalQuery,
    EvalReport,
    EvalResult,
    _citation_accuracy,
    _extract_citations,
    _p95,
    _recall_at_k,
    _reciprocal_rank,
    run_eval,
)
from backend.generation.prompt_templates import NO_ANSWER_RESPONSE
from backend.retrieval.vector_store import SearchResult


def _make_result(
    chunk_id: str = "abc",
    filename: str = "doc.pdf",
    page: int = 1,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        pdf_id="pdf1",
        filename=filename,
        page_number=page,
        text="Some text.",
        token_count=3,
        language="en",
        bbox=None,
        similarity=0.9,
    )


# ── _p95 ─────────────────────────────────────────────────────────────────────

def test_p95_basic():
    assert _p95(list(range(1, 101))) == 95


def test_p95_single_value():
    assert _p95([42.0]) == 42.0


def test_p95_two_values():
    assert _p95([10.0, 20.0]) == 20.0


def test_p95_raises_on_empty():
    with pytest.raises(ValueError):
        _p95([])


# ── _recall_at_k ──────────────────────────────────────────────────────────────

def test_recall_hit():
    assert _recall_at_k(["a", "b", "c"], {"b"}) == 1.0


def test_recall_miss():
    assert _recall_at_k(["a", "b", "c"], {"z"}) == 0.0


def test_recall_empty_retrieved():
    assert _recall_at_k([], {"a"}) == 0.0


# ── _reciprocal_rank ──────────────────────────────────────────────────────────

def test_rr_first():
    assert _reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_rr_second():
    assert _reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)


def test_rr_miss():
    assert _reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_rr_empty_retrieved():
    assert _reciprocal_rank([], {"a"}) == 0.0


# ── _extract_citations ────────────────────────────────────────────────────────

def test_extract_citations_single():
    assert _extract_citations("See [doc.pdf, p.3].") == [("doc.pdf", 3)]


def test_extract_citations_multiple():
    assert _extract_citations("From [a.pdf, p.1] and [b.pdf, p.10].") == [
        ("a.pdf", 1),
        ("b.pdf", 10),
    ]


def test_extract_citations_none():
    assert _extract_citations("No citations here.") == []


# ── _citation_accuracy ────────────────────────────────────────────────────────

def test_citation_accuracy_all_grounded():
    assert _citation_accuracy("Text [doc.pdf, p.1].", [_make_result("c1", "doc.pdf", 1)]) == 1.0


def test_citation_accuracy_ungrounded():
    assert _citation_accuracy("Text [missing.pdf, p.9].", [_make_result("c1", "doc.pdf", 1)]) == 0.0


def test_citation_accuracy_no_citations():
    assert _citation_accuracy("No cites.", [_make_result()]) == 1.0


def test_citation_accuracy_partial():
    answer = "[doc.pdf, p.1] and [other.pdf, p.5]."
    results = [_make_result("c1", "doc.pdf", 1)]
    assert _citation_accuracy(answer, results) == 0.5


# ── run_eval ──────────────────────────────────────────────────────────────────

def test_run_eval_raises_on_empty_queries():
    with pytest.raises(ValueError):
        run_eval([], MagicMock())


def test_run_eval_returns_eval_report():
    retrieved = [_make_result("c1")]
    with patch("backend.eval.eval_runner.search", return_value=retrieved):
        with patch("backend.eval.eval_runner.rerank_fn", return_value=retrieved):
            with patch("backend.eval.eval_runner.generate", return_value=iter(["answer"])):
                report = run_eval([EvalQuery("test?")], MagicMock())
    assert isinstance(report, EvalReport)
    assert report.n_queries == 1


def test_run_eval_p95_latency_is_non_negative():
    retrieved = [_make_result()]
    with patch("backend.eval.eval_runner.search", return_value=retrieved):
        with patch("backend.eval.eval_runner.rerank_fn", return_value=retrieved):
            with patch("backend.eval.eval_runner.generate", return_value=iter(["x"])):
                report = run_eval([EvalQuery("q?")], MagicMock())
    assert report.p95_latency_ms >= 0.0


def test_run_eval_recall_hit():
    retrieved = [_make_result("c1")]
    query = EvalQuery("q?", relevant_chunk_ids=["c1"])
    with patch("backend.eval.eval_runner.search", return_value=retrieved):
        with patch("backend.eval.eval_runner.rerank_fn", return_value=retrieved):
            with patch("backend.eval.eval_runner.generate", return_value=iter(["ans"])):
                report = run_eval([query], MagicMock())
    assert report.recall_at_k == 1.0


def test_run_eval_recall_miss():
    retrieved = [_make_result("c1")]
    query = EvalQuery("q?", relevant_chunk_ids=["other"])
    with patch("backend.eval.eval_runner.search", return_value=retrieved):
        with patch("backend.eval.eval_runner.rerank_fn", return_value=retrieved):
            with patch("backend.eval.eval_runner.generate", return_value=iter(["ans"])):
                report = run_eval([query], MagicMock())
    assert report.recall_at_k == 0.0


def test_run_eval_mrr_first_result():
    retrieved = [_make_result("c1"), _make_result("c2")]
    query = EvalQuery("q?", relevant_chunk_ids=["c1"])
    with patch("backend.eval.eval_runner.search", return_value=retrieved):
        with patch("backend.eval.eval_runner.rerank_fn", return_value=retrieved):
            with patch("backend.eval.eval_runner.generate", return_value=iter(["ans"])):
                report = run_eval([query], MagicMock())
    assert report.mrr == pytest.approx(1.0)


def test_run_eval_nan_metrics_without_ground_truth():
    retrieved = [_make_result()]
    with patch("backend.eval.eval_runner.search", return_value=retrieved):
        with patch("backend.eval.eval_runner.rerank_fn", return_value=retrieved):
            with patch("backend.eval.eval_runner.generate", return_value=iter(["ans"])):
                report = run_eval([EvalQuery("q?")], MagicMock())
    assert math.isnan(report.recall_at_k)
    assert math.isnan(report.mrr)
    assert math.isnan(report.hallucination_rate)


def test_run_eval_hallucination_detected():
    # retrieved c1, but relevant is "other" — system confidently answers → hallucination
    retrieved = [_make_result("c1")]
    query = EvalQuery("q?", relevant_chunk_ids=["other"])
    with patch("backend.eval.eval_runner.search", return_value=retrieved):
        with patch("backend.eval.eval_runner.rerank_fn", return_value=retrieved):
            with patch("backend.eval.eval_runner.generate", return_value=iter(["confident answer"])):
                report = run_eval([query], MagicMock())
    assert report.hallucination_rate == 1.0


def test_run_eval_no_hallucination_when_no_answer_response():
    # system correctly says NO_ANSWER_RESPONSE when no relevant chunk retrieved
    retrieved = [_make_result("c1")]
    query = EvalQuery("q?", relevant_chunk_ids=["other"])
    with patch("backend.eval.eval_runner.search", return_value=retrieved):
        with patch("backend.eval.eval_runner.rerank_fn", return_value=retrieved):
            with patch("backend.eval.eval_runner.generate", return_value=iter([NO_ANSWER_RESPONSE])):
                report = run_eval([query], MagicMock())
    assert report.hallucination_rate == 0.0


def test_run_eval_citation_accuracy_grounded():
    retrieved = [_make_result("c1", "doc.pdf", 1)]
    with patch("backend.eval.eval_runner.search", return_value=retrieved):
        with patch("backend.eval.eval_runner.rerank_fn", return_value=retrieved):
            with patch("backend.eval.eval_runner.generate", return_value=iter(["Answer [doc.pdf, p.1]."])):
                report = run_eval([EvalQuery("q?")], MagicMock())
    assert report.citation_accuracy == 1.0


def test_run_eval_skips_rerank_when_disabled():
    retrieved = [_make_result()]
    with patch("backend.eval.eval_runner.search", return_value=retrieved):
        with patch("backend.eval.eval_runner.rerank_fn") as mock_rerank:
            with patch("backend.eval.eval_runner.generate", return_value=iter(["x"])):
                run_eval([EvalQuery("q?")], MagicMock(), use_rerank=False)
    mock_rerank.assert_not_called()
