# eval_runner.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `backend/eval/eval_runner.py` — an offline evaluator that runs the full RAG pipeline against a set of labelled queries and reports p95 latency, R@k, MRR, citation accuracy, and hallucination rate.

**Architecture:** The module has two layers: (1) pure helper functions for each metric (`_p95`, `_recall_at_k`, `_reciprocal_rank`, `_extract_citations`, `_citation_accuracy`) that are independently testable, and (2) `run_eval` which orchestrates the full pipeline per query and calls `_compute_report` to aggregate. All ground-truth metrics (R@k, MRR, hallucination rate) default to `math.nan` when no `relevant_chunk_ids` are provided, so the evaluator is usable with or without labels.

**Tech Stack:** Python stdlib (`math`, `re`, `statistics`, `time`, `dataclasses`), existing pipeline modules (`search`, `rerank_fn`, `generate`, `NO_ANSWER_RESPONSE`), pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `backend/eval/eval_runner.py` | Modify (replace 3-line stub) | All eval logic |
| `tests/eval/__init__.py` | Create | Package marker |
| `tests/eval/test_eval_runner.py` | Create | All 29 tests |

---

### Task 1: Data models + metric helper functions

**Files:**
- Modify: `backend/eval/eval_runner.py`
- Create: `tests/eval/__init__.py`
- Create: `tests/eval/test_eval_runner.py`

The five pure helper functions plus the three dataclasses. No pipeline calls in this task.

**Key types:**
- `EvalQuery(query: str, relevant_chunk_ids: list[str])` — input; `relevant_chunk_ids` defaults to `[]`
- `EvalResult(query, latency_ms, retrieved_chunk_ids, retrieved_results, answer)` — per-query output
- `EvalReport(n_queries, p95_latency_ms, recall_at_k, mrr, citation_accuracy, hallucination_rate)` — final summary

**`_p95` formula:** `sorted_vals[max(0, math.ceil(0.95 * n) - 1)]`. Verified:
- `[1..100]` → index 94 → value 95 ✓
- `[10.0, 20.0]` → `ceil(1.9)-1 = 1` → 20.0 ✓
- `[42.0]` → `ceil(0.95)-1 = 0` → 42.0 ✓

- [ ] **Step 1: Create package marker and write failing tests**

Create `tests/eval/__init__.py` (empty file).

Create `tests/eval/test_eval_runner.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/sadiya/projects/DocuMitra
python3 -m pytest tests/eval/ -v --tb=short 2>&1 | head -30
```

Expected: `ImportError` — `backend.eval.eval_runner` has none of these names yet.

- [ ] **Step 3: Implement data models and helpers**

Replace `backend/eval/eval_runner.py` entirely:

```python
"""Evaluation: p95 latency, R@k, MRR, citation accuracy, hallucination rate."""
from __future__ import annotations

import math
import re
import statistics
import time
from dataclasses import dataclass, field

from supabase import Client

from backend.generation.llm_client import generate
from backend.generation.prompt_templates import NO_ANSWER_RESPONSE
from backend.retrieval.reranker import rerank as rerank_fn
from backend.retrieval.vector_store import SearchResult, search

CITATION_RE = re.compile(r"\[([^\]]+),\s*p\.(\d+)\]")


@dataclass
class EvalQuery:
    query: str
    relevant_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    query: str
    latency_ms: float
    retrieved_chunk_ids: list[str]
    retrieved_results: list[SearchResult]
    answer: str


@dataclass
class EvalReport:
    n_queries: int
    p95_latency_ms: float
    recall_at_k: float
    mrr: float
    citation_accuracy: float
    hallucination_rate: float


def _p95(values: list[float]) -> float:
    """95th percentile of a non-empty list of floats."""
    if not values:
        raise ValueError("values must be non-empty")
    sorted_vals = sorted(values)
    idx = math.ceil(0.95 * len(sorted_vals)) - 1
    return sorted_vals[max(0, idx)]


def _recall_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1.0 if any relevant chunk was retrieved, else 0.0."""
    return 1.0 if any(cid in relevant_ids for cid in retrieved_ids) else 0.0


def _reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant result; 0.0 if none found."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _extract_citations(answer: str) -> list[tuple[str, int]]:
    """Extract (filename, page_number) pairs from [filename, p.N] markers."""
    return [(m.group(1), int(m.group(2))) for m in CITATION_RE.finditer(answer)]


def _citation_accuracy(answer: str, results: list[SearchResult]) -> float:
    """Fraction of cited (filename, page) pairs present in retrieved results."""
    citations = _extract_citations(answer)
    if not citations:
        return 1.0
    grounded = {(r.filename, r.page_number) for r in results}
    return sum(1 for c in citations if c in grounded) / len(citations)
```

Note: `run_eval` and `_compute_report` are deliberately omitted — they are added in Task 2. The file as written will make all Task 1 tests pass.

- [ ] **Step 4: Run to verify they pass**

```bash
python3 -m pytest tests/eval/ -v --tb=short
```

Expected: 18 PASS (4 `_p95` + 3 recall + 4 rr + 3 citations + 4 accuracy). The `run_eval` tests will not be collected yet (they're not in the file).

- [ ] **Step 5: Run full suite to check no regressions**

```bash
python3 -m pytest --tb=short -q
```

Expected: 176 + 18 = 194 tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/eval/eval_runner.py tests/eval/__init__.py tests/eval/test_eval_runner.py
git commit -m "feat: add eval dataclasses and pure metric helpers"
```

---

### Task 2: run_eval + _compute_report

**Files:**
- Modify: `backend/eval/eval_runner.py` (append `run_eval` and `_compute_report`)
- Modify: `tests/eval/test_eval_runner.py` (append 11 tests)

`run_eval` orchestrates the full pipeline for each query (timed), then delegates to `_compute_report`. All ground-truth metrics (`recall_at_k`, `mrr`, `hallucination_rate`) are `math.nan` when no query has `relevant_chunk_ids`.

**Hallucination definition used here:** A query has a "hallucination" when the answer is NOT `NO_ANSWER_RESPONSE` **and** no retrieved chunk ID appears in `relevant_chunk_ids`. This captures the case where the system answered confidently despite having no relevant context. Only queries that have non-empty `relevant_chunk_ids` contribute to this metric.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_eval_runner.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/sadiya/projects/DocuMitra
python3 -m pytest tests/eval/test_eval_runner.py -k "run_eval" -v 2>&1 | head -20
```

Expected: `ImportError` — `run_eval` not yet defined.

- [ ] **Step 3: Implement run_eval and _compute_report**

Append to `backend/eval/eval_runner.py` (after `_citation_accuracy`):

```python

def _compute_report(
    queries: list[EvalQuery],
    results: list[EvalResult],
) -> EvalReport:
    """Aggregate per-query EvalResults into a summary EvalReport."""
    latencies = [r.latency_ms for r in results]

    gt_pairs = [
        (eq, r)
        for eq, r in zip(queries, results)
        if eq.relevant_chunk_ids
    ]

    if gt_pairs:
        recall_scores = [
            _recall_at_k(r.retrieved_chunk_ids, set(eq.relevant_chunk_ids))
            for eq, r in gt_pairs
        ]
        mrr_scores = [
            _reciprocal_rank(r.retrieved_chunk_ids, set(eq.relevant_chunk_ids))
            for eq, r in gt_pairs
        ]
        hallucination_count = sum(
            1
            for eq, r in gt_pairs
            if r.answer != NO_ANSWER_RESPONSE
            and not any(cid in set(eq.relevant_chunk_ids) for cid in r.retrieved_chunk_ids)
        )
        recall_at_k = statistics.mean(recall_scores)
        mrr = statistics.mean(mrr_scores)
        hallucination_rate: float = hallucination_count / len(gt_pairs)
    else:
        recall_at_k = math.nan
        mrr = math.nan
        hallucination_rate = math.nan

    citation_accuracies = [
        _citation_accuracy(r.answer, r.retrieved_results) for r in results
    ]

    return EvalReport(
        n_queries=len(queries),
        p95_latency_ms=_p95(latencies),
        recall_at_k=recall_at_k,
        mrr=mrr,
        citation_accuracy=statistics.mean(citation_accuracies),
        hallucination_rate=hallucination_rate,
    )


def run_eval(
    queries: list[EvalQuery],
    client: Client,
    k: int = 20,
    use_rerank: bool = True,
) -> EvalReport:
    """Run the full RAG pipeline for each query and compute evaluation metrics."""
    if not queries:
        raise ValueError("queries must be non-empty")

    results: list[EvalResult] = []

    for eq in queries:
        t0 = time.perf_counter()
        retrieved = search(eq.query, client, k=k)
        if use_rerank:
            retrieved = rerank_fn(eq.query, retrieved)
        answer = "".join(generate(eq.query, retrieved))
        latency_ms = (time.perf_counter() - t0) * 1000

        results.append(EvalResult(
            query=eq.query,
            latency_ms=latency_ms,
            retrieved_chunk_ids=[r.chunk_id for r in retrieved],
            retrieved_results=retrieved,
            answer=answer,
        ))

    return _compute_report(queries, results)
```

- [ ] **Step 4: Run to verify they pass**

```bash
python3 -m pytest tests/eval/test_eval_runner.py -k "run_eval" -v
```

Expected: all 11 PASS.

- [ ] **Step 5: Run full suite**

```bash
python3 -m pytest --tb=short -q
```

Expected: 176 + 29 = 205 tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/eval/eval_runner.py tests/eval/test_eval_runner.py
git commit -m "feat: add run_eval and _compute_report pipeline orchestration"
```

---

## Self-Review

**Spec coverage check:**

| CLAUDE.md requirement | Task covering it |
|----------------------|-----------------|
| p95 latency | Task 1 (`_p95`) + Task 2 (`run_eval` times each query) |
| R@k | Task 1 (`_recall_at_k`) + Task 2 (`_compute_report`) |
| MRR | Task 1 (`_reciprocal_rank`) + Task 2 (`_compute_report`) |
| Citation accuracy | Task 1 (`_extract_citations`, `_citation_accuracy`) + Task 2 |
| Hallucination rate | Task 2 (`_compute_report` — counts confident answers with no relevant context) |
| "eval_runner.py should test hallucination directly" | `test_run_eval_hallucination_detected` + `test_run_eval_no_hallucination_when_no_answer_response` |

**Placeholder scan:** No TBDs, todos, or vague steps found.

**Type consistency:**
- `_recall_at_k(retrieved_ids: list[str], relevant_ids: set[str])` — used in `_compute_report` as `_recall_at_k(r.retrieved_chunk_ids, set(eq.relevant_chunk_ids))` ✅
- `_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str])` — same pattern ✅
- `_citation_accuracy(answer: str, results: list[SearchResult])` — used in `_compute_report` as `_citation_accuracy(r.answer, r.retrieved_results)` where `r.retrieved_results: list[SearchResult]` ✅
- `EvalResult.retrieved_results: list[SearchResult]` — populated in `run_eval` from `retrieved` which is `list[SearchResult]` ✅
- `rerank_fn` alias for `rerank` from `reranker.py` — avoids collision with `EvalQuery` field names ✅
