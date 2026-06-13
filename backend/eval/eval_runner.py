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
