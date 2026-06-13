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


def run_eval(queries: list[EvalQuery], client: Client, k: int = 20, use_rerank: bool = True) -> EvalReport:
    raise NotImplementedError
