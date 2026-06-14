"""Keyword-based query routing to data source types."""
from __future__ import annotations

_SOURCE_HINTS: dict[str, list[str]] = {
    "csv": [
        "employee", "salary", "department", "budget", "headcount",
        "hire", "payroll", "staff", "record", "workforce",
    ],
    "json": [
        "log", "audit", "alert", "event", "error", "trace",
        "timestamp", "incident", "access", "login", "logout",
    ],
    "pdf": [
        "policy", "compliance", "regulation", "guideline", "chapter",
        "section", "document", "handbook", "procedure", "clause",
    ],
}

ALL_TYPES = ["pdf", "csv", "json"]


def route_query(query: str) -> list[str]:
    """Return source types to search based on query keywords.

    Returns all types when the query is ambiguous or empty.
    Never returns duplicate types.
    """
    q = query.lower()
    scores = {src: sum(1 for kw in hints if kw in q) for src, hints in _SOURCE_HINTS.items()}
    best = max(scores.values())
    if best == 0:
        return list(ALL_TYPES)
    return [src for src, score in scores.items() if score == best]
