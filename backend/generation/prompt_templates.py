"""All LLM prompt templates — never inline strings."""
from __future__ import annotations

from backend.retrieval.vector_store import SearchResult

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer only from the provided excerpts.\n"
    "Cite every claim immediately after the sentence using [filename, p.N].\n"
    "If there is not enough information in the excerpts to answer the question, respond with:\n"
    "\"I don't have enough information in the provided documents to answer this question.\"\n"
    "Do not guess or add information not present in the excerpts."
)


def build_user_message(query: str, results: list[SearchResult]) -> str:
    """Build the user-turn message: context block followed by the question."""
    context = "\n\n".join(
        f"[{r.filename}, p.{r.page_number}]\n{r.text}"
        for r in results
    )
    if context:
        return f"{context}\n\nQuestion: {query}"
    return f"Question: {query}"
