"""Anthropic primary + Ollama fallback LLM client."""
from __future__ import annotations

import json
import logging
import os
from typing import Iterator

import anthropic
import requests

from backend.generation.prompt_templates import SYSTEM_PROMPT, build_user_message
from backend.retrieval.vector_store import SearchResult

ANTHROPIC_MODEL = "claude-sonnet-4-6"
OLLAMA_MODEL    = "gemma4:e4b"
MAX_TOKENS      = 1024

log = logging.getLogger(__name__)


def _generate_anthropic(prompt: str, system: str) -> Iterator[str]:
    """Stream text chunks from Anthropic claude-sonnet-4-6."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    with client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def _generate_ollama(prompt: str, system: str) -> Iterator[str]:
    """Stream text chunks from Ollama via HTTP."""
    url = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
    }
    with requests.post(url, json=payload, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                if chunk := json.loads(line).get("message", {}).get("content", ""):
                    yield chunk


def generate(
    query: str,
    results: list[SearchResult],
) -> Iterator[str]:
    """Stream an answer with citations; falls back to Ollama if Anthropic fails."""
    system = SYSTEM_PROMPT
    prompt = build_user_message(query, results)
    try:
        yield from _generate_anthropic(prompt, system)
    except anthropic.APIError as exc:
        log.warning("Anthropic failed (%s), falling back to Ollama", exc)
        yield from _generate_ollama(prompt, system)
