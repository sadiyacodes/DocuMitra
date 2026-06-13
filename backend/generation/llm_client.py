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
OLLAMA_MODEL    = "gemma4:4b"
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
