from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import anthropic

from backend.generation.llm_client import (
    ANTHROPIC_MODEL,
    MAX_TOKENS,
    OLLAMA_MODEL,
    _generate_anthropic,
)


def _make_stream_cm(chunks: list[str]) -> MagicMock:
    """Mock Anthropic streaming context manager yielding text chunks."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.text_stream = iter(chunks)
    return cm


def test_constants_defined():
    assert ANTHROPIC_MODEL == "claude-sonnet-4-6"
    assert OLLAMA_MODEL == "gemma4:4b"
    assert MAX_TOKENS == 1024


def test_generate_anthropic_yields_chunks():
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _make_stream_cm(["Hello ", "world"])
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = list(_generate_anthropic("prompt", "system"))
    assert result == ["Hello ", "world"]


def test_generate_anthropic_uses_api_key_from_env():
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _make_stream_cm([])
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "my-secret-key"}):
        with patch("anthropic.Anthropic", return_value=mock_client) as mock_cls:
            list(_generate_anthropic("prompt", "system"))
    mock_cls.assert_called_once_with(api_key="my-secret-key")


def test_generate_anthropic_uses_correct_model():
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _make_stream_cm([])
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
        with patch("anthropic.Anthropic", return_value=mock_client):
            list(_generate_anthropic("prompt", "system"))
    kwargs = mock_client.messages.stream.call_args[1]
    assert kwargs["model"] == ANTHROPIC_MODEL


def test_generate_anthropic_passes_system_and_messages():
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _make_stream_cm([])
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
        with patch("anthropic.Anthropic", return_value=mock_client):
            list(_generate_anthropic("my prompt", "my system"))
    kwargs = mock_client.messages.stream.call_args[1]
    assert kwargs["system"] == "my system"
    assert kwargs["messages"] == [{"role": "user", "content": "my prompt"}]
    assert kwargs["max_tokens"] == MAX_TOKENS


def test_generate_anthropic_yields_empty_on_no_chunks():
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = _make_stream_cm([])
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = list(_generate_anthropic("prompt", "system"))
    assert result == []
