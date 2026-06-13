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
    _generate_ollama,
    generate,
)
from backend.retrieval.vector_store import SearchResult


def _make_stream_cm(chunks: list[str]) -> MagicMock:
    """Mock Anthropic streaming context manager yielding text chunks."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.text_stream = iter(chunks)
    return cm


def test_constants_defined():
    assert ANTHROPIC_MODEL == "claude-sonnet-4-6"
    assert OLLAMA_MODEL == "gemma4:e4b"
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


def _make_ollama_response(chunks: list[str]) -> MagicMock:
    """Mock requests streaming response with newline-delimited JSON lines."""
    lines = [
        json.dumps({"message": {"content": c}, "done": False}).encode()
        for c in chunks
    ]
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    resp.iter_lines.return_value = iter(lines)
    return resp


def test_generate_ollama_yields_chunks():
    with patch("requests.post", return_value=_make_ollama_response(["chunk1", "chunk2"])):
        result = list(_generate_ollama("prompt", "system"))
    assert result == ["chunk1", "chunk2"]


def test_generate_ollama_uses_ollama_url_from_env():
    with patch.dict("os.environ", {"OLLAMA_URL": "http://custom-ollama:1234"}):
        with patch("requests.post", return_value=_make_ollama_response([])) as mock_post:
            list(_generate_ollama("prompt", "system"))
    assert mock_post.call_args[0][0] == "http://custom-ollama:1234/api/chat"


def test_generate_ollama_uses_default_url_when_env_not_set():
    env_without_ollama = {k: v for k, v in os.environ.items() if k != "OLLAMA_URL"}
    with patch.dict("os.environ", env_without_ollama, clear=True):
        with patch("requests.post", return_value=_make_ollama_response([])) as mock_post:
            list(_generate_ollama("prompt", "system"))
    assert mock_post.call_args[0][0] == "http://localhost:11434/api/chat"


def test_generate_ollama_sends_system_and_user_messages():
    with patch("requests.post", return_value=_make_ollama_response([])) as mock_post:
        list(_generate_ollama("user prompt", "sys instruction"))
    payload = mock_post.call_args[1]["json"]
    assert {"role": "system", "content": "sys instruction"} in payload["messages"]
    assert {"role": "user", "content": "user prompt"} in payload["messages"]


def test_generate_ollama_sends_correct_model():
    with patch("requests.post", return_value=_make_ollama_response([])) as mock_post:
        list(_generate_ollama("prompt", "system"))
    assert mock_post.call_args[1]["json"]["model"] == OLLAMA_MODEL


def test_generate_ollama_streams_enabled():
    with patch("requests.post", return_value=_make_ollama_response([])) as mock_post:
        list(_generate_ollama("prompt", "system"))
    assert mock_post.call_args[1]["json"]["stream"] is True


def test_generate_ollama_empty_content_chunks_skipped():
    lines = [
        json.dumps({"message": {"content": ""}, "done": False}).encode(),
        json.dumps({"message": {"content": "real"}, "done": False}).encode(),
    ]
    resp = _make_ollama_response([])
    resp.iter_lines.return_value = iter(lines)
    with patch("requests.post", return_value=resp):
        result = list(_generate_ollama("prompt", "system"))
    assert result == ["real"]


def test_generate_yields_anthropic_chunks():
    chunks = ["Hello ", "world"]
    with patch("backend.generation.llm_client._generate_anthropic", return_value=iter(chunks)):
        result = list(generate("query", []))
    assert result == chunks


def test_generate_falls_back_to_ollama_on_api_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 529
    mock_resp.headers = {}
    api_error = anthropic.APIStatusError("overloaded", response=mock_resp, body={})
    fallback_chunks = ["fallback ", "answer"]

    with patch("backend.generation.llm_client._generate_anthropic", side_effect=api_error):
        with patch("backend.generation.llm_client._generate_ollama", return_value=iter(fallback_chunks)):
            result = list(generate("query", []))

    assert result == fallback_chunks


def test_generate_logs_warning_on_fallback():
    mock_resp = MagicMock()
    mock_resp.status_code = 529
    mock_resp.headers = {}
    api_error = anthropic.APIStatusError("overloaded", response=mock_resp, body={})

    with patch("backend.generation.llm_client._generate_anthropic", side_effect=api_error):
        with patch("backend.generation.llm_client._generate_ollama", return_value=iter([])):
            with patch("backend.generation.llm_client.log") as mock_log:
                list(generate("query", []))

    mock_log.warning.assert_called_once()


def test_generate_calls_build_user_message_with_query_and_results():
    results = [
        SearchResult(
            chunk_id="abc123def456abcd",
            pdf_id="testpdf123456789",
            filename="doc.pdf",
            page_number=1,
            text="Some text.",
            token_count=3,
            language="en",
            bbox=None,
            similarity=0.9,
        )
    ]
    with patch("backend.generation.llm_client.build_user_message", return_value="built") as mock_build:
        with patch("backend.generation.llm_client._generate_anthropic", return_value=iter([])):
            list(generate("my query", results))
    mock_build.assert_called_once_with("my query", results)


def test_generate_does_not_fall_back_on_non_api_error():
    with patch("backend.generation.llm_client._generate_anthropic", side_effect=ValueError("unexpected")):
        with patch("backend.generation.llm_client._generate_ollama", return_value=iter([])) as mock_ollama:
            try:
                list(generate("query", []))
            except ValueError:
                pass
    mock_ollama.assert_not_called()
