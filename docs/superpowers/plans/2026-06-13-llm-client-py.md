# llm_client.py + prompt_templates.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `backend/generation/llm_client.py` and `backend/generation/prompt_templates.py` — streaming RAG answer generation with Anthropic primary and Ollama fallback, citation-formatted prompts.

**Architecture:** `prompt_templates.py` owns all strings (SYSTEM_PROMPT + build_user_message). `llm_client.py` has three functions: `_generate_anthropic` and `_generate_ollama` are independent streaming generators; `generate` orchestrates them with try/except fallback. All three return `Iterator[str]`.

**Tech Stack:** Python 3.11+, `anthropic` SDK (streaming), `requests` (Ollama HTTP), pytest.

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `backend/generation/prompt_templates.py` | Modify (was stub) | SYSTEM_PROMPT constant + build_user_message |
| `backend/generation/llm_client.py` | Modify (was stub) | Constants, _generate_anthropic, _generate_ollama, generate |
| `tests/generation/__init__.py` | Create | Package marker |
| `tests/generation/test_prompt_templates.py` | Create | TDD tests for prompt_templates |
| `tests/generation/test_llm_client.py` | Create | TDD tests for llm_client |

---

## Task 1: prompt_templates.py

**Files:**
- Create: `tests/generation/__init__.py`
- Create: `tests/generation/test_prompt_templates.py`
- Modify: `backend/generation/prompt_templates.py`

- [ ] **Step 1: Create test package marker**

```bash
touch /Users/sadiya/projects/DocuMitra/tests/generation/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/generation/test_prompt_templates.py`:

```python
from __future__ import annotations

from backend.retrieval.vector_store import SearchResult
from backend.generation.prompt_templates import SYSTEM_PROMPT, build_user_message


def _make_result(filename: str = "doc.pdf", page: int = 3, text: str = "Some text.") -> SearchResult:
    return SearchResult(
        chunk_id="abc123def456abcd",
        pdf_id="testpdf123456789",
        filename=filename,
        page_number=page,
        text=text,
        token_count=3,
        language="en",
        bbox=None,
        similarity=0.9,
    )


def test_system_prompt_is_nonempty_string():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0


def test_system_prompt_contains_citation_format():
    assert "[filename, p.N]" in SYSTEM_PROMPT


def test_system_prompt_contains_no_info_instruction():
    assert "not enough information" in SYSTEM_PROMPT


def test_system_prompt_prohibits_guessing():
    assert "not present in the excerpts" in SYSTEM_PROMPT


def test_build_user_message_includes_filename():
    result = _make_result(filename="annual_report.pdf")
    msg = build_user_message("query", [result])
    assert "annual_report.pdf" in msg


def test_build_user_message_includes_page_number():
    result = _make_result(page=7)
    msg = build_user_message("query", [result])
    assert "p.7" in msg


def test_build_user_message_includes_chunk_text():
    result = _make_result(text="Revenue grew 12% YoY.")
    msg = build_user_message("query", [result])
    assert "Revenue grew 12% YoY." in msg


def test_build_user_message_includes_query():
    result = _make_result()
    msg = build_user_message("What is the revenue?", [result])
    assert "What is the revenue?" in msg


def test_build_user_message_empty_results_still_includes_query():
    msg = build_user_message("What is the capital?", [])
    assert "What is the capital?" in msg


def test_build_user_message_multiple_results_all_included():
    results = [
        _make_result(filename="a.pdf", page=1, text="First chunk."),
        _make_result(filename="b.pdf", page=2, text="Second chunk."),
    ]
    msg = build_user_message("query", results)
    assert "a.pdf" in msg
    assert "b.pdf" in msg
    assert "First chunk." in msg
    assert "Second chunk." in msg
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/generation/test_prompt_templates.py -v
```
Expected: `ImportError: cannot import name 'SYSTEM_PROMPT'`

- [ ] **Step 4: Implement prompt_templates.py**

Replace `backend/generation/prompt_templates.py`:

```python
"""All LLM prompt templates — never inline strings."""
from __future__ import annotations

from backend.retrieval.vector_store import SearchResult

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer only from the provided excerpts.\n"
    "Cite every claim immediately after the sentence using [filename, p.N].\n"
    "If the excerpts do not contain enough information, respond with exactly:\n"
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/generation/test_prompt_templates.py -v
```
Expected: 10 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add backend/generation/prompt_templates.py tests/generation/__init__.py tests/generation/test_prompt_templates.py
git commit -m "feat: implement prompt_templates — SYSTEM_PROMPT and build_user_message"
```

---

## Task 2: llm_client skeleton + _generate_anthropic

**Files:**
- Create: `tests/generation/test_llm_client.py`
- Modify: `backend/generation/llm_client.py`

- [ ] **Step 1: Write failing tests**

Create `tests/generation/test_llm_client.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/generation/test_llm_client.py -v
```
Expected: `ImportError: cannot import name 'ANTHROPIC_MODEL'`

- [ ] **Step 3: Implement skeleton + _generate_anthropic**

Replace `backend/generation/llm_client.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/generation/test_llm_client.py -v
```
Expected: 6 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/generation/llm_client.py tests/generation/test_llm_client.py
git commit -m "feat: add llm_client skeleton with _generate_anthropic"
```

---

## Task 3: _generate_ollama

**Files:**
- Modify: `backend/generation/llm_client.py`
- Modify: `tests/generation/test_llm_client.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/generation/test_llm_client.py`:

```python
from backend.generation.llm_client import _generate_ollama


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/generation/test_llm_client.py -k "ollama" -v
```
Expected: `ImportError: cannot import name '_generate_ollama'`

- [ ] **Step 3: Append _generate_ollama to llm_client.py**

Append after `_generate_anthropic`:

```python
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
```

- [ ] **Step 4: Run full test suite for llm_client**

```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/generation/test_llm_client.py -v
```
Expected: 13 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/generation/llm_client.py tests/generation/test_llm_client.py
git commit -m "feat: implement _generate_ollama — HTTP streaming from Ollama"
```

---

## Task 4: generate orchestrator

**Files:**
- Modify: `backend/generation/llm_client.py`
- Modify: `tests/generation/test_llm_client.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/generation/test_llm_client.py`:

```python
from backend.generation.llm_client import generate


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
    from backend.retrieval.vector_store import SearchResult
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/generation/test_llm_client.py -k "generate" -v
```
Expected: `ImportError: cannot import name 'generate'`

- [ ] **Step 3: Append generate to llm_client.py**

Append after `_generate_ollama`:

```python
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
```

- [ ] **Step 4: Run full generation test suite**

```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/generation/ -v
```
Expected: all PASSED (≥ 25 tests across both files).

- [ ] **Step 5: Run combined suite to verify no regressions**

```bash
cd /Users/sadiya/projects/DocuMitra && pytest tests/ -q --tb=short 2>&1 | tail -5
```
Expected: all tests pass (128 existing + generation tests).

- [ ] **Step 6: Commit**

```bash
git add backend/generation/llm_client.py tests/generation/test_llm_client.py
git commit -m "feat: implement generate — Anthropic streaming with Ollama fallback"
```

---

## Self-Review

**Spec coverage:**
- ✅ `SYSTEM_PROMPT` — Task 1
- ✅ `build_user_message(query, results)` with `[filename, p.N]` format — Task 1
- ✅ "not enough information" instruction in system prompt — Task 1
- ✅ `ANTHROPIC_MODEL="claude-sonnet-4-6"`, `OLLAMA_MODEL="gemma4:4b"`, `MAX_TOKENS=1024` — Task 2
- ✅ `_generate_anthropic` streams via `client.messages.stream()` — Task 2
- ✅ `ANTHROPIC_API_KEY` from `os.environ` at call time — Task 2
- ✅ `_generate_ollama` POSTs to `{OLLAMA_URL}/api/chat`, streams newline-delimited JSON — Task 3
- ✅ `OLLAMA_URL` default `http://localhost:11434` — Task 3
- ✅ `generate(query, results)` tries Anthropic, catches `anthropic.APIError`, falls back to Ollama — Task 4
- ✅ `log.warning` on fallback — Task 4
- ✅ `ValueError` and non-APIError exceptions are NOT caught (propagate) — Task 4

**Type consistency:**
- `generate(query: str, results: list[SearchResult]) -> Iterator[str]` ✅
- `_generate_anthropic(prompt: str, system: str) -> Iterator[str]` ✅
- `_generate_ollama(prompt: str, system: str) -> Iterator[str]` ✅
- `build_user_message(query: str, results: list[SearchResult]) -> str` ✅

**No placeholders found.**
