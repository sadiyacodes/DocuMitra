# llm_client.py + prompt_templates.py Design Spec
**Date:** 2026-06-13
**Scope:** `backend/generation/llm_client.py` + `backend/generation/prompt_templates.py` — RAG answer generation with Anthropic primary, Ollama fallback, streaming output, and citation formatting

---

## Context

DocuMitra generation stage. Takes a query string and `list[SearchResult]` from the reranker, builds a prompt with citations context, streams the answer from `claude-sonnet-4-6` (Anthropic) with automatic fallback to `gemma4:4b` (Ollama) on any API error. Every answer must include `[filename, p.N]` citations or state there is not enough information.

---

## Decisions

- **Streaming:** Synchronous `Iterator[str]` — yields text chunks. FastAPI wraps in `StreamingResponse`. Easy to test by collecting chunks.
- **Structure:** Separate `_generate_anthropic` and `_generate_ollama` private functions; `generate()` orchestrates. Each provider path testable in isolation.
- **Prompts:** All strings in `prompt_templates.py`. `llm_client.py` imports from there — never defines inline strings.
- **DEMO_MODE:** Handled upstream in `main.py`, not in `llm_client.py`.

---

## prompt_templates.py

### Exports

```python
SYSTEM_PROMPT: str

def build_user_message(query: str, results: list[SearchResult]) -> str: ...
```

### SYSTEM_PROMPT

```
You are a document Q&A assistant. Answer only from the provided excerpts.
Cite every claim immediately after the sentence using [filename, p.N].
If the excerpts do not contain enough information, respond with exactly:
"I don't have enough information in the provided documents to answer this question."
Do not guess or add information not present in the excerpts.
```

### build_user_message

Formats each result as `[{filename}, p.{page_number}]\n{text}`, joins with `\n\n`, appends `\nQuestion: {query}`. When `results` is empty, the model sees only the question and follows the system instruction to return the no-answer response.

---

## llm_client.py

### Constants

```python
ANTHROPIC_MODEL = "claude-sonnet-4-6"
OLLAMA_MODEL    = "gemma4:4b"
MAX_TOKENS      = 1024
```

`OLLAMA_URL` and `ANTHROPIC_API_KEY` read from environment at call time (not import time) so tests can set them without patching module globals.

### Public Interface

```python
from backend.generation.llm_client import generate
```

Only `generate` is public. `_generate_anthropic` and `_generate_ollama` are private.

**Entry point:**
```python
def generate(
    query: str,
    results: list[SearchResult],
) -> Iterator[str]:
    """Stream an answer with citations; falls back to Ollama if Anthropic fails."""
```

---

## Pipeline

### generate

1. Build prompt: `system = SYSTEM_PROMPT`, `prompt = build_user_message(query, results)`
2. Try `yield from _generate_anthropic(prompt, system)`
3. On `anthropic.APIError`: `log.warning(...)`, then `yield from _generate_ollama(prompt, system)`

### _generate_anthropic(prompt, system) -> Iterator[str]

```python
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

### _generate_ollama(prompt, system) -> Iterator[str]

```python
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

---

## Error Handling

| Situation | Behavior |
|-----------|----------|
| `ANTHROPIC_API_KEY` missing | `KeyError` propagates — caller must configure env |
| Anthropic 5xx / overloaded | `anthropic.APIStatusError` → caught, Ollama fallback |
| Anthropic 401 auth failure | `anthropic.AuthenticationError` → caught, Ollama fallback |
| Anthropic connection error | `anthropic.APIConnectionError` → caught, Ollama fallback |
| Ollama unavailable | `requests.RequestException` propagates |
| Ollama 4xx/5xx | `requests.HTTPError` from `raise_for_status()` propagates |
| `results` empty | Valid prompt built; model returns "not enough information" |

---

## Dependencies

```
anthropic>=0.25.0    # streaming via client.messages.stream()
requests>=2.31.0     # Ollama HTTP streaming
```

Both already in requirements.txt.

---

## Out of Scope

- DEMO_MODE / pre-cached responses (→ `main.py`)
- Prompt caching or token counting
- Retry logic on Ollama failure
- `main.py` FastAPI wiring
