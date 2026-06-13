# DocuMitra — Project Context

## What this is
A RAG chatbot that answers questions from a private corpus of 10+ PDFs (200+ pages each), using open-source embeddings and a free vector DB. Target: 2–5s end-to-end query latency, with source citations (filename + page number) on every answer.

## Stack (locked)
- **Frontend**: Next.js 14 + TypeScript
- **Backend**: Python FastAPI
- **Vector DB**: Supabase (Postgres + pgvector), HNSW index
- **Embedding model**: BAAI/bge-small-en-v1.5 (sentence-transformers, CPU-friendly)
- **Reranker** (optional): cross-encoder/ms-marco-MiniLM-L-6-v2 — toggle off if it breaks the latency budget
- **LLM (generation)**: Anthropic API (`claude-sonnet-4-6`) primary, Ollama `gemma4:e4b` as automatic fallback — see `backend/generation/llm_client.py`
- **PDF extraction**: PyMuPDF for native text, Tesseract for OCR on scanned pages

## Chunking rules
- 500–1000 tokens per chunk, 10–30% overlap
- Metadata per chunk: `pdf_id`, `filename`, `page_number`, `bbox` (if available)
- Strip repeated headers/footers before chunking (detect lines repeating across pages)
- Detect language per chunk
- Normalize text before chunking: fix unicode/whitespace issues and common OCR artifacts
- Chunking must be deterministic — same params + same input always produce the same chunks, for reproducible embeddings
- Run OCR on embedded images/diagrams within text pages (not just fully-scanned pages), to capture labeled diagram text — extract embedded images via PyMuPDF, OCR each with Tesseract, append result to that page's text

## Repo structure
```
DocuMitra/
├── frontend/                   # Next.js 14 + TS
│   ├── app/
│   │   ├── chat/                # main Q&A interface
│   │   ├── ingestion/            # pipeline status view
│   │   └── retrieval/            # top-k chunks visualization
│   └── components/
├── backend/                     # FastAPI
│   ├── ingestion/
│   │   ├── extract.py            # PyMuPDF + Tesseract OCR (scanned pages + embedded images)
│   │   ├── chunker.py            # 500-1000 tok, overlap
│   │   └── embed.py              # bge-small-en-v1.5
│   ├── retrieval/
│   │   ├── vector_store.py       # pgvector (Supabase)
│   │   └── reranker.py           # cross-encoder, optional
│   ├── generation/
│   │   ├── llm_client.py         # Anthropic primary, Ollama fallback
│   │   └── prompt_templates.py
│   ├── eval/
│   │   └── eval_runner.py        # p95 latency, R@k, MRR
│   └── main.py
├── data/
│   ├── pdfs/
│   └── cache/                    # precomputed embeddings
├── scripts/
│   ├── ingest_all.py
│   └── cache_demo.py
└── README.md
```

## Environment variables
- `ANTHROPIC_API_KEY` — required
- `OLLAMA_URL` — default `http://localhost:11434`
- `DEMO_MODE` — true/false; when true, use pre-cached responses from `scripts/cache_demo.py` for demo reliability

## Build order
1. `backend/ingestion/extract.py` — PDF → text (native + OCR)
2. `backend/ingestion/chunker.py` — text → chunks with metadata
3. `backend/ingestion/embed.py` — chunks → embeddings → pgvector
4. `backend/retrieval/vector_store.py` — top-k retrieval (HNSW), returns chunks with metadata + similarity scores
5. `backend/retrieval/reranker.py` — optional rerank
6. `backend/generation/llm_client.py` + `prompt_templates.py` — answer synthesis with citations
7. `backend/main.py` — FastAPI endpoints wiring it all together
8. `frontend/` — chat UI, ingestion status, retrieval visualization
9. `backend/eval/eval_runner.py` — latency (p95), R@k, MRR, citation accuracy, hallucination rate

## Conventions
- Python: type hints on all functions, docstrings for non-trivial logic
- All LLM prompts live in `prompt_templates.py`, never inline strings
- Every generated answer must include citations formatted as `[filename, p.N]`
- If no retrieved chunk clears the relevance threshold, the answer must say there isn't enough information rather than guessing — this is the primary hallucination-rate control, and `eval_runner.py` should test it directly
- Precompute and cache embeddings — never re-embed on every run

## Latency budget (2–5s total)
- Embedding the query: <100ms (model kept warm in memory)
- Retrieval (HNSW top-20): <200ms
- Reranking (optional, top-20 → top-5): ~500ms — skip if over budget
- LLM generation: 1.5–3s (stream response to reduce perceived latency)

## Constraints
- Embedding model and vector DB must be free/open-source — no paid services for these
- Must support ingestion of 10+ PDFs, 200+ pages each
