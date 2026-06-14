# DocuMitra

Enterprise RAG (Retrieval-Augmented Generation) chatbot for private document corpora. Answers questions from PDFs, CSVs, and JSON files with source citations, role-based access control, and streaming responses.

## Features

- **Multi-format ingestion** — PDF (native text + OCR), CSV, JSON
- **Semantic search** — BAAI/bge-small-en-v1.5 embeddings, pgvector HNSW index
- **Streaming answers** — SSE with inline citations (`[filename, p.N]`) and source chips
- **RBAC** — per-document access roles, enforced at retrieval time
- **LLM** — Anthropic claude-sonnet-4-6 (primary), Ollama gemma4 (fallback)
- **Auth** — JWT login, role-aware UI (only admins can ingest)

---

## Architecture

```
Browser (Next.js 16)
  └─ FastAPI backend (Python)
       ├─ Ingestion: PyMuPDF + Tesseract → chunker → bge-small embeddings → Supabase pgvector
       ├─ Retrieval: HNSW top-k, optional cross-encoder rerank, RBAC filter
       └─ Generation: Anthropic API (SSE stream) with Ollama fallback
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 – 3.14 | |
| Node.js | 18+ | |
| Tesseract OCR | 5+ | For scanned PDFs |
| Supabase account | — | Free tier works |
| Anthropic API key | — | Required for generation |
| Ollama | optional | Fallback LLM |

---

## 1 — Install system dependencies

### macOS

```bash
# Homebrew (https://brew.sh)
brew install tesseract node python@3.12
```

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y tesseract-ocr python3 python3-pip python3-venv

# Node.js 20 via NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### Windows

1. **Python 3.12** — download from [python.org](https://www.python.org/downloads/). Check "Add Python to PATH" during install.
2. **Node.js 20** — download from [nodejs.org](https://nodejs.org/).
3. **Tesseract** — download the installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). After install, add the install folder (e.g. `C:\Program Files\Tesseract-OCR`) to your `PATH`.

---

## 2 — Clone and configure

```bash
git clone <repo-url>
cd EnterpriseRAG
```

Create `.env` in the project root:

```env
# Required
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=<your-supabase-service-role-key>

# Optional
OLLAMA_URL=http://localhost:11434
DEMO_MODE=false
```

Get your Supabase credentials from **Supabase dashboard → Settings → API**.

---

## 3 — Supabase database setup

Open the [Supabase SQL Editor](https://supabase.com/dashboard/project/_/sql) for your project and run these two scripts in order.

### Step 1 — Base schema

Paste and run the contents of `scripts/supabase_migration.sql`. This creates:
- `chunks` table with pgvector column
- HNSW index for cosine similarity search
- Initial `match_chunks` RPC function

### Step 2 — Enterprise schema

Paste and run the contents of `scripts/migration_enterprise.sql`. This adds:
- `source_id`, `source_type`, `access_roles` columns
- Updated `match_chunks` RPC with RBAC filtering

---

## 4 — Backend setup

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (Command Prompt)

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Windows note:** If `bcrypt` fails to install, run `pip install bcrypt --only-binary :all:`

---

## 5 — Create users

```bash
# macOS / Linux
python scripts/seed_users.py

# Windows
python scripts\seed_users.py
```

This writes `data/users.json` with four default accounts:

| Username | Password | Role |
|---|---|---|
| alice | admin123 | admin |
| bob | hr123 | hr |
| carol | finance123 | finance |
| dave | eng123 | engineering |

Edit `scripts/seed_users.py` to add or change accounts before running.

---

## 6 — Frontend setup

```bash
cd frontend
npm install
```

---

## 7 — Run the app

Open two terminals.

### Terminal 1 — Backend

```bash
# macOS / Linux
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Windows
.venv\Scripts\activate
uvicorn backend.main:app --reload --port 8000
```

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 8 — Ingest documents

Log in as `alice` (admin). Non-admin users cannot access the Ingest page.

### Via the UI

1. Go to **Ingest** in the sidebar.
2. Select the file type tab: **PDF**, **CSV**, or **JSON**.
3. Optionally fill in **Access roles** (comma-separated, e.g. `hr, admin`). Leave blank to allow all roles to query the document.
4. Drag & drop or click **Choose file**.

### Via bulk script (PDFs only)

```bash
# macOS / Linux
python scripts/ingest_all.py --dir /path/to/pdfs

# Windows
python scripts\ingest_all.py --dir C:\path\to\pdfs
```

### Via API

```bash
# PDF
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer <token>" \
  -F "file=@report.pdf" \
  -F "roles=admin,finance"

# CSV
curl -X POST http://localhost:8000/ingest/csv \
  -H "Authorization: Bearer <token>" \
  -F "file=@employees.csv" \
  -F "roles=hr,admin"

# JSON
curl -X POST http://localhost:8000/ingest/json \
  -H "Authorization: Bearer <token>" \
  -F "file=@audit_logs.json" \
  -F "roles=admin"
```

Get a token first:

```bash
curl -X POST http://localhost:8000/auth/login \
  -d "username=alice&password=admin123"
```

---

## 9 — Role-based access

| Role | Chat | Retrieval | Ingest |
|---|---|---|---|
| admin | ✅ | ✅ | ✅ |
| hr | ✅ | ✅ | ❌ |
| finance | ✅ | ✅ | ❌ |
| engineering | ✅ | ✅ | ❌ |

**Document-level RBAC:** When a document is ingested with `roles=hr,admin`, only users with the `hr` or `admin` role will see it in retrieval results. Documents ingested with no roles are visible to everyone.

---

## 10 — Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key for claude-sonnet-4-6 |
| `SUPABASE_URL` | Yes | — | Supabase project URL |
| `SUPABASE_KEY` | Yes | — | Supabase service role key |
| `OLLAMA_URL` | No | `http://localhost:11434` | Ollama endpoint for fallback LLM |
| `DEMO_MODE` | No | `false` | Use pre-cached responses (no live API calls) |

---

## 11 — Optional: Ollama fallback LLM

Install [Ollama](https://ollama.ai), then pull the model:

```bash
ollama pull gemma4:e4b
```

The backend automatically falls back to Ollama if the Anthropic API is unavailable or returns an error.

---

## 12 — Running tests

```bash
# macOS / Linux
source .venv/bin/activate
pytest

# Windows
.venv\Scripts\activate
pytest
```

---

## Latency targets

| Step | Target |
|---|---|
| Query embedding | < 100 ms |
| HNSW retrieval (top-20) | < 200 ms |
| Reranking (top-20 → top-5) | ~500 ms (optional) |
| LLM generation (streamed) | 1.5 – 3 s |
| **Total** | **2 – 5 s** |

---

## Project structure

```
EnterpriseRAG/
├── backend/
│   ├── auth/               # JWT login, RBAC dependencies
│   ├── ingestion/          # PDF/CSV/JSON extraction, chunking, embedding
│   ├── retrieval/          # pgvector search, reranker, query router
│   ├── generation/         # LLM client (Anthropic + Ollama), prompt templates
│   ├── eval/               # Latency, R@k, MRR evaluation
│   └── main.py             # FastAPI app
├── frontend/               # Next.js 16 + TypeScript
│   ├── app/                # Pages: /login, /chat, /ingestion, /retrieval
│   ├── components/         # Sidebar, ChatPanel, IngestPanel, ChunksPanel
│   └── hooks/              # useStreamingChat, useChunks, useAuthGuard
├── scripts/
│   ├── seed_users.py       # Generate data/users.json
│   ├── ingest_all.py       # Bulk PDF ingestion
│   ├── supabase_migration.sql      # Base schema
│   └── migration_enterprise.sql   # RBAC schema upgrade
├── data/
│   ├── pdfs/               # Place PDF files here for bulk ingestion
│   ├── users.json          # Generated by seed_users.py
│   └── cache/              # Pre-computed embedding cache
└── .env                    # Environment variables (not committed)
```
