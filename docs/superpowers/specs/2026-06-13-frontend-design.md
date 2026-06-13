# DocuMitra Frontend Design

## Goal

A Next.js 14 + TypeScript frontend that lets users chat with a private PDF corpus, upload new PDFs, and inspect retrieval results. Three pages wired to the existing FastAPI backend.

## Architecture

App Router with a persistent sidebar shell. Each route imports one focused client component. Fetch calls are centralized in `lib/api.ts`. The frontend calls FastAPI directly using `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`). CORS middleware is added to `backend/main.py` as the only backend change.

## Stack

- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- `@testing-library/react` + Jest for unit/component tests

---

## File Structure

```
frontend/
├── app/
│   ├── layout.tsx              # Shell: sidebar + content slot
│   ├── page.tsx                # Redirects to /chat
│   ├── chat/page.tsx           # Imports ChatPanel
│   ├── ingestion/page.tsx      # Imports IngestPanel
│   └── retrieval/page.tsx      # Imports ChunksPanel
├── components/
│   ├── Sidebar.tsx             # 'use client' — nav links + usePathname active highlight
│   ├── chat/
│   │   ├── ChatPanel.tsx       # 'use client' — owns useStreamingChat, renders list + input
│   │   ├── MessageList.tsx     # Renders message bubbles (user right, assistant left)
│   │   └── MessageInput.tsx    # Textarea (Enter=send, Shift+Enter=newline) + send button
│   ├── ingestion/
│   │   └── IngestPanel.tsx     # 'use client' — file upload state machine
│   └── retrieval/
│       └── ChunksPanel.tsx     # 'use client' — query input + results cards
├── hooks/
│   ├── useStreamingChat.ts     # SSE streaming + localStorage history
│   └── useChunks.ts            # GET /chunks fetcher + loading/error state
├── lib/
│   └── api.ts                  # All fetch calls; NEXT_PUBLIC_API_URL; ApiError type
├── .env.local.example          # NEXT_PUBLIC_API_URL=http://localhost:8000
├── next.config.ts
├── tailwind.config.ts
├── tsconfig.json
└── package.json
```

---

## Pages

### /chat — Chat Page

**Component:** `ChatPanel` (owns all state via `useStreamingChat`)

**useStreamingChat hook:**
- On mount: reads `documitra_chat_history` from localStorage and populates `messages: Message[]`
- On every `messages` change: writes array back to localStorage
- `sendMessage(query)`: appends user message → appends empty assistant placeholder → streams response → updates placeholder chunk by chunk
- `clearHistory()`: empties `messages`, removes localStorage key
- `isStreaming: boolean`: true while a response is in-flight

**Message type:**
```typescript
interface Message {
  id: string          // crypto.randomUUID()
  role: 'user' | 'assistant'
  content: string
  timestamp: number   // Date.now()
}
```

**UX:**
- User messages: right-aligned bubble
- Assistant messages: left-aligned bubble; streaming message shows blinking cursor
- `MessageInput` disabled while `isStreaming` is true
- "Clear" button in page header calls `clearHistory()`

---

### /ingestion — Ingestion Page

**Component:** `IngestPanel` (local state machine: `idle | uploading | done | error`)

| State | UI |
|-------|----|
| `idle` | PDF-only drag-and-drop zone + "Upload PDF" button |
| `uploading` | Spinner + filename, inputs disabled |
| `done` | Success card: "✓ `filename.pdf` — N chunks added" + "Upload another" button |
| `error` | Error message (from API `detail` or generic) + "Try again" button |

No persistence — stateless per visit.

---

### /retrieval — Retrieval Visualization Page

**Component:** `ChunksPanel` (local state via `useChunks`)

**useChunks hook:** wraps `getChunks(query, k)`, exposes `{ results, isLoading, error, search }`.

**UX:**
- Search input + "Search" button + `k` dropdown (options: 3, 5, 10, 20; default 5)
- While loading: skeleton cards
- Results: vertical list of cards, each showing:
  - Header: `filename` · `p.N` · similarity as percentage (e.g. `92%`)
  - Body: chunk text truncated to ~300 chars, "show more" toggle
  - Footer: language tag + token count
- Empty: "No chunks found for this query."
- Error: inline error message

No persistence — stateless per visit.

---

## API Layer (`lib/api.ts`)

```typescript
const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

class ApiError extends Error {
  constructor(public status: number, message: string) { super(message) }
}

// SSE streaming — yields each text chunk until [DONE]
export async function* streamQuery(query: string, rerank = true): AsyncGenerator<string>

// PDF upload — returns { pdf_id, filename, chunks_added }
export async function ingestFile(file: File): Promise<IngestResponse>

// Top-k retrieval — returns { results: SearchResult[] }
export async function getChunks(query: string, k: number): Promise<ChunksResponse>
```

**SSE parsing in `streamQuery`:** `fetch POST /query` → read `response.body` as `ReadableStream` → decode UTF-8 → split on `\n\n` → for each `data: <json>` line, `JSON.parse` the value and yield; stop on `data: [DONE]`.

**TypeScript types** mirroring the backend:
```typescript
interface SearchResult {
  chunk_id: string; pdf_id: string; filename: string; page_number: number
  text: string; token_count: number; language: string
  bbox: number[] | null; similarity: number
}
interface IngestResponse { pdf_id: string; filename: string; chunks_added: number }
interface ChunksResponse { results: SearchResult[] }
```

---

## Error Handling

- `ApiError` thrown by all `lib/api.ts` functions on non-2xx responses (status + parsed `detail` from FastAPI JSON body).
- Each client component catches errors locally and renders an inline error state.
- No global error boundary or toast library.

---

## Backend Change: CORS

Add to `backend/main.py` (after `app = FastAPI(...)`):

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Testing

- **`useStreamingChat`**: `renderHook` with mocked `fetch`; test localStorage read on mount, write on message change, clearHistory, streaming message accumulation.
- **`useChunks`**: `renderHook` with mocked `fetch`; test loading state, success, error.
- **`IngestPanel`**: render test for each state (idle → uploading → done, idle → error).
- **`ChunksPanel`**: render test — shows skeleton on load, renders result cards, shows empty state.
- **`MessageList`**: render test — user vs assistant alignment, streaming cursor.
- No E2E tests.
