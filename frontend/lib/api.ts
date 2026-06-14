const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

export interface SearchResult {
  chunk_id: string
  source_id: string
  source_type: string
  filename: string
  page_number: number
  text: string
  token_count: number
  language: string
  bbox: number[] | null
  access_roles: string[]
  similarity: number
}

export interface SourceResult {
  filename: string
  page: number
  similarity: number
  source_type: string
}

export interface IngestResponse {
  source_id: string
  filename: string
  chunks_added: number
}

export interface ChunksResponse {
  results: SearchResult[]
}

export type StreamEvent =
  | { type: 'text'; content: string }
  | { type: 'sources'; sources: SourceResult[] }

async function assertOk(res: Response): Promise<void> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body.detail) detail = String(body.detail)
    } catch {}
    throw new ApiError(res.status, detail)
  }
}

function authHeader(token?: string | null): HeadersInit {
  const h: HeadersInit = { 'Content-Type': 'application/json' }
  if (token) h['Authorization'] = `Bearer ${token}`
  return h
}

export async function* streamQuery(
  query: string,
  rerank = true,
  token?: string | null,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: authHeader(token),
    body: JSON.stringify({ query, rerank }),
  })
  await assertOk(res)

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() ?? ''
    for (const part of parts) {
      const lines = part.split('\n')
      let eventType = 'message'
      let data = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) eventType = line.slice(7).trim()
        else if (line.startsWith('data: ')) data = line.slice(6)
      }
      if (eventType === 'sources') {
        try { yield { type: 'sources', sources: JSON.parse(data) as SourceResult[] } } catch {}
      } else if (data === '[DONE]') {
        return
      } else if (data) {
        try { yield { type: 'text', content: JSON.parse(data) as string } } catch {}
      }
    }
  }
}

export async function ingestFile(file: File, roles: string[] = [], token?: string | null): Promise<IngestResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('roles', roles.join(','))
  const headers: HeadersInit = {}
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${BASE}/ingest`, { method: 'POST', headers, body: form })
  await assertOk(res)
  return res.json()
}

export async function getChunks(query: string, k: number, token?: string | null): Promise<ChunksResponse> {
  const params = new URLSearchParams({ query, k: String(k) })
  const headers: HeadersInit = {}
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${BASE}/chunks?${params}`, { headers })
  await assertOk(res)
  return res.json()
}
