const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

export interface SearchResult {
  chunk_id: string
  pdf_id: string
  filename: string
  page_number: number
  text: string
  token_count: number
  language: string
  bbox: number[] | null
  similarity: number
}

export interface IngestResponse {
  pdf_id: string
  filename: string
  chunks_added: number
}

export interface ChunksResponse {
  results: SearchResult[]
}

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

export async function* streamQuery(
  query: string,
  rerank = true,
): AsyncGenerator<string> {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
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
      if (part.startsWith('data: ')) {
        const data = part.slice(6)
        if (data === '[DONE]') return
        try {
          yield JSON.parse(data) as string
        } catch {}
      }
    }
  }
}

export async function ingestFile(file: File): Promise<IngestResponse> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/ingest`, { method: 'POST', body: form })
  await assertOk(res)
  return res.json()
}

export async function getChunks(query: string, k: number): Promise<ChunksResponse> {
  const params = new URLSearchParams({ query, k: String(k) })
  const res = await fetch(`${BASE}/chunks?${params}`)
  await assertOk(res)
  return res.json()
}
