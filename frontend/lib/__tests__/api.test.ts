import { ApiError, getChunks, ingestFile, streamQuery } from '../api'

function makeSSEStream(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) {
        controller.enqueue(enc.encode(`data: ${JSON.stringify(c)}\n\n`))
      }
      controller.enqueue(enc.encode('data: [DONE]\n\n'))
      controller.close()
    },
  })
}

function mockFetch(body: unknown, status = 200) {
  global.fetch = jest.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    body: body instanceof ReadableStream ? body : null,
    json: () => Promise.resolve(body),
  } as unknown as Response)
}

test('streamQuery yields text chunks', async () => {
  mockFetch(makeSSEStream(['Hello ', 'world']))
  const chunks: string[] = []
  for await (const c of streamQuery('test')) chunks.push(c)
  expect(chunks).toEqual(['Hello ', 'world'])
})

test('streamQuery stops at [DONE]', async () => {
  mockFetch(makeSSEStream(['chunk']))
  const chunks: string[] = []
  for await (const c of streamQuery('test')) chunks.push(c)
  expect(chunks).toEqual(['chunk'])
})

test('streamQuery throws ApiError on non-ok response', async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 500,
    json: () => Promise.resolve({ detail: 'server error' }),
  } as unknown as Response)
  await expect(async () => {
    for await (const _ of streamQuery('test')) { /* noop */ }
  }).rejects.toThrow(ApiError)
})

test('ingestFile returns parsed response', async () => {
  mockFetch({ pdf_id: 'abc', filename: 'doc.pdf', chunks_added: 5 })
  const result = await ingestFile(new File([''], 'doc.pdf', { type: 'application/pdf' }))
  expect(result).toEqual({ pdf_id: 'abc', filename: 'doc.pdf', chunks_added: 5 })
})

test('ingestFile throws ApiError on 422', async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 422,
    json: () => Promise.resolve({ detail: "Failed to extract 'bad.pdf': corrupt" }),
  } as unknown as Response)
  await expect(ingestFile(new File([''], 'bad.pdf'))).rejects.toThrow(ApiError)
})

test('getChunks returns results array', async () => {
  mockFetch({ results: [{ chunk_id: 'c1', filename: 'a.pdf', page_number: 1 }] })
  const result = await getChunks('test', 5)
  expect(result.results).toHaveLength(1)
  expect(result.results[0].chunk_id).toBe('c1')
})

test('getChunks throws ApiError on error', async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false, status: 503,
    json: () => Promise.resolve({ detail: 'unavailable' }),
  } as unknown as Response)
  await expect(getChunks('test', 5)).rejects.toThrow(ApiError)
})
