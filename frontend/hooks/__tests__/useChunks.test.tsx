import { act, renderHook } from '@testing-library/react'
import { useChunks } from '../useChunks'

jest.mock('@/lib/api', () => ({
  getChunks: jest.fn(),
  ApiError: class ApiError extends Error {
    constructor(public status: number, message: string) { super(message) }
  },
}))

import { getChunks } from '@/lib/api'
const mockGetChunks = getChunks as jest.MockedFunction<typeof getChunks>

const fakeResult = {
  chunk_id: 'c1', pdf_id: 'p1', filename: 'doc.pdf', page_number: 1,
  text: 'text', token_count: 3, language: 'en', bbox: null, similarity: 0.9,
}

beforeEach(() => jest.clearAllMocks())

test('starts with empty results and not loading', () => {
  const { result } = renderHook(() => useChunks())
  expect(result.current.results).toEqual([])
  expect(result.current.isLoading).toBe(false)
  expect(result.current.error).toBeNull()
})

test('search sets isLoading true then false', async () => {
  let resolve!: (v: unknown) => void
  mockGetChunks.mockReturnValue(new Promise(r => { resolve = r }) as never)
  const { result } = renderHook(() => useChunks())
  act(() => { result.current.search('test', 5) })
  expect(result.current.isLoading).toBe(true)
  await act(async () => { resolve({ results: [] }) })
  expect(result.current.isLoading).toBe(false)
})

test('search populates results on success', async () => {
  mockGetChunks.mockResolvedValue({ results: [fakeResult] } as never)
  const { result } = renderHook(() => useChunks())
  await act(async () => { await result.current.search('test', 5) })
  expect(result.current.results).toHaveLength(1)
  expect(result.current.results[0].chunk_id).toBe('c1')
})

test('search sets error on failure', async () => {
  mockGetChunks.mockRejectedValue(new Error('network error'))
  const { result } = renderHook(() => useChunks())
  await act(async () => { await result.current.search('test', 5) })
  expect(result.current.error).toBe('network error')
  expect(result.current.results).toEqual([])
})

test('search clears previous results when called again', async () => {
  mockGetChunks.mockResolvedValueOnce({ results: [fakeResult] } as never)
  mockGetChunks.mockResolvedValueOnce({ results: [] } as never)
  const { result } = renderHook(() => useChunks())
  await act(async () => { await result.current.search('first', 5) })
  expect(result.current.results).toHaveLength(1)
  await act(async () => { await result.current.search('second', 5) })
  expect(result.current.results).toHaveLength(0)
})
