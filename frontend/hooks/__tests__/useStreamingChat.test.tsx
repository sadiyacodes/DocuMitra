import { act, renderHook } from '@testing-library/react'
import { useStreamingChat } from '../useStreamingChat'

jest.mock('@/lib/api', () => ({
  streamQuery: jest.fn(),
}))

jest.mock('@/lib/auth', () => ({
  getToken: jest.fn(() => null),
}))

import { streamQuery } from '@/lib/api'
const mockStreamQuery = streamQuery as jest.MockedFunction<typeof streamQuery>

function makeAsyncGen(...chunks: string[]) {
  return async function* () {
    for (const c of chunks) yield { type: 'text' as const, content: c }
  }
}

beforeEach(() => {
  localStorage.clear()
  jest.clearAllMocks()
})

test('starts with empty messages', () => {
  const { result } = renderHook(() => useStreamingChat())
  expect(result.current.messages).toEqual([])
})

test('loads messages from localStorage on mount', () => {
  const stored = [{ id: '1', role: 'user', content: 'hi', timestamp: 1 }]
  localStorage.setItem('documitra_chat_history', JSON.stringify(stored))
  const { result } = renderHook(() => useStreamingChat())
  expect(result.current.messages).toHaveLength(1)
  expect(result.current.messages[0].content).toBe('hi')
})

test('saves messages to localStorage when they change', async () => {
  mockStreamQuery.mockImplementation(makeAsyncGen('answer') as never)
  const { result } = renderHook(() => useStreamingChat())
  await act(async () => { await result.current.sendMessage('hello') })
  const stored = JSON.parse(localStorage.getItem('documitra_chat_history')!)
  expect(stored.length).toBe(2)
})

test('sendMessage appends user then assistant message', async () => {
  mockStreamQuery.mockImplementation(makeAsyncGen('resp') as never)
  const { result } = renderHook(() => useStreamingChat())
  await act(async () => { await result.current.sendMessage('hello') })
  expect(result.current.messages[0].role).toBe('user')
  expect(result.current.messages[0].content).toBe('hello')
  expect(result.current.messages[1].role).toBe('assistant')
})

test('sendMessage accumulates chunks into assistant message', async () => {
  mockStreamQuery.mockImplementation(makeAsyncGen('Hello ', 'world') as never)
  const { result } = renderHook(() => useStreamingChat())
  await act(async () => { await result.current.sendMessage('q') })
  expect(result.current.messages[1].content).toBe('Hello world')
})

test('isStreaming is false after sendMessage completes', async () => {
  mockStreamQuery.mockImplementation(makeAsyncGen('done') as never)
  const { result } = renderHook(() => useStreamingChat())
  await act(async () => { await result.current.sendMessage('q') })
  expect(result.current.isStreaming).toBe(false)
})

test('clearHistory empties messages and localStorage', async () => {
  mockStreamQuery.mockImplementation(makeAsyncGen('x') as never)
  const { result } = renderHook(() => useStreamingChat())
  await act(async () => { await result.current.sendMessage('q') })
  act(() => { result.current.clearHistory() })
  expect(result.current.messages).toEqual([])
  expect(localStorage.getItem('documitra_chat_history')).toBeNull()
})
