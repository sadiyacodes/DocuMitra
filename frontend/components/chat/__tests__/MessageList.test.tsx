import { render, screen } from '@testing-library/react'
import MessageList from '../MessageList'
import type { Message } from '@/hooks/useStreamingChat'

const userMsg: Message = { id: '1', role: 'user', content: 'Hello', timestamp: 1 }
const assistantMsg: Message = { id: '2', role: 'assistant', content: 'Hi there', timestamp: 2 }

test('renders user message content', () => {
  render(<MessageList messages={[userMsg]} isStreaming={false} />)
  expect(screen.getByText('Hello')).toBeInTheDocument()
})

test('renders assistant message content', () => {
  render(<MessageList messages={[assistantMsg]} isStreaming={false} />)
  expect(screen.getByText('Hi there')).toBeInTheDocument()
})

test('shows streaming cursor on last assistant message when streaming', () => {
  render(<MessageList messages={[assistantMsg]} isStreaming={true} />)
  expect(screen.getByText('▍')).toBeInTheDocument()
})

test('does not show cursor when not streaming', () => {
  render(<MessageList messages={[assistantMsg]} isStreaming={false} />)
  expect(screen.queryByText('▍')).not.toBeInTheDocument()
})

test('does not show cursor on user message even when streaming', () => {
  render(<MessageList messages={[userMsg]} isStreaming={true} />)
  expect(screen.queryByText('▍')).not.toBeInTheDocument()
})
