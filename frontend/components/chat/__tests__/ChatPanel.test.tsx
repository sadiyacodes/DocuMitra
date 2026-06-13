import { render, screen } from '@testing-library/react'
import ChatPanel from '../ChatPanel'

jest.mock('@/hooks/useStreamingChat', () => ({
  useStreamingChat: () => ({
    messages: [],
    isStreaming: false,
    sendMessage: jest.fn(),
    clearHistory: jest.fn(),
  }),
}))

test('renders Clear button', () => {
  render(<ChatPanel />)
  expect(screen.getByText('Clear')).toBeInTheDocument()
})

test('renders Send button', () => {
  render(<ChatPanel />)
  expect(screen.getByRole('button', { name: 'Send' })).toBeInTheDocument()
})
