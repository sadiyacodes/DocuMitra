import { fireEvent, render, screen } from '@testing-library/react'
import MessageInput from '../MessageInput'

test('calls onSend with trimmed value on Enter', () => {
  const onSend = jest.fn()
  render(<MessageInput onSend={onSend} disabled={false} />)
  const textarea = screen.getByRole('textbox')
  fireEvent.change(textarea, { target: { value: '  hello  ' } })
  fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false })
  expect(onSend).toHaveBeenCalledWith('hello')
})

test('does not call onSend on Shift+Enter', () => {
  const onSend = jest.fn()
  render(<MessageInput onSend={onSend} disabled={false} />)
  const textarea = screen.getByRole('textbox')
  fireEvent.change(textarea, { target: { value: 'hello' } })
  fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true })
  expect(onSend).not.toHaveBeenCalled()
})

test('textarea is disabled when disabled prop is true', () => {
  render(<MessageInput onSend={jest.fn()} disabled={true} />)
  expect(screen.getByRole('textbox')).toBeDisabled()
})

test('calls onSend on Send button click', () => {
  const onSend = jest.fn()
  render(<MessageInput onSend={onSend} disabled={false} />)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'hi' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  expect(onSend).toHaveBeenCalledWith('hi')
})
