'use client'
import { type KeyboardEvent, useRef } from 'react'

interface Props {
  onSend: (value: string) => void
  disabled: boolean
}

export default function MessageInput({ onSend, disabled }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null)

  function submit() {
    const value = ref.current?.value.trim()
    if (value) {
      onSend(value)
      ref.current!.value = ''
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="border-t border-gray-200 p-4 bg-white">
      <div className="flex gap-2">
        <textarea
          ref={ref}
          rows={2}
          disabled={disabled}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
          className="flex-1 resize-none rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        />
        <button
          disabled={disabled}
          onClick={submit}
          className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  )
}
