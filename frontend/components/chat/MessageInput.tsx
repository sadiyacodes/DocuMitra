'use client'
import { type KeyboardEvent, useEffect, useRef } from 'react'

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
      // reset height after clearing
      if (ref.current) ref.current.style.height = 'auto'
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  // auto-grow as user types
  function handleInput() {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }

  useEffect(() => {
    ref.current?.focus()
  }, [])

  return (
    <div className="border-t border-gray-200 px-4 py-3 bg-white">
      <div className="flex items-end gap-2">
        <textarea
          ref={ref}
          rows={1}
          disabled={disabled}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
          className="flex-1 resize-none rounded-xl border border-gray-300 px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 leading-relaxed overflow-hidden"
          style={{ minHeight: '42px', maxHeight: '160px' }}
        />
        <button
          disabled={disabled}
          onClick={submit}
          className="shrink-0 px-4 py-2.5 bg-blue-600 text-white rounded-xl text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {disabled ? (
            <span className="flex items-center gap-1.5">
              <span className="flex gap-0.5">
                <span className="w-1 h-1 bg-white rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1 h-1 bg-white rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1 h-1 bg-white rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </span>
            </span>
          ) : 'Send'}
        </button>
      </div>
      <p className="text-xs text-gray-400 mt-1.5 pl-1">Shift+Enter for newline</p>
    </div>
  )
}
