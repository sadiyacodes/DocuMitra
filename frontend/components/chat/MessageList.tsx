'use client'
import type { Message } from '@/hooks/useStreamingChat'

interface Props {
  messages: Message[]
  isStreaming: boolean
}

export default function MessageList({ messages, isStreaming }: Props) {
  return (
    <div className="flex flex-col gap-3 p-4">
      {messages.map((msg, i) => {
        const isLast = i === messages.length - 1
        const showCursor = isStreaming && isLast && msg.role === 'assistant'
        return (
          <div
            key={msg.id}
            className={`max-w-2xl rounded-lg px-4 py-2 text-sm whitespace-pre-wrap ${
              msg.role === 'user'
                ? 'self-end bg-blue-600 text-white'
                : 'self-start bg-white border border-gray-200 text-gray-900'
            }`}
          >
            {msg.content}
            {showCursor && <span className="animate-pulse ml-0.5">▍</span>}
          </div>
        )
      })}
    </div>
  )
}
