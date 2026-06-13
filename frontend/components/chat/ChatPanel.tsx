'use client'
import { useEffect, useRef } from 'react'
import { useStreamingChat } from '@/hooks/useStreamingChat'
import MessageInput from './MessageInput'
import MessageList from './MessageList'

export default function ChatPanel() {
  const { messages, isStreaming, sendMessage, clearHistory } = useStreamingChat()
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-200 bg-white">
        <div>
          <h1 className="text-base font-semibold text-gray-900">Chat</h1>
          {messages.length > 0 && (
            <p className="text-xs text-gray-400">{messages.filter(m => m.role === 'user').length} questions</p>
          )}
        </div>
        {messages.length > 0 && (
          <button
            type="button"
            onClick={clearHistory}
            className="text-xs text-gray-400 hover:text-red-500 transition-colors px-2 py-1 rounded hover:bg-red-50"
          >
            Clear history
          </button>
        )}
      </div>
      <div className="flex-1 overflow-auto">
        <MessageList
          messages={messages}
          isStreaming={isStreaming}
          bottomRef={bottomRef}
          onSuggest={sendMessage}
        />
      </div>
      <MessageInput onSend={sendMessage} disabled={isStreaming} />
    </div>
  )
}
