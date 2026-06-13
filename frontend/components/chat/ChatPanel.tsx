'use client'
import { useStreamingChat } from '@/hooks/useStreamingChat'
import MessageInput from './MessageInput'
import MessageList from './MessageList'

export default function ChatPanel() {
  const { messages, isStreaming, sendMessage, clearHistory } = useStreamingChat()

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-white">
        <h1 className="text-base font-semibold text-gray-900">Chat</h1>
        <button
          onClick={clearHistory}
          className="text-xs text-gray-500 hover:text-gray-700"
        >
          Clear
        </button>
      </div>
      <div className="flex-1 overflow-auto flex flex-col">
        <MessageList messages={messages} isStreaming={isStreaming} />
      </div>
      <MessageInput onSend={sendMessage} disabled={isStreaming} />
    </div>
  )
}
