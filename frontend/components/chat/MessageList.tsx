'use client'
import ReactMarkdown from 'react-markdown'
import type { Message } from '@/hooks/useStreamingChat'

interface Props {
  messages: Message[]
  isStreaming: boolean
  bottomRef: React.RefObject<HTMLDivElement | null>
}

export default function MessageList({ messages, isStreaming, bottomRef }: Props) {
  return (
    <div className="flex flex-col gap-4 p-4 pb-2">
      {messages.length === 0 && (
        <div className="flex flex-col items-center justify-center h-48 gap-2 text-gray-400">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <p className="text-sm">Ask a question about your documents</p>
        </div>
      )}
      {messages.map((msg, i) => {
        const isLast = i === messages.length - 1
        const showCursor = isStreaming && isLast && msg.role === 'assistant'

        if (msg.role === 'user') {
          return (
            <div key={msg.id} className="flex justify-end">
              <div className="max-w-xl rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm bg-blue-600 text-white shadow-sm">
                {msg.content}
              </div>
            </div>
          )
        }

        return (
          <div key={msg.id} className="flex justify-start">
            <div className="max-w-2xl rounded-2xl rounded-tl-sm px-4 py-3 text-sm bg-white border border-gray-200 text-gray-900 shadow-sm">
              <div className="markdown-body">
                <ReactMarkdown
                  components={{
                    h1: ({ children }) => <h1 className="text-base font-bold mt-2 mb-1 first:mt-0">{children}</h1>,
                    h2: ({ children }) => <h2 className="text-sm font-bold mt-2 mb-1 first:mt-0">{children}</h2>,
                    h3: ({ children }) => <h3 className="text-sm font-semibold mt-1.5 mb-0.5 first:mt-0">{children}</h3>,
                    h4: ({ children }) => <h4 className="text-sm font-medium mt-1 mb-0.5 first:mt-0">{children}</h4>,
                    p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
                    ul: ({ children }) => <ul className="list-disc list-outside pl-4 mb-2 space-y-0.5">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal list-outside pl-4 mb-2 space-y-0.5">{children}</ol>,
                    li: ({ children }) => <li className="text-sm">{children}</li>,
                    strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                    em: ({ children }) => <em className="italic">{children}</em>,
                    code: ({ children, className }) => {
                      const isBlock = className?.includes('language-')
                      return isBlock
                        ? <code className="block bg-gray-50 border border-gray-200 rounded-md px-3 py-2 text-xs font-mono my-2 overflow-x-auto whitespace-pre">{children}</code>
                        : <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs font-mono text-gray-800">{children}</code>
                    },
                    pre: ({ children }) => <>{children}</>,
                    hr: () => <hr className="my-3 border-gray-200" />,
                    blockquote: ({ children }) => <blockquote className="border-l-2 border-blue-300 pl-3 text-gray-600 italic my-2">{children}</blockquote>,
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              </div>
              {showCursor && <span className="animate-pulse text-gray-400 ml-0.5">▍</span>}
            </div>
          </div>
        )
      })}
      <div ref={bottomRef} className="h-1" />
    </div>
  )
}
