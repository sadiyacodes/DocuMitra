'use client'
import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import type { Message } from '@/hooks/useStreamingChat'

const SUGGESTED_QUESTIONS = [
  'What are the main topics covered in this document?',
  'Summarise the key points from the first chapter',
  'What examples or case studies are mentioned?',
  'Are there any definitions or glossary terms?',
]

interface Props {
  messages: Message[]
  isStreaming: boolean
  bottomRef: React.RefObject<HTMLDivElement | null>
  onSuggest: (q: string) => void
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  function copy() {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }
  return (
    <button
      onClick={copy}
      className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-md hover:bg-gray-100 text-gray-400 hover:text-gray-600"
      title="Copy response"
    >
      {copied ? (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="20,6 9,17 4,12" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )}
    </button>
  )
}

function ThinkingBubble() {
  return (
    <div className="flex justify-start">
      <div className="rounded-2xl rounded-tl-sm px-4 py-3 bg-white border border-gray-200 shadow-sm">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </div>
  )
}

export default function MessageList({ messages, isStreaming, bottomRef, onSuggest }: Props) {
  // detect if the last assistant message is still empty (thinking phase)
  const lastMsg = messages[messages.length - 1]
  const isThinking = isStreaming && lastMsg?.role === 'assistant' && lastMsg.content === ''

  return (
    <div className="flex flex-col gap-4 p-4 pb-2">
      {messages.length === 0 && (
        <div className="flex flex-col items-center gap-6 pt-12">
          <div className="flex flex-col items-center gap-2 text-gray-400">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <p className="text-sm font-medium">Ask a question about your documents</p>
            <p className="text-xs">Or try one of these to get started</p>
          </div>
          <div className="grid grid-cols-2 gap-2 w-full max-w-xl">
            {SUGGESTED_QUESTIONS.map(q => (
              <button
                key={q}
                onClick={() => onSuggest(q)}
                className="text-left px-3 py-2.5 rounded-xl border border-gray-200 bg-white hover:border-blue-300 hover:bg-blue-50 text-xs text-gray-600 hover:text-blue-700 transition-colors shadow-sm"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {messages.map((msg, i) => {
        const isLast = i === messages.length - 1
        const showCursor = isStreaming && isLast && msg.role === 'assistant' && msg.content !== ''

        if (msg.role === 'user') {
          return (
            <div key={msg.id} className="flex justify-end">
              <div className="max-w-xl rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm bg-blue-600 text-white shadow-sm">
                {msg.content}
              </div>
            </div>
          )
        }

        // hide the empty assistant placeholder while thinking — ThinkingBubble renders instead
        if (msg.content === '' && isThinking && isLast) return null

        return (
          <div key={msg.id} className="flex justify-start">
            <div className="relative group max-w-2xl rounded-2xl rounded-tl-sm px-4 py-3 text-sm bg-white border border-gray-200 text-gray-900 shadow-sm">
              {!isStreaming && <CopyButton text={msg.content} />}
              <div className="markdown-body pr-6">
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

      {isThinking && <ThinkingBubble />}

      <div ref={bottomRef} className="h-1" />
    </div>
  )
}
