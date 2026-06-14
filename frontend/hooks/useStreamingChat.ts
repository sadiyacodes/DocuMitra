'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { streamQuery, type SourceResult } from '@/lib/api'
import { getToken } from '@/lib/auth'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceResult[]
  timestamp: number
}

const STORAGE_KEY = 'documitra_chat_history'

export function useStreamingChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const skipNextSave = useRef(false)
  const isStreamingRef = useRef(false)

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) setMessages(JSON.parse(stored))
    } catch {}
  }, [])

  useEffect(() => {
    if (skipNextSave.current) {
      skipNextSave.current = false
      return
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages))
  }, [messages])

  const sendMessage = useCallback(async (query: string) => {
    if (isStreamingRef.current) return

    isStreamingRef.current = true
    setIsStreaming(true)

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      timestamp: Date.now(),
    }
    const assistantId = crypto.randomUUID()
    const assistantMsg: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    }

    setMessages(prev => [...prev, userMsg, assistantMsg])

    try {
      for await (const event of streamQuery(query, true, getToken())) {
        if (event.type === 'sources') {
          setMessages(prev =>
            prev.map(m => m.id === assistantId ? { ...m, sources: event.sources } : m),
          )
        } else if (event.type === 'text') {
          setMessages(prev =>
            prev.map(m => m.id === assistantId ? { ...m, content: m.content + event.content } : m),
          )
        }
      }
    } finally {
      isStreamingRef.current = false
      setIsStreaming(false)
    }
  }, [])

  const clearHistory = useCallback(() => {
    skipNextSave.current = true
    setMessages([])
    localStorage.removeItem(STORAGE_KEY)
  }, [])

  return { messages, isStreaming, sendMessage, clearHistory }
}
