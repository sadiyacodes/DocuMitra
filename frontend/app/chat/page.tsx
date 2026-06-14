'use client'
import { useAuthGuard } from '@/hooks/useAuthGuard'
import ChatPanel from '@/components/chat/ChatPanel'

export default function ChatPage() {
  useAuthGuard()
  return <ChatPanel />
}
