'use client'
import { useAuthGuard } from '@/hooks/useAuthGuard'
import ChunksPanel from '@/components/retrieval/ChunksPanel'

export default function RetrievalPage() {
  useAuthGuard()
  return <ChunksPanel />
}
