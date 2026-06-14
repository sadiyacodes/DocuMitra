'use client'
import { useAuthGuard } from '@/hooks/useAuthGuard'
import IngestPanel from '@/components/ingestion/IngestPanel'

export default function IngestionPage() {
  useAuthGuard({ adminOnly: true })
  return <IngestPanel />
}
