'use client'
import { useState } from 'react'
import { useChunks } from '@/hooks/useChunks'
import type { SearchResult } from '@/lib/api'

const K_OPTIONS = [3, 5, 10, 20]

function ChunkCard({ result }: { result: SearchResult }) {
  const [expanded, setExpanded] = useState(false)
  const truncated = result.text.slice(0, 300)
  const needsTruncation = result.text.length > 300

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-900">
          {result.filename} · p.{result.page_number}
        </span>
        <span className="text-sm text-blue-600 font-medium">
          {Math.round(result.similarity * 100)}%
        </span>
      </div>
      <p className="text-sm text-gray-700 whitespace-pre-wrap">
        {expanded ? result.text : truncated}
        {!expanded && needsTruncation && '…'}
      </p>
      {needsTruncation && (
        <button
          type="button"
          onClick={() => setExpanded(v => !v)}
          className="mt-2 text-xs text-blue-600 hover:underline"
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
      <div className="mt-2 flex flex-wrap gap-2 text-xs text-gray-400">
        <span>{result.language}</span>
        <span>·</span>
        <span>{result.token_count} tokens</span>
        <span>·</span>
        <span className="uppercase tracking-wide">{result.source_type}</span>
        {result.access_roles.length > 0 && (
          <>
            <span>·</span>
            <span>{result.access_roles.join(', ')}</span>
          </>
        )}
      </div>
    </div>
  )
}

export default function ChunksPanel() {
  const [query, setQuery] = useState('')
  const [k, setK] = useState(5)
  const [searched, setSearched] = useState(false)
  const { results, isLoading, error, search } = useChunks()

  function handleSearch() {
    const trimmed = query.trim()
    if (!trimmed) return
    setSearched(true)
    search(trimmed, k)
  }

  return (
    <div className="p-6 flex flex-col gap-4">
      <h1 className="text-base font-semibold text-gray-900">Retrieval</h1>
      <div className="flex gap-2">
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSearch()}
          placeholder="Enter a query…"
          className="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <select
          value={k}
          onChange={e => setK(Number(e.target.value))}
          className="rounded-md border border-gray-300 px-2 py-2 text-sm"
        >
          {K_OPTIONS.map(n => (
            <option key={n} value={n}>Top {n}</option>
          ))}
        </select>
        <button
          type="button"
          onClick={handleSearch}
          className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm"
        >
          Search
        </button>
      </div>
      {isLoading && (
        <div className="flex flex-col gap-3">
          {Array.from({ length: k }).map((_, i) => (
            <div key={i} className="h-24 bg-gray-100 animate-pulse rounded-lg" />
          ))}
        </div>
      )}
      {error && <p className="text-sm text-red-600">{error}</p>}
      {!isLoading && !error && searched && results.length === 0 && (
        <p className="text-sm text-gray-500">No chunks found for this query.</p>
      )}
      <div className="flex flex-col gap-3">
        {results.map(r => <ChunkCard key={r.chunk_id} result={r} />)}
      </div>
    </div>
  )
}
