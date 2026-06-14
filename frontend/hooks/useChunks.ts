'use client'

import { useCallback, useState } from 'react'
import { type SearchResult, getChunks } from '@/lib/api'
import { getToken } from '@/lib/auth'

interface State {
  results: SearchResult[]
  isLoading: boolean
  error: string | null
}

export function useChunks() {
  const [state, setState] = useState<State>({
    results: [],
    isLoading: false,
    error: null,
  })

  const search = useCallback(async (query: string, k: number) => {
    setState({ results: [], isLoading: true, error: null })
    try {
      const data = await getChunks(query, k, getToken())
      setState({ results: data.results, isLoading: false, error: null })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error'
      setState({ results: [], isLoading: false, error: message })
    }
  }, [])

  return { ...state, search }
}
