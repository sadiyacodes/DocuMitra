'use client'
import { type ChangeEvent, type DragEvent, useRef, useState } from 'react'
import { ApiError, ingestFile } from '@/lib/api'

type State =
  | { kind: 'idle' }
  | { kind: 'uploading'; filename: string }
  | { kind: 'done'; filename: string; chunks_added: number }
  | { kind: 'error'; message: string }

export default function IngestPanel() {
  const [state, setState] = useState<State>({ kind: 'idle' })
  const inputRef = useRef<HTMLInputElement>(null)

  async function upload(file: File) {
    setState({ kind: 'uploading', filename: file.name })
    try {
      const res = await ingestFile(file)
      setState({ kind: 'done', filename: res.filename, chunks_added: res.chunks_added })
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : 'Upload failed. Please try again.'
      setState({ kind: 'error', message })
    }
  }

  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) upload(file)
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file?.type === 'application/pdf') upload(file)
  }

  if (state.kind === 'uploading') {
    return (
      <div className="p-8 flex flex-col items-center gap-4">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
        <p className="text-sm text-gray-600">Uploading {state.filename}…</p>
      </div>
    )
  }

  if (state.kind === 'done') {
    return (
      <div className="p-8 flex flex-col items-center gap-4">
        <p className="text-green-600 text-sm font-medium">
          ✓ {state.filename} — {state.chunks_added} chunks added
        </p>
        <button
          type="button"
          onClick={() => setState({ kind: 'idle' })}
          className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm"
        >
          Upload another
        </button>
      </div>
    )
  }

  if (state.kind === 'error') {
    return (
      <div className="p-8 flex flex-col items-center gap-4">
        <p className="text-red-600 text-sm">{state.message}</p>
        <button
          type="button"
          onClick={() => setState({ kind: 'idle' })}
          className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm"
        >
          Try again
        </button>
      </div>
    )
  }

  return (
    <div className="p-8">
      <div
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-blue-400 transition-colors"
      >
        <p className="text-sm text-gray-600 mb-4">Drag a PDF here or click to browse</p>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm"
        >
          Upload PDF
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={handleChange}
        />
      </div>
    </div>
  )
}
