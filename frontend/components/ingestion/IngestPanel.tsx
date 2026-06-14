'use client'
import { type ChangeEvent, type DragEvent, useRef, useState } from 'react'
import { ApiError, ingestFile, ingestCsv, ingestJson } from '@/lib/api'
import { getToken } from '@/lib/auth'

type FileType = 'pdf' | 'csv' | 'json'

type State =
  | { kind: 'idle'; dragging: boolean }
  | { kind: 'uploading'; filename: string }
  | { kind: 'done'; filename: string; chunks_added: number }
  | { kind: 'error'; message: string }

const FILE_TYPE_CONFIG: Record<FileType, { label: string; accept: string; hint: string }> = {
  pdf:  { label: 'PDF',  accept: '.pdf',       hint: 'PDF documents' },
  csv:  { label: 'CSV',  accept: '.csv',        hint: 'Spreadsheets / tabular data' },
  json: { label: 'JSON', accept: '.json',       hint: 'JSON arrays or objects' },
}

export default function IngestPanel() {
  const [fileType, setFileType] = useState<FileType>('pdf')
  const [roles, setRoles] = useState('')
  const [state, setState] = useState<State>({ kind: 'idle', dragging: false })
  const inputRef = useRef<HTMLInputElement>(null)

  const config = FILE_TYPE_CONFIG[fileType]

  async function upload(file: File) {
    setState({ kind: 'uploading', filename: file.name })
    const roleList = roles.split(',').map(r => r.trim()).filter(Boolean)
    const token = getToken()
    try {
      let res
      if (fileType === 'csv') res = await ingestCsv(file, roleList, token)
      else if (fileType === 'json') res = await ingestJson(file, roleList, token)
      else res = await ingestFile(file, roleList, token)
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
    setState(s => s.kind === 'idle' ? { ...s, dragging: false } : s)
    const file = e.dataTransfer.files[0]
    if (file) upload(file)
  }

  if (state.kind === 'uploading') {
    return (
      <div className="p-8 flex flex-col items-center justify-center gap-5 h-72">
        <div className="relative">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-gray-200 border-t-blue-600" />
          <div className="absolute inset-0 flex items-center justify-center">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="14,2 14,8 20,8" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-gray-700">Processing {state.filename}</p>
          <p className="text-xs text-gray-400 mt-1">Extracting · Chunking · Embedding…</p>
        </div>
      </div>
    )
  }

  if (state.kind === 'done') {
    return (
      <div className="p-8 flex flex-col items-center justify-center gap-5 h-72">
        <div className="w-14 h-14 rounded-full bg-green-50 flex items-center justify-center">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2">
            <polyline points="20,6 9,17 4,12" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <div className="text-center">
          <p className="text-sm font-semibold text-gray-900">{state.filename}</p>
          <p className="text-sm text-green-600 mt-0.5">{state.chunks_added.toLocaleString()} chunks indexed</p>
        </div>
        <button
          type="button"
          onClick={() => setState({ kind: 'idle', dragging: false })}
          className="px-5 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
        >
          Upload another
        </button>
      </div>
    )
  }

  if (state.kind === 'error') {
    return (
      <div className="p-8 flex flex-col items-center justify-center gap-5 h-72">
        <div className="w-14 h-14 rounded-full bg-red-50 flex items-center justify-center">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2">
            <circle cx="12" cy="12" r="10" strokeLinecap="round" strokeLinejoin="round"/>
            <line x1="12" y1="8" x2="12" y2="12" strokeLinecap="round"/>
            <line x1="12" y1="16" x2="12.01" y2="16" strokeLinecap="round"/>
          </svg>
        </div>
        <div className="text-center max-w-sm">
          <p className="text-sm font-medium text-gray-900 mb-1">Upload failed</p>
          <p className="text-xs text-red-600">{state.message}</p>
        </div>
        <button
          type="button"
          onClick={() => setState({ kind: 'idle', dragging: false })}
          className="px-5 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
        >
          Try again
        </button>
      </div>
    )
  }

  const isDragging = state.dragging

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-gray-900">Ingest a Document</h1>
        <p className="text-sm text-gray-500 mt-1">Upload a document to extract, chunk, and index it for Q&A.</p>
      </div>

      {/* File type tabs */}
      <div className="flex gap-1 mb-5 bg-gray-100 p-1 rounded-lg w-fit">
        {(Object.keys(FILE_TYPE_CONFIG) as FileType[]).map(ft => (
          <button
            key={ft}
            type="button"
            onClick={() => { setFileType(ft); setState({ kind: 'idle', dragging: false }) }}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
              fileType === ft
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {FILE_TYPE_CONFIG[ft].label}
          </button>
        ))}
      </div>

      {/* Roles field */}
      <div className="mb-5">
        <label className="block text-xs font-medium text-gray-600 mb-1">
          Access roles <span className="text-gray-400 font-normal">(comma-separated, leave blank for all)</span>
        </label>
        <input
          type="text"
          value={roles}
          onChange={e => setRoles(e.target.value)}
          placeholder="e.g. admin, hr, finance"
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={e => { e.preventDefault(); setState(s => s.kind === 'idle' ? { ...s, dragging: true } : s) }}
        onDragLeave={() => setState(s => s.kind === 'idle' ? { ...s, dragging: false } : s)}
        className={`border-2 border-dashed rounded-xl p-16 text-center transition-all cursor-pointer ${
          isDragging
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-50/40'
        }`}
        onClick={() => inputRef.current?.click()}
      >
        <div className="flex flex-col items-center gap-4">
          <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-colors ${isDragging ? 'bg-blue-100' : 'bg-white border border-gray-200'}`}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke={isDragging ? '#2563eb' : '#6b7280'} strokeWidth="1.5">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" strokeLinecap="round" strokeLinejoin="round"/>
              <polyline points="14,2 14,8 20,8" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="12" y1="12" x2="12" y2="18" strokeLinecap="round"/>
              <polyline points="9,15 12,12 15,15" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-700">
              {isDragging ? `Drop your ${config.label} here` : `Drag & drop a ${config.label}, or click to browse`}
            </p>
            <p className="text-xs text-gray-400 mt-1">{config.hint} · No size limit</p>
          </div>
          <button
            type="button"
            onClick={e => { e.stopPropagation(); inputRef.current?.click() }}
            className="px-5 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors shadow-sm"
          >
            Choose file
          </button>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={config.accept}
          className="hidden"
          onChange={handleChange}
        />
      </div>
    </div>
  )
}
