import { act, fireEvent, render, screen } from '@testing-library/react'
import IngestPanel from '../IngestPanel'

jest.mock('@/lib/api', () => ({
  ingestFile: jest.fn(),
  ApiError: class ApiError extends Error {
    constructor(public status: number, message: string) { super(message) }
  },
}))

import { ingestFile } from '@/lib/api'
const mockIngest = ingestFile as jest.MockedFunction<typeof ingestFile>

beforeEach(() => jest.clearAllMocks())

function uploadFile(filename = 'test.pdf') {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement
  const file = new File(['content'], filename, { type: 'application/pdf' })
  fireEvent.change(input, { target: { files: [file] } })
}

test('renders idle state with upload button', () => {
  render(<IngestPanel />)
  expect(screen.getByRole('button', { name: 'Upload PDF' })).toBeInTheDocument()
})

test('shows spinner while uploading', async () => {
  let resolve!: (v: unknown) => void
  mockIngest.mockReturnValue(new Promise(r => { resolve = r }) as never)
  render(<IngestPanel />)
  act(() => { uploadFile() })
  expect(screen.getByText(/Uploading/)).toBeInTheDocument()
  await act(async () => { resolve({ pdf_id: 'x', filename: 'test.pdf', chunks_added: 1 }) })
})

test('shows success card after upload', async () => {
  mockIngest.mockResolvedValue({ pdf_id: 'x', filename: 'doc.pdf', chunks_added: 7 } as never)
  render(<IngestPanel />)
  await act(async () => { uploadFile('doc.pdf') })
  expect(screen.getByText(/doc\.pdf/)).toBeInTheDocument()
  expect(screen.getByText(/7 chunks added/)).toBeInTheDocument()
})

test('"Upload another" resets to idle', async () => {
  mockIngest.mockResolvedValue({ pdf_id: 'x', filename: 'doc.pdf', chunks_added: 3 } as never)
  render(<IngestPanel />)
  await act(async () => { uploadFile() })
  fireEvent.click(screen.getByRole('button', { name: 'Upload another' }))
  expect(screen.getByRole('button', { name: 'Upload PDF' })).toBeInTheDocument()
})

test('shows error message on failure', async () => {
  const { ApiError } = jest.requireMock('@/lib/api')
  mockIngest.mockRejectedValue(new ApiError(422, "Failed to extract 'bad.pdf'"))
  render(<IngestPanel />)
  await act(async () => { uploadFile('bad.pdf') })
  expect(screen.getByText(/Failed to extract/)).toBeInTheDocument()
})

test('"Try again" resets to idle', async () => {
  const { ApiError } = jest.requireMock('@/lib/api')
  mockIngest.mockRejectedValue(new ApiError(422, 'error'))
  render(<IngestPanel />)
  await act(async () => { uploadFile() })
  fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
  expect(screen.getByRole('button', { name: 'Upload PDF' })).toBeInTheDocument()
})
