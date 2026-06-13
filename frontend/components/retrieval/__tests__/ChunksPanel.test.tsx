import { fireEvent, render, screen } from '@testing-library/react'
import ChunksPanel from '../ChunksPanel'

jest.mock('@/hooks/useChunks', () => ({
  useChunks: jest.fn(),
}))

import { useChunks } from '@/hooks/useChunks'
const mockUseChunks = useChunks as jest.MockedFunction<typeof useChunks>

const fakeResult = {
  chunk_id: 'c1', pdf_id: 'p1', filename: 'report.pdf', page_number: 3,
  text: 'Some interesting text content here.', token_count: 6,
  language: 'en', bbox: null, similarity: 0.92,
}

function makeHook(overrides = {}) {
  return { results: [], isLoading: false, error: null, search: jest.fn(), ...overrides }
}

beforeEach(() => jest.clearAllMocks())

test('renders search input and Search button', () => {
  mockUseChunks.mockReturnValue(makeHook() as never)
  render(<ChunksPanel />)
  expect(screen.getByPlaceholderText(/Enter a query/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Search' })).toBeInTheDocument()
})

test('renders k selector with options 3, 5, 10, 20', () => {
  mockUseChunks.mockReturnValue(makeHook() as never)
  render(<ChunksPanel />)
  ;[3, 5, 10, 20].forEach(n => {
    expect(screen.getByRole('option', { name: `Top ${n}` })).toBeInTheDocument()
  })
})

test('calls search with query and k on button click', () => {
  const search = jest.fn()
  mockUseChunks.mockReturnValue(makeHook({ search }) as never)
  render(<ChunksPanel />)
  fireEvent.change(screen.getByPlaceholderText(/Enter a query/), { target: { value: 'test query' } })
  fireEvent.click(screen.getByRole('button', { name: 'Search' }))
  expect(search).toHaveBeenCalledWith('test query', 5)
})

test('shows error message when error is set', () => {
  mockUseChunks.mockReturnValue(makeHook({ error: 'Something went wrong' }) as never)
  render(<ChunksPanel />)
  expect(screen.getByText('Something went wrong')).toBeInTheDocument()
})

test('renders chunk cards with filename, page and similarity', () => {
  mockUseChunks.mockReturnValue(makeHook({ results: [fakeResult] }) as never)
  render(<ChunksPanel />)
  expect(screen.getByText(/report\.pdf/)).toBeInTheDocument()
  expect(screen.getByText(/p\.3/)).toBeInTheDocument()
  expect(screen.getByText('92%')).toBeInTheDocument()
})

test('shows empty state after search when no results', () => {
  const search = jest.fn()
  // Simulate: user typed and clicked search, hook returns empty results
  mockUseChunks.mockReturnValue(makeHook({ search, results: [] }) as never)
  render(<ChunksPanel />)
  // Simulate having searched (set internal query state)
  fireEvent.change(screen.getByPlaceholderText(/Enter a query/), { target: { value: 'nothing' } })
  fireEvent.click(screen.getByRole('button', { name: 'Search' }))
  // After search, if results=[] and query is non-empty, show empty state
  expect(screen.getByText(/No chunks found/)).toBeInTheDocument()
})
