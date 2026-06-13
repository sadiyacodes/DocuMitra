import { render, screen } from '@testing-library/react'
import Sidebar from '../Sidebar'

jest.mock('next/navigation', () => ({
  usePathname: () => '/chat',
}))

jest.mock('next/link', () => ({
  __esModule: true,
  default: ({ href, children, className }: { href: string; children: React.ReactNode; className?: string }) => (
    <a href={href} className={className}>{children}</a>
  ),
}))

test('renders all three nav links', () => {
  render(<Sidebar />)
  expect(screen.getByText('Chat')).toBeInTheDocument()
  expect(screen.getByText('Ingest')).toBeInTheDocument()
  expect(screen.getByText('Retrieval')).toBeInTheDocument()
})

test('highlights the active link', () => {
  render(<Sidebar />)
  const chatLink = screen.getByText('Chat').closest('a')
  expect(chatLink?.className).toContain('bg-blue-50')
})

test('does not highlight inactive links', () => {
  render(<Sidebar />)
  const ingestLink = screen.getByText('Ingest').closest('a')
  expect(ingestLink?.className).not.toContain('bg-blue-50')
})
