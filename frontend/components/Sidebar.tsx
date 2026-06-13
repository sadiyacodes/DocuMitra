'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const NAV = [
  { href: '/chat', label: 'Chat' },
  { href: '/ingestion', label: 'Ingest' },
  { href: '/retrieval', label: 'Retrieval' },
]

export default function Sidebar() {
  const pathname = usePathname()
  return (
    <aside className="w-44 shrink-0 bg-white border-r border-gray-200 flex flex-col p-4 gap-1">
      <span className="text-base font-bold mb-4 text-gray-900">DocuMitra</span>
      {NAV.map(({ href, label }) => (
        <Link
          key={href}
          href={href}
          className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
            pathname === href
              ? 'bg-blue-50 text-blue-700'
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          {label}
        </Link>
      ))}
    </aside>
  )
}
