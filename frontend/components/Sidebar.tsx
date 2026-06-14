'use client'

import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'
import { clearToken, getRole, getUsername } from '@/lib/auth'

const BASE_NAV = [
  { href: '/chat',      label: 'Chat',      adminOnly: false },
  { href: '/ingestion', label: 'Ingest',    adminOnly: true  },
  { href: '/retrieval', label: 'Retrieval', adminOnly: false },
]

export default function Sidebar() {
  const pathname = usePathname()
  const router   = useRouter()
  const [role, setRole]         = useState<string | null>(null)
  const [username, setUsername] = useState<string | null>(null)

  useEffect(() => {
    setRole(getRole())
    setUsername(getUsername())
  }, [pathname])

  if (pathname === '/login') return null

  const nav = BASE_NAV.filter(item => !item.adminOnly || role === 'admin')

  function handleLogout() {
    clearToken()
    router.push('/login')
  }

  return (
    <aside className="w-44 shrink-0 bg-white border-r border-gray-200 flex flex-col p-4 gap-1">
      <span className="text-base font-bold mb-4 text-gray-900">DocuMitra</span>

      {nav.map(({ href, label }) => (
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

      <div className="mt-auto pt-4 border-t border-gray-100 flex flex-col gap-2">
        {username && (
          <div className="px-3">
            <p className="text-xs font-medium text-gray-700 truncate">{username}</p>
            <p className="text-xs text-gray-400 capitalize">{role}</p>
          </div>
        )}
        <button
          type="button"
          onClick={handleLogout}
          className="px-3 py-1.5 text-xs text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-md text-left transition-colors"
        >
          Sign out
        </button>
      </div>
    </aside>
  )
}
