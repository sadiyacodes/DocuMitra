'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { getToken, getRole } from '@/lib/auth'

export function useAuthGuard(options: { adminOnly?: boolean } = {}) {
  const router = useRouter()
  useEffect(() => {
    if (!getToken()) {
      router.replace('/login')
      return
    }
    if (options.adminOnly && getRole() !== 'admin') {
      router.replace('/chat')
    }
  }, [router, options.adminOnly])
}
