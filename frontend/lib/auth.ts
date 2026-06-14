const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
const TOKEN_KEY = 'documitra_token'

export function getToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

export async function login(username: string, password: string): Promise<void> {
  const body = new URLSearchParams({ username, password, grant_type: 'password' })
  const res = await fetch(`${BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  })
  if (!res.ok) throw new Error('Invalid credentials')
  const { access_token } = (await res.json()) as { access_token: string }
  setToken(access_token)
}

export function logout(): void {
  clearToken()
}

function _decodePayload(token: string): Record<string, unknown> | null {
  try {
    const b64 = token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')
    return JSON.parse(atob(b64))
  } catch { return null }
}

export function getRole(): string | null {
  const token = getToken()
  if (!token) return null
  const p = _decodePayload(token)
  return (p?.role as string) ?? null
}

export function getUsername(): string | null {
  const token = getToken()
  if (!token) return null
  const p = _decodePayload(token)
  return (p?.sub as string) ?? null
}
