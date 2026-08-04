import path from 'node:path'
import { loadEnv } from 'vite'
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

const ABSOLUTE_URL_PATTERN = /^[a-zA-Z][a-zA-Z\d+\-.]*:\/\//

function normalizeApiBase(rawValue: string | undefined): string {
  const candidate = (rawValue || '/api').trim()
  if (!candidate) return '/api'
  if (ABSOLUTE_URL_PATTERN.test(candidate)) {
    throw new Error('VITE_API_BASE_URL must be a relative path (for example /api).')
  }
  const withLeadingSlash = candidate.startsWith('/') ? candidate : `/${candidate}`
  return withLeadingSlash.length > 1 && withLeadingSlash.endsWith('/')
    ? withLeadingSlash.slice(0, -1)
    : withLeadingSlash
}

export default defineConfig(({ mode }) => {
  const settingsDir = path.resolve(__dirname, '../../settings')
  const clientEnv = loadEnv(mode, __dirname, '')
  const settingsEnv = loadEnv(mode, settingsDir, '')
  const env = { ...settingsEnv, ...clientEnv, ...process.env }
  const apiHost = env.FASTAPI_HOST || '127.0.0.1'
  const apiPort = env.FASTAPI_PORT || '8000'
  const uiHost = env.UI_HOST || '127.0.0.1'
  const uiPort = Number.parseInt(env.UI_PORT || '5173', 10)
  const apiBase = normalizeApiBase(env.VITE_API_BASE_URL)
  const apiTarget = `http://${apiHost}:${apiPort}`
  const proxy = {
    [apiBase]: { target: apiTarget, changeOrigin: true, ws: true },
    '/health': { target: apiTarget, changeOrigin: true },
  }

  return {
    plugins: [react()],
    server: {
      host: uiHost,
      port: uiPort,
      strictPort: true,
      proxy,
    },
    preview: {
      host: uiHost,
      port: uiPort,
      strictPort: true,
      proxy,
    },
    test: {
      environment: 'jsdom',
      setupFiles: './src/test/setup.ts',
      css: true,
    },
  }
})
