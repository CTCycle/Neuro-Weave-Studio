import path from 'node:path'
import { loadEnv } from 'vite'
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const settingsDir = path.resolve(__dirname, '../settings')
  const settingsEnv = loadEnv(mode, settingsDir, '')
  const clientEnv = loadEnv(mode, __dirname, '')
  const env = { ...settingsEnv, ...clientEnv, ...process.env }
  const apiHost = env.FASTAPI_HOST || '127.0.0.1'
  const apiPort = env.FASTAPI_PORT || '8000'
  const uiHost = env.UI_HOST || '127.0.0.1'
  const uiPort = Number.parseInt(env.UI_PORT || '5173', 10)

  return {
    plugins: [react()],
    server: {
      host: uiHost,
      port: uiPort,
      strictPort: true,
      proxy: {
        '/api': `http://${apiHost}:${apiPort}`,
        '/health': `http://${apiHost}:${apiPort}`,
      },
    },
    test: {
      environment: 'jsdom',
      setupFiles: './src/test/setup.ts',
      css: true,
    },
  }
})
