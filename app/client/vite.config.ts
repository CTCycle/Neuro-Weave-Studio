import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

const apiProxy = {
  target: 'http://127.0.0.1:8000',
  changeOrigin: false,
}

export default defineConfig({
  plugins: [react()],
  envDir: '../../settings',
  server: {
    port: 5173,
    proxy: {
      '/api': apiProxy,
      '/health': apiProxy,
    },
  },
  preview: {
    port: 5173,
    proxy: {
      '/api': apiProxy,
      '/health': apiProxy,
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    css: true,
  },
})
