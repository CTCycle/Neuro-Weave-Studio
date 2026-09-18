import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitest/config'

const cacheDir = fileURLToPath(new URL('../../runtimes/cache/vite/', import.meta.url))

export default defineConfig({
  cacheDir,
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts', 'src/**/*.spec.ts'],
  },
})
