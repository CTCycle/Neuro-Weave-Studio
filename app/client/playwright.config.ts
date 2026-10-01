// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, devices } from '@playwright/test'

const playwrightCacheRoot = fileURLToPath(new URL('../../runtimes/cache/playwright/', import.meta.url))
process.env['PLAYWRIGHT_BROWSERS_PATH'] = join(playwrightCacheRoot, 'browsers')

const configuredWorkers = process.env['PLAYWRIGHT_WORKERS']
const workers = configuredWorkers
  ? Number.parseInt(configuredWorkers, 10)
  : 1
const isCi = Boolean(process.env['CI'])
const baseURL = process.env['PLAYWRIGHT_BASE_URL'] ?? 'http://127.0.0.1:6426'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: isCi,
  retries: isCi ? 2 : 0,
  // E2E workers share one SQLite database; fixture-only workflows can exercise
  // concurrent access while the runtime-job group owns its CPU lease serially.
  workers,
  reporter: 'line',
  outputDir: join(playwrightCacheRoot, 'test-results'),
  use: {
    baseURL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    ...devices['Desktop Chrome'],
  },
  ...(process.env['PLAYWRIGHT_BASE_URL'] ? {} : {
    webServer: {
      command: 'npm run dev -- --host 127.0.0.1 --port 6426',
      url: 'http://127.0.0.1:6426/models',
      reuseExistingServer: false,
      timeout: 120_000,
    },
  }),
})
