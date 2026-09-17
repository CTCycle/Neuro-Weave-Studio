import { defineConfig, devices } from '@playwright/test'

const configuredWorkers = process.env['PLAYWRIGHT_WORKERS']
const workers = configuredWorkers
  ? Number.parseInt(configuredWorkers, 10)
  : 1
const isCi = Boolean(process.env['CI'])

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: isCi,
  retries: isCi ? 2 : 0,
  // E2E workers share one SQLite database, so serial execution avoids write-lock races by default.
  // Set PLAYWRIGHT_WORKERS when the backend/storage is isolated for parallel execution.
  workers,
  reporter: 'line',
  use: {
    baseURL: 'http://127.0.0.1:6426',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    ...devices['Desktop Chrome'],
  },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 6426',
    url: 'http://127.0.0.1:6426/models',
    reuseExistingServer: false,
    timeout: 120_000,
  },
})
