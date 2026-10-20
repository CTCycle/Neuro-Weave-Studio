// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { mkdir, rm } from 'node:fs/promises'
import path from 'node:path'
import { expect, test } from '@playwright/test'

const repositoryRoot = path.resolve(process.cwd(), '..', '..')
const evidenceRoot = path.join(repositoryRoot, 'assets', 'QA', 'database-settings-ui-e2e-20261020')
test.describe.configure({ mode: 'serial' })

test('tests, switches, reloads, and restores independent SQLite settings through Settings UI', async ({ page, request }) => {
  test.setTimeout(120_000)
  await mkdir(evidenceRoot, { recursive: true })

  const initialStatusResponse = await request.get('/api/settings/database')
  expect(initialStatusResponse.ok()).toBeTruthy()
  const initialStatus = await initialStatusResponse.json() as { sqlite_path: string }
  const initialSettingsResponse = await request.get('/api/settings')
  expect(initialSettingsResponse.ok()).toBeTruthy()
  const initialSettings = await initialSettingsResponse.json() as { log_level: string }
  const secondaryPath = path.join(evidenceRoot, `switch-candidate-${process.pid}.sqlite`)
  const invalidPath = path.join(evidenceRoot, `invalid-target-${process.pid}`)
  await mkdir(invalidPath, { recursive: true })

  try {
    await page.goto('/settings')
    await expect(page.getByRole('heading', { name: 'Database', exact: true })).toBeVisible()
    await expect(page.getByText('SQLITE · Connected', { exact: true })).toBeVisible()
    await expect(page.getByLabel('Log level')).toHaveValue(initialSettings.log_level)

    await page.getByLabel('Log level').selectOption('WARNING')
    const savePrimary = page.waitForResponse((response) => response.url().endsWith('/api/settings') && response.request().method() === 'PUT')
    await page.getByRole('button', { name: 'Save application settings', exact: true }).click()
    expect((await savePrimary).status()).toBe(200)
    await expect(page.getByRole('status').filter({ hasText: 'Application settings saved to the active database.' })).toBeVisible()

    await page.getByLabel('Database file path').fill(invalidPath)
    const failedTest = page.waitForResponse((response) => response.url().endsWith('/api/settings/database/test') && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Test connection', exact: true }).click()
    expect((await failedTest).status()).toBe(422)
    await expect(page.getByRole('alert')).toContainText('Connection test failed')

    await page.getByLabel('Database file path').fill(secondaryPath)
    const successfulTest = page.waitForResponse((response) => response.url().endsWith('/api/settings/database/test') && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Test connection', exact: true }).click()
    expect((await successfulTest).status()).toBe(200)
    await expect(page.getByText('SQLite connection and required schema are ready.', { exact: true })).toBeVisible()
    await page.screenshot({ path: path.join(evidenceRoot, 'settings-database-tested.png'), fullPage: true })

    const switchToSecondary = page.waitForResponse((response) => response.url().endsWith('/api/settings/database') && response.request().method() === 'PUT')
    await page.getByRole('button', { name: 'Save and switch', exact: true }).click()
    expect((await switchToSecondary).status()).toBe(200)
    await expect(page.getByText(secondaryPath, { exact: true })).toBeVisible()
    await expect(page.getByLabel('Log level')).toHaveValue('INFO')
    await expect(page.getByRole('status').filter({ hasText: 'Database switched.' })).toBeVisible()

    await page.getByLabel('Log level').selectOption('DEBUG')
    const saveSecondary = page.waitForResponse((response) => response.url().endsWith('/api/settings') && response.request().method() === 'PUT')
    await page.getByRole('button', { name: 'Save application settings', exact: true }).click()
    expect((await saveSecondary).status()).toBe(200)
    await expect(page.getByLabel('Log level')).toHaveValue('DEBUG')
    await page.screenshot({ path: path.join(evidenceRoot, 'settings-database-switched.png'), fullPage: true })

    await page.reload()
    await expect(page.getByText(secondaryPath, { exact: true })).toBeVisible()
    await expect(page.getByLabel('Log level')).toHaveValue('DEBUG')
    await page.screenshot({ path: path.join(evidenceRoot, 'settings-database-reloaded.png'), fullPage: true })

    await page.getByLabel('Database file path').fill(initialStatus.sqlite_path)
    const switchBack = page.waitForResponse((response) => response.url().endsWith('/api/settings/database') && response.request().method() === 'PUT')
    await page.getByRole('button', { name: 'Save and switch', exact: true }).click()
    expect((await switchBack).status()).toBe(200)
    await expect(page.getByText(initialStatus.sqlite_path, { exact: true })).toBeVisible()
    await expect(page.getByLabel('Log level')).toHaveValue('WARNING')
  } finally {
    const status = await request.get('/api/settings/database')
    if (status.ok()) {
      const current = await status.json() as { sqlite_path: string }
      if (current.sqlite_path !== initialStatus.sqlite_path) {
        await request.put('/api/settings/database', {
          data: { backend: 'sqlite', schema_version: 1, sqlite_path: initialStatus.sqlite_path, connect_timeout_seconds: 30 },
        })
      }
    }
    for (const suffix of ['', '-wal', '-shm', '-journal']) {
      await rm(`${secondaryPath}${suffix}`, { force: true })
    }
    await rm(invalidPath, { recursive: true, force: true })
  }
})

test('renders the selected database and its settings from the active backend state', async ({ page, request }) => {
  const databaseResponse = await request.get('/api/settings/database')
  const settingsResponse = await request.get('/api/settings')
  expect(databaseResponse.ok()).toBeTruthy()
  expect(settingsResponse.ok()).toBeTruthy()
  const database = await databaseResponse.json() as { backend: string; sqlite_path: string; available: boolean }
  const settings = await settingsResponse.json() as { log_level: string }

  await page.goto('/settings')
  await expect(page.getByText(`${database.backend.toUpperCase()} · Connected`, { exact: true })).toBeVisible()
  await expect(page.getByText(database.sqlite_path, { exact: true })).toBeVisible()
  await expect(page.getByLabel('Log level')).toHaveValue(settings.log_level)
  expect(database.available).toBe(true)
  await page.screenshot({ path: path.join(evidenceRoot, 'settings-database-after-backend-restart.png'), fullPage: true })
})
