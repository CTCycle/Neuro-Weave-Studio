// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { mkdir } from 'node:fs/promises'
import path from 'node:path'
import { expect, test } from '@playwright/test'

const repositoryRoot = path.resolve(process.cwd(), '..', '..')
const evidenceRoot = path.join(repositoryRoot, 'assets', 'QA', 'settings-autosave-ui-e2e-20261021')
type SettingsRecord = Record<string, unknown>

async function currentSettings(request: import('@playwright/test').APIRequestContext): Promise<SettingsRecord> {
  const response = await request.get('/api/settings')
  expect(response.ok()).toBeTruthy()
  return response.json() as Promise<SettingsRecord>
}

test.describe.configure({ mode: 'serial' })

let settingsBaseline: SettingsRecord | null = null

test.beforeEach(async ({ request }) => {
  settingsBaseline = await currentSettings(request)
})

test.afterEach(async ({ request }) => {
  if (!settingsBaseline) return
  const response = await request.patch('/api/settings', { data: settingsBaseline })
  expect(response.ok(), 'restore settings after the autosave test').toBeTruthy()
  settingsBaseline = null
})

test('automatically saves every ordinary application settings control', async ({ page, request }) => {
  await mkdir(evidenceRoot, { recursive: true })
  await page.goto('/settings')
  await expect(page.getByRole('heading', { name: 'General', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: /save application settings|apply|cancel/i })).toHaveCount(0)

  const original = await currentSettings(request)
  expect(original).not.toHaveProperty('cors_origins')
  await expect(page.getByLabel('Allowed browser origins')).toHaveCount(0)
  const nextNumber = (field: string, minimum: number, maximum: number): number => {
    const current = Number(original[field])
    return current < maximum ? Math.min(maximum, current + 1) : Math.max(minimum, current - 1)
  }
  const uniquePath = (name: string): string => path.join(evidenceRoot, `${process.pid}-${name}`)
  const values: Array<{ label: string; field: string; value: string | number; select?: boolean }> = [
    { label: 'Log level', field: 'log_level', value: original['log_level'] === 'DEBUG' ? 'INFO' : 'DEBUG', select: true },
    { label: 'Worker timeout (seconds)', field: 'ml_worker_timeout_seconds', value: nextNumber('ml_worker_timeout_seconds', 1, 86400) },
    { label: 'Runtime discovery timeout (seconds)', field: 'runtime_discovery_timeout_seconds', value: nextNumber('runtime_discovery_timeout_seconds', 1, 600) },
    { label: 'Tensor inspection retention (seconds)', field: 'tensor_inspection_ttl_seconds', value: nextNumber('tensor_inspection_ttl_seconds', 60, 86400) },
    { label: 'Cancel grace period (seconds)', field: 'ml_cancel_grace_seconds', value: nextNumber('ml_cancel_grace_seconds', 1, 300) },
    { label: 'Terminate grace period (seconds)', field: 'ml_terminate_grace_seconds', value: nextNumber('ml_terminate_grace_seconds', 1, 300) },
    { label: 'Job history limit', field: 'ml_job_history_limit', value: nextNumber('ml_job_history_limit', 10, 1000) },
    { label: 'Log file', field: 'log_file', value: uniquePath('backend.log') },
    { label: 'Export folder', field: 'export_root', value: uniquePath('exports') },
    { label: 'ML job folder', field: 'ml_job_root', value: uniquePath('jobs') },
    { label: 'Tensor inspection folder', field: 'tensor_inspection_root', value: uniquePath('inspections') },
    { label: 'Runtime discovery cache', field: 'runtime_discovery_path', value: path.join(repositoryRoot, 'runtimes', 'cache', `settings-autosave-discovery-${process.pid}.json`) },
  ]

  for (const item of values) {
    if (item.field === 'log_file') await page.getByText('Storage and log locations', { exact: true }).click()
    const responsePromise = page.waitForResponse((response) => response.url().endsWith('/api/settings') && response.request().method() === 'PATCH')
    const control = page.getByLabel(item.label)
    if (item.select) await control.selectOption(String(item.value))
    else await control.fill(String(item.value))
    const response = await responsePromise
    expect(response.status(), `${item.label} should save successfully`).toBe(200)
    const saved = await response.json() as SettingsRecord
    const expected = item.field.endsWith('_root') || item.field === 'log_file' || item.field === 'runtime_discovery_path'
      ? path.resolve(String(item.value))
      : item.value
    expect(saved[item.field], `${item.label} should be applied immediately`).toBe(expected)
    expect((await currentSettings(request))[item.field], `${item.label} should persist in the active database`).toBe(expected)
  }

  await expect(page.getByRole('status').filter({ hasText: 'Changes saved automatically.' })).toBeVisible()
  await page.locator('main.app-content').evaluate((element) => element.scrollTo(0, 0))
  await page.screenshot({ path: path.join(evidenceRoot, 'settings-general-autosave.png'), fullPage: true })
  await page.reload()
  await page.getByText('Storage and log locations', { exact: true }).click()
  const restored = await currentSettings(request)
  for (const item of values) {
    const expected = item.field.endsWith('_root') || item.field === 'log_file' || item.field === 'runtime_discovery_path'
      ? path.resolve(String(item.value))
      : item.value
    expect(restored[item.field], `${item.label} should restore after reload`).toBe(expected)
    await expect(page.getByLabel(item.label)).toHaveValue(String(expected))
  }
})

test('debounces rapid text edits and never lets a stale value win', async ({ page, request }) => {
  await page.goto('/settings')
  const original = await currentSettings(request)
  await page.getByText('Storage and log locations', { exact: true }).click()
  const input = page.getByLabel('Log file')
  const patchBodies: string[] = []
  page.on('request', (requestEvent) => {
    if (requestEvent.url().endsWith('/api/settings') && requestEvent.method() === 'PATCH') {
      patchBodies.push(requestEvent.postData() ?? '')
    }
  })

  const intermediateValue = path.join(evidenceRoot, `${process.pid}-intermediate.log`)
  const finalValue = path.join(evidenceRoot, `${process.pid}-debounced.log`)
  await input.fill(intermediateValue)
  const finalSave = page.waitForResponse((response) => response.url().endsWith('/api/settings') && response.request().method() === 'PATCH')
  await input.fill(finalValue)
  expect((await finalSave).status()).toBe(200)
  await expect.poll(async () => (await currentSettings(request))['log_file']).toBe(path.resolve(finalValue))
  expect(patchBodies).toHaveLength(1)
  expect(JSON.parse(patchBodies[0])['log_file']).toBe(finalValue)
  expect(original['log_file']).not.toBe(finalValue)
})

test('reverts a failed application settings write and explains the failure', async ({ page, request }) => {
  const initial = await currentSettings(request)
  await page.route('**/api/settings', async (route) => {
    if (route.request().method() === 'PATCH') {
      await route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'injected settings write failure' }) })
      return
    }
    await route.continue()
  })
  await page.goto('/settings')
  await page.getByText('Storage and log locations', { exact: true }).click()
  const input = page.getByLabel('Log file')
  await input.fill(path.join(evidenceRoot, `${process.pid}-failed.log`))
  await expect(page.getByRole('alert')).toContainText('injected settings write failure')
  await expect(input).toHaveValue(String(initial['log_file']))
  expect((await currentSettings(request))['log_file']).toBe(initial['log_file'])
})
