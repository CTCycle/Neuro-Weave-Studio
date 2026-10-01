// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { join } from 'node:path'
import { expect, test } from '@playwright/test'

type RuntimeApiStatus = {
  id: 'keras' | 'native'
  available: boolean
  state: string
  source: string
}

type RuntimeDevice = {
  device_id: string
  lease_key: string
  kind: string
  name: string
  accelerator: string
  available: boolean
  supported_memory_policies: string[]
}

type ExecutionBackendStatus = {
  id: 'tensorflow' | 'pytorch' | 'jax'
  installed: boolean
  available: boolean
  discovery_state: string
  runtime_apis: RuntimeApiStatus[]
  devices: RuntimeDevice[]
}

type MetadataResponse = {
  execution_backends: ExecutionBackendStatus[]
  runtime_apis: RuntimeApiStatus[]
}

const runtimePreferencesStorageKey = 'neuroweave.runtime-preferences.v1'
const runtimePreferenceBackendStorageKey = 'neuroweave.runtime-preference-backend.v1'

async function saveEvidenceScreenshot(page: import('@playwright/test').Page, name: string): Promise<void> {
  const evidenceDirectory = process.env['NW_T1_04_EVIDENCE_DIR']
  if (evidenceDirectory) await page.screenshot({ path: join(evidenceDirectory, name), fullPage: true })
}

test.beforeEach(async ({ page }) => {
  await page.route('**/api/metadata', async (route) => {
    const upstream = await route.fetch()
    const metadata = await upstream.json() as MetadataResponse
    const pytorch = metadata.execution_backends.find((item) => item.id === 'pytorch')
    const tensorflow = metadata.execution_backends.find((item) => item.id === 'tensorflow')
    if (!pytorch || !tensorflow) throw new Error('The runtime metadata did not include PyTorch and TensorFlow.')

    pytorch.installed = true
    pytorch.available = true
    pytorch.discovery_state = 'verified'
    pytorch.runtime_apis = [
      { id: 'keras', available: true, state: 'verified', source: 'live' },
      { id: 'native', available: true, state: 'verified', source: 'live' },
    ]
    pytorch.devices = [
      { device_id: 'cpu', lease_key: 'cpu', kind: 'cpu', name: 'CPU', accelerator: 'cpu', available: true, supported_memory_policies: ['default'] },
      { device_id: 'cuda:0', lease_key: 'cuda:0', kind: 'gpu', name: 'QA GPU', accelerator: 'cuda', available: true, supported_memory_policies: ['pytorch_fraction'] },
    ]
    tensorflow.installed = true
    tensorflow.available = true
    tensorflow.discovery_state = 'verified'
    tensorflow.runtime_apis = [
      { id: 'keras', available: true, state: 'verified', source: 'live' },
      { id: 'native', available: true, state: 'verified', source: 'live' },
    ]
    tensorflow.devices = [
      { device_id: 'cpu', lease_key: 'cpu', kind: 'cpu', name: 'CPU', accelerator: 'cpu', available: true, supported_memory_policies: ['default'] },
      { device_id: 'gpu:0', lease_key: 'gpu:0', kind: 'gpu', name: 'QA TensorFlow GPU', accelerator: 'cuda', available: true, supported_memory_policies: ['tensorflow_growth', 'tensorflow_limit_mb'] },
    ]
    metadata.runtime_apis = [
      { id: 'keras', available: true, state: 'verified', source: 'live' },
      { id: 'native', available: true, state: 'verified', source: 'live' },
    ]
    const unavailableBackendId = new URL(page.url()).searchParams.get('unavailable_backend')
    const unavailableBackend = metadata.execution_backends.find((item) => item.id === unavailableBackendId)
    if (unavailableBackend) {
      unavailableBackend.available = false
      unavailableBackend.discovery_state = 'unavailable'
      unavailableBackend.runtime_apis = unavailableBackend.runtime_apis.map((api) => ({ ...api, available: false, state: 'unavailable' }))
      unavailableBackend.devices = []
    }
    await route.fulfill({ status: upstream.status(), headers: upstream.headers(), contentType: 'application/json', body: JSON.stringify(metadata) })
  })
})

test('persists the exact runtime device and memory policy after reload', async ({ page }) => {
  await page.goto('/environment')
  const card = page.locator('.backend-status-card-container').filter({ hasText: 'PyTorch' })
  await expect(card).toContainText('Usable')

  await card.getByLabel('Device').selectOption('cuda:0')
  await card.getByLabel('Memory policy').selectOption('pytorch_fraction')
  await card.getByRole('spinbutton', { name: 'Fraction' }).fill('0.25')

  await expect(card.getByLabel('Device')).toHaveValue('cuda:0')
  await expect(card.getByLabel('Memory policy')).toHaveValue('pytorch_fraction')
  await expect(card.getByRole('spinbutton', { name: 'Fraction' })).toHaveValue('0.25')
  await expect.poll(async () => page.evaluate((key) => {
    const value = window.localStorage.getItem(key)
    return value ? JSON.parse(value).pytorch : null
  }, runtimePreferencesStorageKey)).toEqual({
    api: 'keras',
    backend: 'pytorch',
    device_id: 'cuda:0',
    memory_policy: { kind: 'pytorch_fraction', fraction: 0.25 },
    precision: 'float32',
  })

  await page.reload()
  const reloadedCard = page.locator('.backend-status-card-container').filter({ hasText: 'PyTorch' })
  await expect(reloadedCard.getByLabel('Device')).toHaveValue('cuda:0')
  await expect(reloadedCard.getByLabel('Memory policy')).toHaveValue('pytorch_fraction')
  await expect(reloadedCard.getByRole('spinbutton', { name: 'Fraction' })).toHaveValue('0.25')
})

test('restores the last active backend and API with its device and memory policy', async ({ page }) => {
  const pageErrors: string[] = []
  page.on('pageerror', (error) => pageErrors.push(error.message))

  await page.goto('/environment')
  const card = page.locator('.backend-status-card-container').filter({ hasText: 'TensorFlow' })
  await expect(card).toContainText('Usable')
  await card.locator('.backend-status-card').click()
  await page.locator('.runtime-api-option').filter({ hasText: 'Native framework' }).click()
  await card.getByLabel('Device').selectOption('gpu:0')
  await expect(card.getByLabel('Memory policy')).toHaveValue('tensorflow_growth')
  await card.getByLabel('Memory policy').selectOption('tensorflow_limit_mb')
  await expect(card.getByLabel('Memory policy')).toHaveValue('tensorflow_limit_mb')
  await card.getByRole('spinbutton', { name: 'Limit (MB)' }).fill('2048')

  const expectedPreference = {
    api: 'native',
    backend: 'tensorflow',
    device_id: 'gpu:0',
    memory_policy: { kind: 'tensorflow_limit_mb', limit_mb: 2048 },
    precision: 'float32',
  }
  await expect.poll(async () => page.evaluate(({ preferencesKey, activeBackendKey }) => ({
    preference: JSON.parse(window.localStorage.getItem(preferencesKey) ?? '{}').tensorflow ?? null,
    activeBackend: window.localStorage.getItem(activeBackendKey),
  }), {
    preferencesKey: runtimePreferencesStorageKey,
    activeBackendKey: runtimePreferenceBackendStorageKey,
  })).toEqual({ preference: expectedPreference, activeBackend: 'tensorflow' })

  await page.reload()
  const reloadedCard = page.locator('.backend-status-card-container').filter({ hasText: 'TensorFlow' })
  await expect(reloadedCard.locator('.backend-status-card')).toHaveAttribute('aria-pressed', 'true')
  await expect(page.locator('.runtime-api-option').filter({ hasText: 'Native framework' })).toHaveClass(/selected/)
  await expect(reloadedCard.getByLabel('Device')).toHaveValue('gpu:0')
  await expect(reloadedCard.getByLabel('Memory policy')).toHaveValue('tensorflow_limit_mb')
  await expect(reloadedCard.getByRole('spinbutton', { name: 'Limit (MB)' })).toHaveValue('2048')
  expect(pageErrors).toEqual([])
  await saveEvidenceScreenshot(page, 'nw-t1-04-active-preference-reload.png')
})

test('reconciles a stored device that is no longer available', async ({ page }) => {
  await page.addInitScript(({ preferencesKey, activeBackendKey }) => {
    window.localStorage.setItem(preferencesKey, JSON.stringify({
      tensorflow: {
        api: 'native',
        backend: 'tensorflow',
        device_id: 'retired-gpu',
        memory_policy: { kind: 'tensorflow_limit_mb', limit_mb: 2048 },
        precision: 'float32',
      },
    }))
    window.localStorage.setItem(activeBackendKey, 'tensorflow')
  }, { preferencesKey: runtimePreferencesStorageKey, activeBackendKey: runtimePreferenceBackendStorageKey })

  await page.goto('/environment')
  const card = page.locator('.backend-status-card-container').filter({ hasText: 'TensorFlow' })
  await expect(card.locator('.backend-status-card')).toHaveAttribute('aria-pressed', 'true')
  await expect(card.getByLabel('Device')).toHaveValue('cpu')
  await expect(card.getByLabel('Memory policy')).toHaveValue('default')
  await expect.poll(() => page.evaluate((key) => {
    const value = window.localStorage.getItem(key)
    return value ? JSON.parse(value).tensorflow : null
  }, runtimePreferencesStorageKey)).toEqual({
    api: 'keras',
    backend: 'tensorflow',
    device_id: 'cpu',
    memory_policy: { kind: 'default' },
    precision: 'float32',
  })
})

test('falls back to the preferred available backend when the active backend disappears', async ({ page }) => {
  await page.addInitScript(({ preferencesKey, activeBackendKey }) => {
    window.localStorage.setItem(preferencesKey, JSON.stringify({
      tensorflow: {
        api: 'native',
        backend: 'tensorflow',
        device_id: 'gpu:0',
        memory_policy: { kind: 'tensorflow_limit_mb', limit_mb: 2048 },
        precision: 'float32',
      },
    }))
    window.localStorage.setItem(activeBackendKey, 'tensorflow')
  }, { preferencesKey: runtimePreferencesStorageKey, activeBackendKey: runtimePreferenceBackendStorageKey })

  await page.goto('/environment?unavailable_backend=tensorflow')
  const pytorchCard = page.locator('.backend-status-card-container').filter({ hasText: 'PyTorch' })
  await expect(pytorchCard.locator('.backend-status-card')).toHaveAttribute('aria-pressed', 'true')
  await expect(page.locator('.runtime-api-option').filter({ hasText: 'Keras' })).toHaveClass(/selected/)
  await expect.poll(() => page.evaluate(({ preferencesKey, activeBackendKey }) => ({
    preferences: JSON.parse(window.localStorage.getItem(preferencesKey) ?? '{}'),
    activeBackend: window.localStorage.getItem(activeBackendKey),
  }), {
    preferencesKey: runtimePreferencesStorageKey,
    activeBackendKey: runtimePreferenceBackendStorageKey,
  })).toEqual({
    preferences: {
      pytorch: {
        api: 'keras',
        backend: 'pytorch',
        device_id: 'cpu',
        memory_policy: { kind: 'default' },
        precision: 'float32',
      },
    },
    activeBackend: 'pytorch',
  })
})

test('reconciles malformed nested preferences to a valid runtime default', async ({ page }) => {
  const pageErrors: string[] = []
  page.on('pageerror', (error) => pageErrors.push(error.message))
  await page.addInitScript(({ preferencesKey, activeBackendKey }) => {
    window.localStorage.setItem(preferencesKey, JSON.stringify({
      tensorflow: {
        api: 'native',
        backend: 'tensorflow',
        device_id: 'gpu:0',
        memory_policy: null,
        precision: 'float32',
      },
    }))
    window.localStorage.setItem(activeBackendKey, 'tensorflow')
  }, { preferencesKey: runtimePreferencesStorageKey, activeBackendKey: runtimePreferenceBackendStorageKey })

  await page.goto('/environment')
  const card = page.locator('.backend-status-card-container').filter({ hasText: 'TensorFlow' })
  await expect(card.locator('.backend-status-card')).toHaveAttribute('aria-pressed', 'true')
  await expect(card.getByLabel('Device')).toHaveValue('cpu')
  await expect(card.getByLabel('Memory policy')).toHaveValue('default')
  await expect.poll(() => page.evaluate((key) => {
    const value = window.localStorage.getItem(key)
    return value ? JSON.parse(value).tensorflow : null
  }, runtimePreferencesStorageKey)).toEqual({
    api: 'keras',
    backend: 'tensorflow',
    device_id: 'cpu',
    memory_policy: { kind: 'default' },
    precision: 'float32',
  })
  expect(pageErrors).toEqual([])
})

test('keeps runtime controls usable when browser storage reads and writes fail', async ({ page }) => {
  await page.addInitScript((keys) => {
    const blocked = new Set(keys)
    const getItem = Storage.prototype.getItem
    const setItem = Storage.prototype.setItem
    Storage.prototype.getItem = function (key: string): string | null {
      if (blocked.has(key)) throw new DOMException('Storage is disabled.', 'SecurityError')
      return getItem.call(this, key)
    }
    Storage.prototype.setItem = function (key: string, value: string): void {
      if (blocked.has(key)) throw new DOMException('Storage is disabled.', 'SecurityError')
      setItem.call(this, key, value)
    }
  }, [runtimePreferencesStorageKey, runtimePreferenceBackendStorageKey])
  const pageErrors: string[] = []
  page.on('pageerror', (error) => pageErrors.push(error.message))
  await page.goto('/environment')
  const usableCard = page.locator('.backend-status-card-container').filter({ hasText: 'TensorFlow' })
  await expect(usableCard).toContainText('Usable')
  await usableCard.locator('.backend-status-card').click()
  await page.locator('.runtime-api-option').filter({ hasText: 'Native framework' }).click()
  await expect(usableCard.locator('.backend-status-card')).toHaveAttribute('aria-pressed', 'true')
  await expect(page.locator('.runtime-api-option').filter({ hasText: 'Native framework' })).toHaveClass(/selected/)
  await expect(usableCard.getByLabel('Device')).toHaveValue('cpu')
  expect(pageErrors).toEqual([])
})
