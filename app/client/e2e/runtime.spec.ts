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
}

const runtimePreferencesStorageKey = 'neuroweave.runtime-preferences.v1'

test.beforeEach(async ({ page }) => {
  await page.route('**/api/metadata', async (route) => {
    const upstream = await route.fetch()
    const metadata = await upstream.json() as MetadataResponse
    const pytorch = metadata.execution_backends.find((item) => item.id === 'pytorch')
    if (!pytorch) throw new Error('The runtime metadata did not include PyTorch.')

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
