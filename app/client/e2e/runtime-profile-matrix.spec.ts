// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { createHash } from 'node:crypto'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

type Backend = 'tensorflow' | 'pytorch' | 'jax'
type Api = 'keras' | 'native'

type ProfileCase = {
  id: string
  backend: Backend
  api: Api
  label: string
}

type RuntimeApiStatus = {
  id: Api
  available: boolean
  reason_code: string | null
  state: string
  source: string
  checked_at: string | null
}

type RuntimeDevice = {
  device_id: string
  name: string
  available: boolean
  accelerator: string
  supported_memory_policies: string[]
}

type BackendStatus = {
  id: Backend
  installed: boolean
  version: string | null
  available: boolean
  accelerator: string | null
  device_name: string | null
  devices: RuntimeDevice[]
  runtime_apis: RuntimeApiStatus[]
  native_compilers: Record<string, boolean>
  discovery_state: string
  discovery_source: string
  last_checked_at: string | null
}

type MetadataResponse = {
  execution_backends: BackendStatus[]
}

type CatalogResponse = {
  schema_version: number
  nodes: unknown[]
}

type ProjectDocument = {
  id: string
  current_revision: number
  execution_backend: Backend
  keras_enabled: boolean
  graph: {
    nodes: unknown[]
    edges: unknown[]
  }
}

type JobResponse = {
  id: string
  state: string
  operation: string
  worker: { api: Api; backend: Backend; device_id: string }
  artifact: unknown
  result: {
    status: string
    inputs: Array<{ node_id: string; shape: number[]; dtype: string }>
    outputs: Array<{ node_id: string; shape: number[]; dtype: string }>
    parameter_count: number
    training?: string
    compilation?: { verified?: boolean; training?: string }
  } | null
}

const profiles: ProfileCase[] = [
  { id: 'T4-01-A', backend: 'tensorflow', api: 'keras', label: 'Keras / TensorFlow' },
  { id: 'T4-01-B', backend: 'pytorch', api: 'keras', label: 'Keras / PyTorch' },
  { id: 'T4-01-C', backend: 'jax', api: 'keras', label: 'Keras / JAX' },
  { id: 'T4-01-D', backend: 'tensorflow', api: 'native', label: 'Native TensorFlow' },
  { id: 'T4-01-E', backend: 'pytorch', api: 'native', label: 'Native PyTorch' },
  { id: 'T4-01-F', backend: 'jax', api: 'native', label: 'Native JAX' },
]

const backendLabels: Record<Backend, string> = {
  tensorflow: 'TensorFlow',
  pytorch: 'PyTorch',
  jax: 'JAX / Flax',
}

const environmentBackendLabels: Record<Backend, string> = {
  tensorflow: 'TensorFlow',
  pytorch: 'PyTorch',
  jax: 'JAX',
}

const evidenceDir = path.resolve(
  process.env['NW_VALIDATION_EVIDENCE_DIR'] ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260930', 'evidence', 'nw-t4-01'),
)

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map((item) => stableValue(item))
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>
    return Object.fromEntries(Object.keys(record).sort().map((key) => [key, stableValue(record[key])]))
  }
  return value
}

function computeCatalogFingerprint(catalog: CatalogResponse): string {
  return createHash('sha256').update(JSON.stringify(stableValue(catalog))).digest('hex')
}

function smokeGraph() {
  return {
    schema_version: 5,
    nodes: [
      {
        id: 't4-input',
        definition_id: 'input.tabular',
        definition_version: 1,
        position: { x: 48, y: 144 },
        parameters: { shape: [4], dtype: 'float32' },
      },
      {
        id: 't4-dense',
        definition_id: 'layer.dense',
        definition_version: 1,
        position: { x: 352, y: 144 },
        parameters: { units: 1, activation: 'sigmoid', use_bias: true },
      },
      {
        id: 't4-output',
        definition_id: 'output.classification',
        definition_version: 1,
        position: { x: 656, y: 144 },
        parameters: {},
      },
    ],
    edges: [
      { id: 't4-edge-1', source_node_id: 't4-input', source_port_id: 'output', target_node_id: 't4-dense', target_port_id: 'input' },
      { id: 't4-edge-2', source_node_id: 't4-dense', source_port_id: 'output', target_node_id: 't4-output', target_port_id: 'input' },
    ],
    model_configuration: {},
    export_settings: {},
    future_training_configuration: {},
  }
}

async function waitForDiscovery(request: APIRequestContext, profile: ProfileCase): Promise<{ terminal: RuntimeApiStatus & { backend: Backend; api: Api; available: boolean }; backend: BackendStatus }> {
  const started = await request.post('/api/execution/discovery', {
    data: { backend: profile.backend, api: profile.api, force_refresh: true },
  })
  expect(started.status()).toBe(202)

  let terminal: (RuntimeApiStatus & { backend: Backend; api: Api; available: boolean }) | null = null
  await expect.poll(async () => {
    const response = await request.get(`/api/execution/discovery?backend=${profile.backend}&api=${profile.api}`)
    const value = await response.json() as RuntimeApiStatus & { backend: Backend; api: Api; available: boolean }
    if (value.state !== 'checking') terminal = value
    return value.state
  }, { timeout: 120_000, intervals: [250, 500, 1_000] }).not.toBe('checking')

  if (!terminal) throw new Error(`Discovery did not reach a terminal state for ${profile.label}`)
  expect(terminal).toMatchObject({ backend: profile.backend, api: profile.api, state: 'verified', available: true, source: 'live' })

  const metadataResponse = await request.get('/api/metadata')
  expect(metadataResponse.ok()).toBeTruthy()
  const metadata = await metadataResponse.json() as MetadataResponse
  const backend = metadata.execution_backends.find((item) => item.id === profile.backend)
  if (!backend) throw new Error(`Metadata omitted ${profile.backend}`)
  const api = backend.runtime_apis.find((item) => item.id === profile.api)
  expect(api).toMatchObject({ id: profile.api, available: true, state: 'verified', source: 'live' })
  expect(backend.version).toBeTruthy()
  expect(backend.last_checked_at).toBeTruthy()
  expect(backend.devices.some((device) => device.device_id === 'cpu' && device.available)).toBeTruthy()
  return { terminal, backend }
}

async function createProject(request: APIRequestContext, profile: ProfileCase): Promise<ProjectDocument> {
  const response = await request.post('/api/projects', {
    data: {
      name: `NW-T4-01 ${profile.id} ${Date.now()}`,
      description: `Fresh six-profile runtime matrix: ${profile.label}`,
      execution_backend: 'pytorch',
      keras_enabled: true,
      graph: smokeGraph(),
    },
  })
  expect(response.status()).toBe(201)
  return response.json() as Promise<ProjectDocument>
}

async function deleteProject(request: APIRequestContext, projectId: string): Promise<void> {
  const response = await request.delete(`/api/projects/${projectId}`)
  expect([204, 404]).toContain(response.status())
}

async function selectEnvironmentProfile(page: Page, profile: ProfileCase): Promise<void> {
  await page.goto('/environment')
  const card = page.locator('.backend-status-card-container').filter({ hasText: environmentBackendLabels[profile.backend] })
  await expect(card).toContainText('Usable', { timeout: 75_000 })
  await card.locator('.backend-status-card').click()
  const apiLabel = profile.api === 'keras' ? 'Keras' : 'Native framework'
  const apiOption = page.locator('.runtime-api-option').filter({ hasText: apiLabel })
  await expect(apiOption).toBeVisible()
  await apiOption.click()
  await card.getByLabel('Device').selectOption('cpu')
  await expect(card.locator('.backend-status-card')).toHaveAttribute('aria-pressed', 'true')
  await expect(apiOption).toHaveClass(/selected/)
  await expect(card.getByLabel('Device')).toHaveValue('cpu')
  await page.screenshot({ path: path.join(evidenceDir, `${profile.id.toLowerCase()}-environment-profile.png`), fullPage: true })
}

async function selectBuilderProfile(page: Page, profile: ProfileCase): Promise<void> {
  const backend = page.getByRole('combobox', { name: 'Execution backend' })
  await backend.click()
  await page.getByRole('option', { name: backendLabels[profile.backend], exact: true }).click()
  const keras = page.getByLabel('Enable Keras', { exact: true })
  if (profile.api === 'keras') {
    if (await keras.isChecked()) {
      await keras.uncheck()
    }
    await keras.check()
  }
  else await keras.uncheck()
  await expect(backend).toHaveAttribute('data-value', profile.backend)
  await expect(keras).toBeChecked({ checked: profile.api === 'keras' })
}

test.describe('NW-T4-01 fresh serial runtime profile matrix', () => {
  test.describe.configure({ mode: 'serial' })

  test('executes all six exact Keras/native profiles with live discovery and reload', async ({ page, request }) => {
    test.setTimeout(900_000)
    await mkdir(evidenceDir, { recursive: true })
    const catalogResponse = await request.get('/api/nodes/catalog')
    expect(catalogResponse.ok()).toBeTruthy()
    const catalog = await catalogResponse.json() as CatalogResponse
    const catalogFingerprint = computeCatalogFingerprint(catalog)
    expect(catalog.nodes).toHaveLength(149)

    const matrix: Array<Record<string, unknown>> = []
    for (const profile of profiles) {
      const runtime = await waitForDiscovery(request, profile)
      const fixture = await createProject(request, profile)
      const initialProject = await (await request.get(`/api/projects/${fixture.id}`)).json() as ProjectDocument
      const expectedApi = profile.api
      try {
        await selectEnvironmentProfile(page, profile)
        await page.goto(`/builder/${fixture.id}`)
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 75_000 })
        await selectBuilderProfile(page, profile)

        const saveResponse = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT')
        await page.getByRole('button', { name: 'Save model', exact: true }).click()
        const saved = await (await saveResponse).json() as ProjectDocument
        expect(saved.execution_backend).toBe(profile.backend)
        expect(saved.keras_enabled).toBe(profile.api === 'keras')
        const persisted = await (await request.get(`/api/projects/${fixture.id}`)).json() as ProjectDocument
        expect(persisted).toMatchObject({ execution_backend: profile.backend, keras_enabled: profile.api === 'keras' })

        const validationResponse = page.waitForResponse((response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST')
        await page.getByRole('button', { name: 'Validate', exact: true }).click()
        const validation = await (await validationResponse).json() as { valid: boolean; diagnostics: unknown[] }
        expect(validation.valid).toBeTruthy()
        await expect(page.locator('.validation-panel')).toHaveAttribute('data-validation-state', 'ready')
        await expect(page.locator('.validation-panel .status-indicator.valid')).toBeVisible()
        await page.screenshot({ path: path.join(evidenceDir, `${profile.id.toLowerCase()}-builder-profile.png`), fullPage: true })

        await page.reload()
        await expect(page.getByRole('combobox', { name: 'Execution backend' })).toHaveAttribute('data-value', profile.backend)
        await expect(page.getByLabel('Enable Keras', { exact: true })).toBeChecked({ checked: profile.api === 'keras' })

        await page.goto(`/models/${fixture.id}/validation`)
        await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 75_000 })
        await expect(page.getByText(new RegExp(`${expectedApi} · ${profile.backend} · cpu`, 'i'))).toBeVisible({ timeout: 120_000 })
        const submitResponse = page.waitForResponse((response) => response.url().endsWith('/api/execution/jobs') && response.request().method() === 'POST')
        await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
        const submitted = await (await submitResponse).json() as JobResponse
        expect(submitted).toMatchObject({ operation: 'smoke_inference', worker: { api: profile.api, backend: profile.backend, device_id: 'cpu' } })
        await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 180_000 })
        await expect.poll(async () => {
          const response = await request.get(`/api/execution/jobs/${submitted.id}`)
          return response.ok() ? String((await response.json() as JobResponse).state) : 'missing'
        }, { timeout: 180_000, intervals: [250, 500, 1_000] }).toBe('succeeded')
        const completed = await (await request.get(`/api/execution/jobs/${submitted.id}`)).json() as JobResponse
        expect(completed).toMatchObject({
          state: 'succeeded',
          operation: 'smoke_inference',
          worker: { api: profile.api, backend: profile.backend, device_id: 'cpu' },
          artifact: null,
          result: {
            status: 'succeeded',
            inputs: [{ node_id: 't4-input', shape: [1, 4], dtype: 'float32' }],
            outputs: [{ node_id: 't4-output', shape: [1, 1], dtype: 'float32' }],
            parameter_count: 5,
          },
        })
        expect(completed.result?.compilation).toMatchObject({ verified: true, training: 'not_run' })
        if (profile.api === 'native') expect(completed.result?.training).toBe('not_run')
        else expect(completed.result?.training).toBeUndefined()
        const afterJob = await (await request.get(`/api/projects/${fixture.id}`)).json() as ProjectDocument
        expect(afterJob.current_revision).toBe(saved.current_revision)
        expect(afterJob.graph).toEqual(persisted.graph)

        await page.reload()
        await expect(page.getByText('succeeded', { exact: true })).toBeVisible({ timeout: 45_000 })
        await expect(page.getByText(/Completed on/)).toBeVisible()
        await page.screenshot({ path: path.join(evidenceDir, `${profile.id.toLowerCase()}-functional-validation.png`), fullPage: true })

        matrix.push({
          profile,
          discovery: runtime,
          project: { id: fixture.id, initial: initialProject, saved, reloaded: persisted, after_job: afterJob },
          validation,
          job: completed,
          catalog: { schema_version: catalog.schema_version, count: catalog.nodes.length, fingerprint_sha256: catalogFingerprint },
          result: 'PASS',
        })
        await writeFile(path.join(evidenceDir, `${profile.id.toLowerCase()}-receipt.json`), JSON.stringify(matrix[matrix.length - 1], null, 2), 'utf8')
      } finally {
        await deleteProject(request, fixture.id)
      }
    }

    await writeFile(path.join(evidenceDir, 'nw-t4-01-six-profile-matrix.json'), JSON.stringify({ catalog: { schema_version: catalog.schema_version, count: catalog.nodes.length, fingerprint_sha256: catalogFingerprint }, profiles: matrix }, null, 2), 'utf8')
    expect(matrix).toHaveLength(6)
  })
})
