// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

type Backend = 'tensorflow' | 'pytorch' | 'jax'

type NativeCompilerConfiguration = {
  compiler: { id: string; parameters: Record<string, unknown> }
}

type ProjectDocument = {
  id: string
  current_revision: number
  execution_backend: Backend
  keras_enabled: boolean
  graph: {
    nodes: unknown[]
    edges: unknown[]
    model_configuration: { native: Record<Backend, NativeCompilerConfiguration> }
  }
}

type JobResponse = {
  id: string
  state: string
  operation: string
  worker: { api: 'native'; backend: Backend; device_id: string }
  artifact: unknown
  result: {
    status: string
    inputs: Array<{ node_id: string; shape: number[]; dtype: string }>
    outputs: Array<{ node_id: string; shape: number[]; dtype: string }>
    parameter_count: number
    training: string
    compilation: { verified?: boolean; training?: string }
  } | null
}

const backends: Backend[] = ['tensorflow', 'jax', 'pytorch']
const backendLabels: Record<Backend, string> = {
  tensorflow: 'TensorFlow',
  pytorch: 'PyTorch',
  jax: 'JAX / Flax',
}
const compilerIds: Record<Backend, string> = {
  tensorflow: 'tensorflow_function',
  pytorch: 'pytorch_compile',
  jax: 'jax_jit',
}
const compilerParameters: Record<Backend, string> = {
  tensorflow: 'jit_compile',
  pytorch: 'enabled',
  jax: 'enabled',
}
const compilerLabels: Record<Backend, string> = {
  tensorflow: 'XLA JIT compilation',
  pytorch: 'Enable torch.compile',
  jax: 'Enable jax.jit',
}
const evidenceDir = path.resolve(
  process.env['NW_VALIDATION_EVIDENCE_DIR'] ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260930', 'evidence', 'nw-t4-02'),
)

function smokeGraph() {
  return {
    schema_version: 5,
    nodes: [
      {
        id: 't4-compiler-input',
        definition_id: 'input.tabular',
        definition_version: 1,
        position: { x: 48, y: 144 },
        parameters: { shape: [4], dtype: 'float32' },
      },
      {
        id: 't4-compiler-dense',
        definition_id: 'layer.dense',
        definition_version: 1,
        position: { x: 352, y: 144 },
        parameters: { units: 1, activation: 'sigmoid', use_bias: true },
      },
      {
        id: 't4-compiler-output',
        definition_id: 'output.classification',
        definition_version: 1,
        position: { x: 656, y: 144 },
        parameters: {},
      },
    ],
    edges: [
      { id: 't4-compiler-edge-1', source_node_id: 't4-compiler-input', source_port_id: 'output', target_node_id: 't4-compiler-dense', target_port_id: 'input' },
      { id: 't4-compiler-edge-2', source_node_id: 't4-compiler-dense', source_port_id: 'output', target_node_id: 't4-compiler-output', target_port_id: 'input' },
    ],
    model_configuration: {},
    export_settings: {},
    future_training_configuration: {},
  }
}

async function createProject(request: APIRequestContext, backend: Backend): Promise<ProjectDocument> {
  const response = await request.post('/api/projects', {
    data: {
      name: `NW-T4-02 ${backend} compiler ${Date.now()}`,
      description: `Native compiler worker gate: ${backend}`,
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

async function selectNativeProfile(page: Page, fixture: ProjectDocument, backend: Backend): Promise<ProjectDocument> {
  await page.goto(`/builder/${fixture.id}`)
  const backendSelect = page.getByRole('combobox', { name: 'Execution backend' })
  await backendSelect.click()
  await page.getByRole('option', { name: backendLabels[backend], exact: true }).click()
  await page.getByLabel('Enable Keras', { exact: true }).uncheck()
  await expect(backendSelect).toHaveAttribute('data-value', backend)
  const saveResponse = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT')
  await page.getByRole('button', { name: 'Save model', exact: true }).click()
  const saved = await (await saveResponse).json() as ProjectDocument
  expect(saved).toMatchObject({ execution_backend: backend, keras_enabled: false })
  return saved
}

async function enableCompiler(page: Page, request: APIRequestContext, fixture: ProjectDocument, backend: Backend): Promise<void> {
  await page.goto(`/models/${fixture.id}/compilation`)
  await page.getByRole('button', { name: 'Configure', exact: true }).click()
  const dialog = page.getByRole('dialog')
  await expect(dialog).toBeVisible({ timeout: 75_000 })
  await dialog.getByRole('tab', { name: /Compiler/ }).click()
  const toggle = dialog.locator('.compilation-parameter-grid label').filter({ hasText: compilerLabels[backend] }).locator('input[type="checkbox"]')
  await expect(toggle).toHaveCount(1)
  await expect(toggle).toBeEnabled()
  await toggle.check()
  await dialog.getByRole('button', { name: 'Apply configuration', exact: true }).click()
  await expect(page.getByRole('button', { name: /Saved validated compilation settings/ })).toBeVisible({ timeout: 75_000 })
  const project = await (await request.get(`/api/projects/${fixture.id}`)).json() as ProjectDocument
  const configuration = project.graph.model_configuration.native[backend].compiler
  expect(configuration).toMatchObject({ id: compilerIds[backend], parameters: { [compilerParameters[backend]]: true } })
  await page.screenshot({ path: path.join(evidenceDir, `nw-t4-02-${backend}-compiler-enabled.png`), fullPage: true })
}

async function runWorkerValidation(page: Page, request: APIRequestContext, fixture: ProjectDocument, backend: Backend): Promise<JobResponse> {
  await page.goto(`/models/${fixture.id}/validation`)
  await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 75_000 })
  await expect(page.getByText(new RegExp(`native · ${backend} · cpu`, 'i'))).toBeVisible({ timeout: 75_000 })
  const submitResponse = page.waitForResponse((response) => response.url().endsWith('/api/execution/jobs') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
  const submitted = await (await submitResponse).json() as JobResponse
  expect(submitted).toMatchObject({ operation: 'smoke_inference', worker: { api: 'native', backend, device_id: 'cpu' } })
  await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 240_000 })
  await expect.poll(async () => {
    const response = await request.get(`/api/execution/jobs/${submitted.id}`)
    return response.ok() ? String((await response.json() as JobResponse).state) : 'missing'
  }, { timeout: 240_000, intervals: [500, 1_000, 2_000] }).toBe('succeeded')
  const completed = await (await request.get(`/api/execution/jobs/${submitted.id}`)).json() as JobResponse
  expect(completed).toMatchObject({
    state: 'succeeded',
    operation: 'smoke_inference',
    worker: { api: 'native', backend, device_id: 'cpu' },
    artifact: null,
    result: {
      status: 'succeeded',
      inputs: [{ node_id: 't4-compiler-input', shape: [1, 4], dtype: 'float32' }],
      outputs: [{ node_id: 't4-compiler-output', shape: [1, 1], dtype: 'float32' }],
      parameter_count: 5,
      training: 'not_run',
      compilation: { verified: true, training: 'not_run' },
    },
  })
  await page.reload()
  await expect(page.getByText('succeeded', { exact: true })).toBeVisible({ timeout: 45_000 })
  await expect(page.getByText(/Completed on/)).toBeVisible()
  await page.screenshot({ path: path.join(evidenceDir, `nw-t4-02-${backend}-compiler-worker.png`), fullPage: true })
  return completed
}

test.describe('NW-T4-02 native compiler project-worker gates', () => {
  test.describe.configure({ mode: 'serial' })

  test('refreshes XLA/JIT and proves the supported PyTorch Inductor worker path', async ({ page, request }) => {
    test.setTimeout(900_000)
    await mkdir(evidenceDir, { recursive: true })
    const metadataResponse = await request.get('/api/metadata')
    expect(metadataResponse.ok()).toBeTruthy()
    const metadata = await metadataResponse.json() as { configuration: { compilation_capabilities: { native: Record<Backend, { available: boolean }> } } }
    for (const backend of backends) expect(metadata.configuration.compilation_capabilities.native[backend].available).toBeTruthy()

    const receipts: Array<Record<string, unknown>> = []
    for (const backend of backends) {
      const fixture = await createProject(request, backend)
      try {
        const saved = await selectNativeProfile(page, fixture, backend)
        await enableCompiler(page, request, fixture, backend)
        const completed = await runWorkerValidation(page, request, fixture, backend)
        const afterJob = await (await request.get(`/api/projects/${fixture.id}`)).json() as ProjectDocument
        expect(afterJob.current_revision).toBeGreaterThanOrEqual(saved.current_revision)
        receipts.push({
          backend,
          compiler_id: compilerIds[backend],
          project_id: fixture.id,
          saved_revision: saved.current_revision,
          after_job_revision: afterJob.current_revision,
          job: completed,
          export_validation: 'not_run_by_design',
          result: 'PASS',
        })
        await writeFile(path.join(evidenceDir, `nw-t4-02-${backend}-receipt.json`), JSON.stringify(receipts[receipts.length - 1], null, 2), 'utf8')
      } finally {
        await deleteProject(request, fixture.id)
      }
    }
    await writeFile(path.join(evidenceDir, 'nw-t4-02-native-compiler-worker-matrix.json'), JSON.stringify({ backends, receipts }, null, 2), 'utf8')
    expect(receipts).toHaveLength(backends.length)
  })
})
