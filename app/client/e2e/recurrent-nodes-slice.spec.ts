// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

type CreatedProject = { id: string; current_revision: number; name: string }
type GraphNode = {
  id: string
  definition_id: string
  definition_version: number
  label: string
  position: { x: number; y: number }
  parameters: Record<string, unknown>
}
type ShapeSummary = {
  node_id: string
  parameter_count: number
  outputs: Record<string, { shape: Array<number | null> }>
}
type ValidationPayload = {
  valid: boolean
  diagnostics: unknown[]
  shape_summaries: ShapeSummary[]
  total_parameters: number
}
type RuntimeJob = {
  state: string
  worker?: { api: string; backend: string; device_id: string }
  result?: {
    status: string
    outputs: Array<{ shape: number[] }>
    parameter_count: number
  }
}
type NativeCase = {
  name: string
  catalogNode: string
  definitionId: string
  parameters: Record<string, unknown>
  staticOutputShape: Array<number | null>
  runtimeOutputShape: number[]
  staticParameterCount: number
  runtimeParameterCount: number
}

const evidenceDir = path.resolve(
  process.env.NW_VALIDATION_EVIDENCE_DIR
    ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260928', 'evidence', 'nw-t3-05-recurrent'),
)

function graphShell(nodes: GraphNode[]) {
  return {
    schema_version: 5,
    nodes,
    edges: [],
    model_configuration: {
      keras: {
        optimizer: 'adam',
        optimizer_parameters: { learning_rate: 0.001 },
        losses: {},
        metrics: {},
        default_initializer: 'glorot_uniform',
        default_regularizer: 'none',
        default_constraint: 'none',
        mixed_precision: false,
        random_seed: 42,
      },
      native: {
        tensorflow: { jit_compile: false },
        pytorch: {
          optimizer: 'adamw',
          optimizer_parameters: {
            learning_rate: 0.0005,
            weight_decay: 0.004,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-7,
          },
          losses: {},
          metrics: {},
          compile_enabled: false,
          mode: 'default',
          fullgraph: false,
          dynamic: false,
        },
        jax: { jit_enabled: false },
      },
    },
    export_settings: {
      include_comments: true,
      include_shape_assertions: true,
      preferred_binary_format: 'keras',
    },
    future_training_configuration: {},
  }
}

function node(
  id: string,
  definitionId: string,
  label: string,
  position: { x: number; y: number },
  parameters: Record<string, unknown>,
): GraphNode {
  return {
    id,
    definition_id: definitionId,
    definition_version: 1,
    label,
    position,
    parameters,
  }
}

async function createProject(
  request: APIRequestContext,
  name: string,
  modelGraph: ReturnType<typeof graphShell>,
): Promise<CreatedProject> {
  const response = await request.post('/api/projects', {
    data: {
      name,
      description: 'NW-T3-05 recurrent catalog validation fixture',
      execution_backend: 'pytorch',
      keras_enabled: false,
      graph: modelGraph,
    },
  })
  expect(response.ok()).toBeTruthy()
  return response.json() as Promise<CreatedProject>
}

async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
  const response = await request.delete(`/api/projects/${id}`)
  expect([204, 404]).toContain(response.status())
}

async function addCatalogNode(page: Page, displayName: string): Promise<string> {
  const nodes = page.locator('[data-node-id]')
  const before = new Set(
    await nodes.evaluateAll((elements) => elements
      .map((element) => element.getAttribute('data-node-id'))
      .filter(Boolean) as string[]),
  )
  await page.getByRole('textbox', { name: 'Search nodes' }).fill(displayName)
  const item = page.locator('.catalog-item').filter({ hasText: displayName }).first()
  await expect(item, `Catalog node ${displayName} should be available`).toBeEnabled()
  await item.click()
  await expect(nodes).toHaveCount(before.size + 1)
  const after = await nodes.evaluateAll((elements) => elements
    .map((element) => element.getAttribute('data-node-id'))
    .filter(Boolean) as string[])
  const created = after.find((nodeId) => !before.has(nodeId))
  expect(created, `Builder should materialize ${displayName}`).toBeTruthy()
  await page.getByRole('textbox', { name: 'Search nodes' }).fill('')
  return created!
}

async function connectByPorts(
  page: Page,
  sourceLabel: string,
  targetLabel: string,
  expectedConnectionCount: number,
): Promise<void> {
  const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
  const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
  const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
  const target = targetNode.getByRole('button', { name: /^Port Input \(input\)/ })
  await expect(source).toBeVisible()
  await expect(target).toBeVisible()
  await source.press('Enter')
  await expect(source).toHaveAttribute('aria-pressed', 'true')
  const responsePromise = page.waitForResponse(
    (response) => response.url().endsWith('/api/validation/connection') && response.request().method() === 'POST',
  )
  await target.press('Enter')
  const response = await responsePromise
  expect(response.ok()).toBeTruthy()
  const payload = await response.json() as { valid: boolean }
  expect(payload.valid, `${sourceLabel} → ${targetLabel} should be accepted`).toBe(true)
  await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
}

const nativeCases: NativeCase[] = [
  {
    name: 'nw-t3-05-simple-rnn-native-pytorch',
    catalogNode: 'Simple RNN',
    definitionId: 'recurrent.simple_rnn',
    parameters: { units: 4, return_sequences: true },
    staticOutputShape: [null, 4, 4],
    runtimeOutputShape: [1, 4, 4],
    staticParameterCount: 32,
    runtimeParameterCount: 36,
  },
  {
    name: 'nw-t3-05-gru-native-pytorch',
    catalogNode: 'GRU',
    definitionId: 'recurrent.gru',
    parameters: { units: 4, return_sequences: true },
    staticOutputShape: [null, 4, 4],
    runtimeOutputShape: [1, 4, 4],
    staticParameterCount: 108,
    runtimeParameterCount: 108,
  },
  {
    name: 'nw-t3-05-lstm-native-pytorch',
    catalogNode: 'LSTM',
    definitionId: 'recurrent.lstm',
    parameters: { units: 4, return_sequences: true },
    staticOutputShape: [null, 4, 4],
    runtimeOutputShape: [1, 4, 4],
    staticParameterCount: 128,
    runtimeParameterCount: 144,
  },
  {
    name: 'nw-t3-05-bidirectional-native-pytorch',
    catalogNode: 'Bidirectional Recurrent Layer',
    definitionId: 'recurrent.bidirectional',
    parameters: { cell: 'simple_rnn', units: 4, return_sequences: true, merge_mode: 'concat' },
    staticOutputShape: [null, 4, 8],
    runtimeOutputShape: [1, 4, 8],
    staticParameterCount: 64,
    runtimeParameterCount: 72,
  },
  {
    name: 'nw-t3-05-stack-native-pytorch',
    catalogNode: 'Recurrent Stack',
    definitionId: 'recurrent.stack',
    parameters: { cell: 'simple_rnn', units: 4, layers: 2, return_sequences: true },
    staticOutputShape: [null, 4, 4],
    runtimeOutputShape: [1, 4, 4],
    staticParameterCount: 68,
    runtimeParameterCount: 76,
  },
]

test.describe('NW-T3-05 recurrent catalog slice', () => {
  test.describe.configure({ mode: 'serial' })
  test.setTimeout(180_000)

  test('live catalog renders all recurrent definitions with native support mappings', async ({ page, request }) => {
    const response = await request.get('/api/nodes/catalog')
    expect(response.ok()).toBeTruthy()
    const payload = await response.json() as {
      schema_version: number
      nodes: Array<{
        id: string
        display_name: string
        category: string
        implementations: Record<string, { supported?: boolean }>
      }>
    }
    const selected = payload.nodes.filter((item) => item.category === 'recurrent')
    expect(selected.map((item) => item.id).sort()).toEqual(nativeCases.map((item) => item.definitionId).sort())
    expect(selected).toHaveLength(5)
    for (const definition of selected) {
      expect(definition.implementations.keras?.supported).toBe(true)
      expect(definition.implementations.pytorch?.supported).toBe(true)
    }

    await page.goto('/builder')
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
    await page.getByRole('button', { name: /Recurrent 5 nodes/ }).click()
    for (const candidate of nativeCases) {
      await expect(page.getByRole('button', { name: new RegExp(`${candidate.catalogNode} recurrent\\.`) })).toBeEnabled()
    }
    await mkdir(evidenceDir, { recursive: true })
    await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-05-recurrent-catalog-rendered.png'), fullPage: true })
    await writeFile(path.join(evidenceDir, 'nw-t3-05-live-catalog.json'), JSON.stringify({
      schema_version: payload.schema_version,
      catalog_count: payload.nodes.length,
      recurrent: selected,
    }, null, 2))
  })

  for (const candidate of nativeCases) {
    test(`${candidate.name} builds, persists, and completes exact-profile functional validation`, async ({ page, request }, testInfo) => {
      const fixture = await createProject(
        request,
        `Tier 3 native recurrent ${candidate.catalogNode} ${Date.now()}`,
        graphShell([
          node('sequence-input', 'input.sequence', 'Sequence input', { x: 48, y: 160 }, { shape: [4, 3], dtype: 'float32' }),
          node('sequence-output', 'output.sequence', 'Sequence output', { x: 1040, y: 160 }, {}),
        ]),
      )
      const browserErrors: string[] = []
      page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
      page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
      try {
        await page.goto(`/builder/${fixture.id}`)
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        await expect(page.getByLabel('Enable Keras', { exact: true })).not.toBeChecked()
        const nodeId = await addCatalogNode(page, candidate.catalogNode)
        const materialized = page.locator(`[data-node-id="${nodeId}"]`)
        await materialized.click()
        const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
        await inspector.locator('label.inspector-field').filter({ hasText: 'Units' }).locator('input').fill(String(candidate.parameters.units))
        if (candidate.parameters.return_sequences === true) {
          await inspector.locator('label.inspector-field').filter({ hasText: 'Return sequences' }).locator('input[type="checkbox"]').check()
        }
        if (candidate.definitionId === 'recurrent.bidirectional' || candidate.definitionId === 'recurrent.stack') {
          await inspector.locator('label.inspector-field').filter({ hasText: 'Cell' }).locator('select').selectOption(String(candidate.parameters.cell))
        }
        if (candidate.definitionId === 'recurrent.bidirectional') {
          await inspector.locator('label.inspector-field').filter({ hasText: 'Merge mode' }).locator('select').selectOption(String(candidate.parameters.merge_mode))
        }
        if (candidate.definitionId === 'recurrent.stack') {
          await inspector.locator('label.inspector-field').filter({ hasText: 'Layers' }).locator('input').fill(String(candidate.parameters.layers))
        }
        await connectByPorts(page, 'Sequence input', candidate.catalogNode, 1)
        await connectByPorts(page, candidate.catalogNode, 'Sequence output', 2)

        const validationResponsePromise = page.waitForResponse(
          (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
        )
        await page.getByRole('button', { name: 'Validate', exact: true }).click()
        const validationResponse = await validationResponsePromise
        expect(validationResponse.ok()).toBeTruthy()
        const validationPayload = await validationResponse.json() as ValidationPayload
        expect(validationPayload.valid).toBe(true)
        expect(validationPayload.diagnostics).toEqual([])
        const summary = validationPayload.shape_summaries.find((item) => item.node_id === nodeId)
        expect(summary?.outputs.output?.shape).toEqual(candidate.staticOutputShape)
        expect(summary?.parameter_count).toBe(candidate.staticParameterCount)
        expect(validationPayload.total_parameters).toBe(candidate.staticParameterCount)
        await expect(page.locator('.validation-summary')).toContainText('Ready')
        await page.screenshot({ path: testInfo.outputPath(`${candidate.name}-builder-ready.png`), fullPage: true })

        const saveResponsePromise = page.waitForResponse(
          (response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT',
        )
        await page.getByRole('button', { name: 'Save model', exact: true }).click()
        const saveResponse = await saveResponsePromise
        expect(saveResponse.ok()).toBeTruthy()
        const saved = await saveResponse.json() as {
          current_revision: number
          validation_status: string
          graph: { nodes: unknown[]; edges: unknown[] }
        }
        expect(saved.validation_status).toBe('valid')
        expect(saved.graph.nodes).toHaveLength(3)
        expect(saved.graph.edges).toHaveLength(2)

        await page.reload()
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        await expect(page.getByLabel('Enable Keras', { exact: true })).not.toBeChecked()
        await expect(page.locator('[data-node-id]')).toHaveCount(3)
        await mkdir(evidenceDir, { recursive: true })
        await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-builder-ready.png`), fullPage: true })

        await page.goto(`/models/${fixture.id}/validation`)
        await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
        await expect(page.getByText(/native.*pytorch/i).first()).toBeVisible({ timeout: 75_000 })
        const submitResponsePromise = page.waitForResponse(
          (response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST',
        )
        await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
        const submitResponse = await submitResponsePromise
        expect(submitResponse.ok()).toBeTruthy()
        const submitted = await submitResponse.json() as { id: string }
        await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 90_000 })

        let finalJob: RuntimeJob = { state: 'missing' }
        await expect.poll(async () => {
          const jobResponse = await request.get(`/api/execution/jobs/${submitted.id}`)
          if (!jobResponse.ok()) return 'missing'
          finalJob = await jobResponse.json() as RuntimeJob
          return finalJob.state
        }, { timeout: 90_000 }).toBe('succeeded')

        expect(finalJob.worker).toMatchObject({ api: 'native', backend: 'pytorch', device_id: 'cpu' })
        expect(finalJob.result?.status).toBe('succeeded')
        expect(finalJob.result?.outputs[0]?.shape).toEqual(candidate.runtimeOutputShape)
        expect(finalJob.result?.parameter_count).toBe(candidate.runtimeParameterCount)
        expect(browserErrors, `${candidate.name} should not emit browser errors`).toEqual([])
        await page.screenshot({ path: testInfo.outputPath(`${candidate.name}-runtime-succeeded.png`), fullPage: true })
        await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-runtime-succeeded.png`), fullPage: true })
        await writeFile(path.join(evidenceDir, `${candidate.name}.json`), JSON.stringify({
          case: candidate.name,
          project_id: fixture.id,
          definition_id: candidate.definitionId,
          profile: { execution_backend: 'pytorch', keras_enabled: false, worker: finalJob.worker },
          revision: saved.current_revision,
          expected: {
            static_output_shape: candidate.staticOutputShape,
            runtime_output_shape: candidate.runtimeOutputShape,
            static_parameter_count: candidate.staticParameterCount,
            runtime_parameter_count: candidate.runtimeParameterCount,
          },
          validation: validationPayload,
          submitted_job_id: submitted.id,
          final_job: finalJob,
          browser_errors: browserErrors,
        }, null, 2))
        await writeFile(path.join(evidenceDir, `${candidate.name}-browser-errors.json`), JSON.stringify(browserErrors, null, 2))
      } finally {
        await deleteProject(request, fixture.id)
      }
    })
  }
})
