import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page, type TestInfo } from '@playwright/test'

type CreatedProject = { id: string; current_revision: number; name: string }
type GraphNode = {
  id: string
  definition_id: string
  definition_version: number
  label: string
  position: { x: number; y: number }
  parameters: Record<string, unknown>
}
type AdvancedCase = {
  name: string
  inputDefinition: 'input.sequence' | 'input.image' | 'input.tensor'
  inputShape: number[]
  catalogNode: string
  staticOutputShape: Array<number | null>
  runtimeOutputShape: number[]
  parameterCount: number
  kerasEnabled?: boolean
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
    inputs: Array<{ shape: number[] }>
    outputs: Array<{ shape: number[] }>
    parameter_count: number
  }
}

const evidenceDir = path.resolve(
  process.env.NW_VALIDATION_EVIDENCE_DIR
    ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260928', 'evidence'),
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

function inputOutputGraph(inputDefinition: AdvancedCase['inputDefinition'], inputShape: number[]) {
  return graphShell([
    node('matrix-input', inputDefinition, 'Matrix input', { x: 64, y: 180 }, { shape: inputShape, dtype: 'float32' }),
    node('matrix-output', 'output.custom', 'Matrix output', { x: 1200, y: 180 }, {}),
  ])
}

async function createProject(
  request: APIRequestContext,
  name: string,
  modelGraph: ReturnType<typeof graphShell>,
  kerasEnabled: boolean,
): Promise<CreatedProject> {
  const response = await request.post('/api/projects', {
    data: {
      name,
      description: 'NW-T3-04 advanced and spatial convolution validation fixture',
      execution_backend: 'pytorch',
      keras_enabled: kerasEnabled,
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

async function connectByLabels(
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

async function validateSaveAndRun(
  page: Page,
  request: APIRequestContext,
  fixture: CreatedProject,
  advancedNodeId: string,
  expected: AdvancedCase,
  testInfo: TestInfo,
  browserErrors: string[],
): Promise<void> {
  const validationResponsePromise = page.waitForResponse(
    (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Validate', exact: true }).click()
  const validationResponse = await validationResponsePromise
  expect(validationResponse.ok()).toBeTruthy()
  const validationPayload = await validationResponse.json() as ValidationPayload
  expect(validationPayload.valid).toBe(true)
  expect(validationPayload.diagnostics).toEqual([])
  const summary = validationPayload.shape_summaries.find((item) => item.node_id === advancedNodeId)
  expect(summary?.outputs.output?.shape).toEqual(expected.staticOutputShape)
  expect(summary?.parameter_count).toBe(expected.parameterCount)
  expect(validationPayload.total_parameters).toBe(expected.parameterCount)
  await expect(page.locator('.validation-summary')).toContainText('Ready')
  await page.screenshot({ path: testInfo.outputPath(`${expected.name}-builder-ready.png`), fullPage: true })

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
  await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('[data-node-id]')).toHaveCount(3)
  await mkdir(evidenceDir, { recursive: true })
  await page.screenshot({ path: path.join(evidenceDir, `${expected.name}-builder-ready.png`), fullPage: true })

  await page.goto(`/models/${fixture.id}/validation`)
  await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
  const profilePattern = expected.kerasEnabled === false ? /native.*pytorch/i : /keras.*pytorch/i
  await expect(page.getByText(profilePattern).first()).toBeVisible({ timeout: 75_000 })
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
    const response = await request.get(`/api/execution/jobs/${submitted.id}`)
    if (!response.ok()) return 'missing'
    finalJob = await response.json() as RuntimeJob
    return finalJob.state
  }, { timeout: 90_000 }).toBe('succeeded')

  expect(finalJob.worker).toMatchObject({
    api: expected.kerasEnabled === false ? 'native' : 'keras',
    backend: 'pytorch',
    device_id: 'cpu',
  })
  expect(finalJob.result?.status).toBe('succeeded')
  expect(finalJob.result?.parameter_count).toBe(expected.parameterCount)
  expect(finalJob.result?.outputs[0]?.shape).toEqual(expected.runtimeOutputShape)
  expect(browserErrors, `${expected.name} should not emit browser errors`).toEqual([])
  await page.screenshot({ path: testInfo.outputPath(`${expected.name}-runtime-succeeded.png`), fullPage: true })
  await page.screenshot({ path: path.join(evidenceDir, `${expected.name}-runtime-succeeded.png`), fullPage: true })
  await writeFile(path.join(evidenceDir, `${expected.name}.json`), JSON.stringify({
    case: expected.name,
    project_id: fixture.id,
    profile: {
      execution_backend: 'pytorch',
      keras_enabled: expected.kerasEnabled !== false,
      worker: finalJob.worker,
    },
    revision: saved.current_revision,
    expected: {
      static_output_shape: expected.staticOutputShape,
      runtime_output_shape: expected.runtimeOutputShape,
      parameter_count: expected.parameterCount,
    },
    validation: validationPayload,
    submitted_job_id: submitted.id,
    final_job: finalJob,
    browser_errors: browserErrors,
  }, null, 2))
  await writeFile(path.join(evidenceDir, `${expected.name}-browser-errors.json`), JSON.stringify(browserErrors, null, 2))
}

const advancedCases: AdvancedCase[] = [
  {
    name: 'nw-t3-04-max-pool-1d-keras-pytorch',
    inputDefinition: 'input.sequence',
    inputShape: [8, 2],
    catalogNode: 'Max Pooling 1D',
    staticOutputShape: [null, 4, 2],
    runtimeOutputShape: [1, 4, 2],
    parameterCount: 0,
  },
  {
    name: 'nw-t3-04-average-pool-2d-keras-pytorch',
    inputDefinition: 'input.image',
    inputShape: [8, 8, 2],
    catalogNode: 'Average Pooling 2D',
    staticOutputShape: [null, 4, 4, 2],
    runtimeOutputShape: [1, 4, 4, 2],
    parameterCount: 0,
  },
  {
    name: 'nw-t3-04-max-pool-3d-keras-pytorch',
    inputDefinition: 'input.tensor',
    inputShape: [4, 4, 4, 2],
    catalogNode: 'Max Pooling 3D',
    staticOutputShape: [null, 2, 2, 2, 2],
    runtimeOutputShape: [1, 2, 2, 2, 2],
    parameterCount: 0,
  },
  {
    name: 'nw-t3-04-depthwise-keras-pytorch',
    inputDefinition: 'input.image',
    inputShape: [4, 4, 2],
    catalogNode: 'Depthwise Convolution',
    staticOutputShape: [null, 2, 2, 2],
    runtimeOutputShape: [1, 2, 2, 2],
    parameterCount: 20,
  },
  {
    name: 'nw-t3-04-depthwise-native-pytorch',
    inputDefinition: 'input.image',
    inputShape: [4, 4, 2],
    catalogNode: 'Depthwise Convolution',
    staticOutputShape: [null, 2, 2, 2],
    runtimeOutputShape: [1, 2, 2, 2],
    parameterCount: 20,
    kerasEnabled: false,
  },
  {
    name: 'nw-t3-04-separable-keras-pytorch',
    inputDefinition: 'input.image',
    inputShape: [4, 4, 2],
    catalogNode: 'Separable Convolution',
    staticOutputShape: [null, 2, 2, 32],
    runtimeOutputShape: [1, 2, 2, 32],
    parameterCount: 114,
  },
  {
    name: 'nw-t3-04-transposed-keras-pytorch',
    inputDefinition: 'input.image',
    inputShape: [4, 4, 1],
    catalogNode: 'Transposed Convolution',
    staticOutputShape: [null, 6, 6, 32],
    runtimeOutputShape: [1, 6, 6, 32],
    parameterCount: 320,
  },
  {
    name: 'nw-t3-04-upsampling-keras-pytorch',
    inputDefinition: 'input.image',
    inputShape: [3, 4, 1],
    catalogNode: 'Upsampling',
    staticOutputShape: [null, 6, 8, 1],
    runtimeOutputShape: [1, 6, 8, 1],
    parameterCount: 0,
  },
  {
    name: 'nw-t3-04-padding-keras-pytorch',
    inputDefinition: 'input.image',
    inputShape: [4, 5, 1],
    catalogNode: 'Padding',
    staticOutputShape: [null, 6, 7, 1],
    runtimeOutputShape: [1, 6, 7, 1],
    parameterCount: 0,
  },
  {
    name: 'nw-t3-04-cropping-keras-pytorch',
    inputDefinition: 'input.image',
    inputShape: [6, 7, 1],
    catalogNode: 'Cropping',
    staticOutputShape: [null, 4, 5, 1],
    runtimeOutputShape: [1, 4, 5, 1],
    parameterCount: 0,
  },
]

test.describe('NW-T3-04 advanced and spatial convolution catalog slice', () => {
  test.describe.configure({ mode: 'serial' })
  test.setTimeout(180_000)

  test('live catalog exposes the selected convolution definitions and profile mappings', async ({ request }) => {
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
    const selected = payload.nodes.filter((item) => item.category === 'convolutional' && advancedCases.some((candidate) => candidate.catalogNode === item.display_name))
    const selectedIds = selected.map((item) => item.id).sort()
    expect(selectedIds).toEqual([
      'conv.average_pool2d',
      'conv.cropping',
      'conv.depthwise',
      'conv.max_pool1d',
      'conv.max_pool3d',
      'conv.padding',
      'conv.separable',
      'conv.transposed',
      'conv.upsampling',
    ])
    expect(selected).toHaveLength(9)
    for (const definition of selected) {
      expect(definition.implementations.keras?.supported).toBe(true)
      expect(definition.implementations.pytorch?.supported).toBe(true)
    }
    await mkdir(evidenceDir, { recursive: true })
    await writeFile(path.join(evidenceDir, 'nw-t3-04-live-catalog.json'), JSON.stringify({
      schema_version: payload.schema_version,
      catalog_count: payload.nodes.length,
      selected,
    }, null, 2))
  })

  for (const candidate of advancedCases) {
    test(`${candidate.name} builds, persists, and completes exact-profile functional validation`, async ({ page, request }, testInfo) => {
      const fixture = await createProject(
        request,
        `NW-T3-04 ${candidate.catalogNode} ${Date.now()}`,
        inputOutputGraph(candidate.inputDefinition, candidate.inputShape),
        candidate.kerasEnabled !== false,
      )
      const browserErrors: string[] = []
      page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
      page.on('console', (message) => {
        if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`)
      })
      try {
        await page.goto(`/builder/${fixture.id}`)
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        await expect(page.locator('[data-node-id]')).toHaveCount(2)
        const advancedNodeId = await addCatalogNode(page, candidate.catalogNode)
        await connectByLabels(page, 'Matrix input', candidate.catalogNode, 1)
        await connectByLabels(page, candidate.catalogNode, 'Matrix output', 2)
        await validateSaveAndRun(page, request, fixture, advancedNodeId, candidate, testInfo, browserErrors)
      } finally {
        await deleteProject(request, fixture.id)
      }
    })
  }
})
