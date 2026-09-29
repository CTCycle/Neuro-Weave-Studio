import { mkdir, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
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
type GraphEdge = {
  id: string
  source_node_id: string
  source_port_id: string
  target_node_id: string
  target_port_id: string
}
type ShapeSummary = {
  node_id: string
  parameter_count: number
  outputs: Record<string, { shape: Array<number | null> }>
}
type ValidationDiagnostic = { code?: string; message?: string; node_id?: string | null }
type ValidationPayload = {
  valid: boolean
  diagnostics: ValidationDiagnostic[]
  shape_summaries: ShapeSummary[]
  total_parameters: number
}
type RuntimeJob = {
  state: string
  worker?: { api: string; backend: string; device_id: string }
  result?: { status: string; outputs: Array<{ shape: number[] }>; parameter_count: number }
}
type SavedProject = {
  current_revision: number
  validation_status: string
  graph: { nodes: GraphNode[]; edges: GraphEdge[] }
}
type MergeCase = {
  name: string
  displayName: string
  definitionId: string
  kerasEnabled: boolean
  worker: { api: string; backend: string; device_id: string }
}

const evidenceDir = path.resolve(
  process.env.NW_VALIDATION_EVIDENCE_DIR
    ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260929', 'evidence', 'nw-t3-08-merge'),
)

function graphShell(nodes: GraphNode[], edges: GraphEdge[] = []) {
  return {
    schema_version: 5,
    nodes,
    edges,
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

function fanInGraph(definitionId?: string, mismatch = false) {
  const nodes = [
    node('left-input', 'input.tensor', 'Left input', { x: 48, y: 80 }, { shape: [4], dtype: 'float32' }),
    node('middle-input', 'input.tensor', 'Middle input', { x: 48, y: 240 }, { shape: [mismatch ? 5 : 4], dtype: 'float32' }),
    node('right-input', 'input.tensor', 'Right input', { x: 48, y: 400 }, { shape: [4], dtype: 'float32' }),
    node('merge', definitionId ?? 'merge.add', 'Merge operation', { x: 560, y: 240 }, {}),
    node('merged-output', 'output.custom', 'Merged output', { x: 1040, y: 240 }, {}),
  ]
  if (!definitionId) return graphShell(nodes)
  const edges = [
    ['left', 'left-input'],
    ['middle', 'middle-input'],
    ['right', 'right-input'],
  ].map(([edgeId, sourceNodeId]) => ({
    id: `${edgeId}-merge`,
    source_node_id: sourceNodeId,
    source_port_id: 'output',
    target_node_id: 'merge',
    target_port_id: 'inputs',
  }))
  edges.push({
    id: 'merge-output',
    source_node_id: 'merge',
    source_port_id: 'output',
    target_node_id: 'merged-output',
    target_port_id: 'input',
  })
  return graphShell(nodes, edges)
}

async function createProject(
  request: APIRequestContext,
  name: string,
  graph: ReturnType<typeof graphShell>,
  kerasEnabled: boolean,
): Promise<CreatedProject> {
  const response = await request.post('/api/projects', {
    data: {
      name,
      description: 'NW-T3-08 arithmetic merge validation fixture',
      execution_backend: 'pytorch',
      keras_enabled: kerasEnabled,
      graph,
    },
  })
  expect(response.ok()).toBeTruthy()
  return response.json() as Promise<CreatedProject>
}

async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
  const response = await request.delete(`/api/projects/${id}`)
  expect([204, 404]).toContain(response.status())
}

async function addCatalogNode(page: Page, displayName: string, definitionId: string): Promise<string> {
  const nodes = page.locator('[data-node-id]')
  const before = new Set(
    await nodes.evaluateAll((elements) => elements
      .map((element) => element.getAttribute('data-node-id'))
      .filter(Boolean) as string[]),
  )
  await page.getByRole('textbox', { name: 'Search nodes' }).fill(definitionId)
  const item = page.locator('.catalog-item').filter({ hasText: definitionId }).first()
  await expect(item, `${definitionId} should be available in the live catalog`).toBeEnabled()
  await item.click()
  await expect(nodes).toHaveCount(before.size + 1)
  const after = await nodes.evaluateAll((elements) => elements
    .map((element) => element.getAttribute('data-node-id'))
    .filter(Boolean) as string[])
  const created = after.find((nodeId) => !before.has(nodeId))
  expect(created, `Builder should materialize ${definitionId}`).toBeTruthy()
  await page.getByRole('textbox', { name: 'Search nodes' }).fill('')
  return created!
}

async function connectByPorts(
  page: Page,
  sourceLabel: string,
  targetLabel: string,
  targetPort: string,
  expectedConnectionCount: number,
): Promise<void> {
  const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
  const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
  const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
  const target = targetNode.getByRole('button', { name: new RegExp(`^Port ${targetPort} \\(input\\)`) })
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
  expect(payload.valid, `${sourceLabel} → ${targetLabel}.${targetPort} should be accepted`).toBe(true)
  await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
}

async function validateSaveAndRun(
  page: Page,
  request: APIRequestContext,
  fixture: CreatedProject,
  candidate: MergeCase,
  mergeNodeId: string,
  browserErrors: string[],
  failedRequests: string[],
  testInfo: TestInfo,
): Promise<{ validation: ValidationPayload; saved: SavedProject; finalJob: RuntimeJob; submittedJobId: string }> {
  await connectByPorts(page, 'Left input', candidate.displayName, 'Inputs', 1)
  await connectByPorts(page, 'Middle input', candidate.displayName, 'Inputs', 2)
  await connectByPorts(page, 'Right input', candidate.displayName, 'Inputs', 3)
  await connectByPorts(page, candidate.displayName, 'Merged output', 'Input', 4)

  const validationResponsePromise = page.waitForResponse(
    (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Validate', exact: true }).click()
  const validationResponse = await validationResponsePromise
  expect(validationResponse.ok()).toBeTruthy()
  const validation = await validationResponse.json() as ValidationPayload
  expect(validation.valid).toBe(true)
  expect(validation.diagnostics).toEqual([])
  const summary = validation.shape_summaries.find((item) => item.node_id === mergeNodeId)
  expect(summary?.outputs.output?.shape).toEqual([null, 4])
  expect(summary?.parameter_count).toBe(0)
  expect(validation.total_parameters).toBe(0)
  await expect(page.locator('.validation-summary')).toContainText('Ready')
  await page.screenshot({ path: testInfo.outputPath(`${candidate.name}-builder-ready.png`), fullPage: true })

  const saveResponsePromise = page.waitForResponse(
    (response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT',
  )
  await page.getByRole('button', { name: 'Save model', exact: true }).click()
  const saveResponse = await saveResponsePromise
  expect(saveResponse.ok()).toBeTruthy()
  const saved = await saveResponse.json() as SavedProject
  expect(saved.validation_status).toBe('valid')
  expect(saved.graph.nodes).toHaveLength(5)
  expect(saved.graph.edges).toHaveLength(4)
  expect(saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)?.parameters).toEqual({})

  await page.reload()
  await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('[data-node-id]')).toHaveCount(5)
  await expect(page.getByRole('group', { name: `Node ${candidate.displayName}`, exact: true })).toBeVisible()
  await mkdir(evidenceDir, { recursive: true })
  await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-builder-ready.png`), fullPage: true })

  await page.goto(`/models/${fixture.id}/validation`)
  await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText(candidate.kerasEnabled ? /keras.*pytorch/i : /native.*pytorch/i).first()).toBeVisible({ timeout: 75_000 })
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
  expect(finalJob.worker).toMatchObject(candidate.worker)
  expect(finalJob.result?.status).toBe('succeeded')
  expect(finalJob.result?.outputs[0]?.shape).toEqual([1, 4])
  expect(finalJob.result?.parameter_count).toBe(0)
  expect(browserErrors, `${candidate.name} should not emit browser errors`).toEqual([])
  expect(failedRequests, `${candidate.name} should not emit failed product requests`).toEqual([])
  await page.screenshot({ path: testInfo.outputPath(`${candidate.name}-runtime-succeeded.png`), fullPage: true })
  await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-runtime-succeeded.png`), fullPage: true })
  await writeFile(path.join(evidenceDir, `${candidate.name}.json`), JSON.stringify({
    case: candidate.name,
    project_id: fixture.id,
    definition_id: candidate.definitionId,
    profile: { execution_backend: 'pytorch', keras_enabled: candidate.kerasEnabled, worker: finalJob.worker },
    saved_revision: saved.current_revision,
    graph: { node_count: saved.graph.nodes.length, edge_count: saved.graph.edges.length },
    expected: {
      static_output_shape: [null, 4],
      runtime_output_shape: [1, 4],
      parameter_count: 0,
    },
    validation,
    submitted_job_id: submitted.id,
    final_job: finalJob,
    browser_errors: browserErrors,
    failed_requests: failedRequests,
    cleanup: 'project deleted in finally block',
  }, null, 2))
  await writeFile(path.join(evidenceDir, `${candidate.name}-browser-errors.json`), JSON.stringify({ browser_errors: browserErrors, failed_requests: failedRequests }, null, 2))
  return { validation, saved, finalJob, submittedJobId: submitted.id }
}

const arithmeticCases: MergeCase[] = [
  { name: 'nw-t3-08-add-keras-pytorch', displayName: 'Add', definitionId: 'merge.add', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-08-subtract-keras-pytorch', displayName: 'Subtract', definitionId: 'merge.subtract', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-08-multiply-keras-pytorch', displayName: 'Multiply', definitionId: 'merge.multiply', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-08-divide-keras-pytorch', displayName: 'Divide', definitionId: 'merge.divide', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-08-maximum-keras-pytorch', displayName: 'Maximum', definitionId: 'merge.maximum', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-08-minimum-keras-pytorch', displayName: 'Minimum', definitionId: 'merge.minimum', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-08-average-keras-pytorch', displayName: 'Average', definitionId: 'merge.average', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-08-add-native-pytorch', displayName: 'Add', definitionId: 'merge.add', kerasEnabled: false, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-08-average-native-pytorch', displayName: 'Average', definitionId: 'merge.average', kerasEnabled: false, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
]

test.describe('NW-T3-08 arithmetic merge and fan-in catalog slice', () => {
  test.describe.configure({ mode: 'serial' })
  test.setTimeout(240_000)

  test('live catalog renders arithmetic merge definitions and verified mappings', async ({ page, request }) => {
    const response = await request.get('/api/nodes/catalog')
    expect(response.ok()).toBeTruthy()
    const catalog = await response.json() as {
      schema_version: number
      nodes: Array<{
        id: string
        display_name: string
        category: string
        operation?: { family?: string; shape_behavior?: string; variadic?: boolean }
        implementations: Record<string, { supported?: boolean; adapter?: string | null }>
      }>
    }
    const merge = catalog.nodes.filter((item) => item.category === 'merge').sort((left, right) => left.id.localeCompare(right.id))
    expect(merge).toHaveLength(14)
    const expectedIds = [
      'merge.add', 'merge.average', 'merge.concatenate', 'merge.divide', 'merge.dot', 'merge.matmul',
      'merge.maximum', 'merge.minimum', 'merge.multiply', 'merge.residual', 'merge.skip', 'merge.split',
      'merge.stack', 'merge.subtract',
    ]
    expect(merge.map((item) => item.id)).toEqual(expectedIds)
    const arithmetic = merge.filter((item) => [
      'merge.add', 'merge.subtract', 'merge.multiply', 'merge.divide', 'merge.maximum', 'merge.minimum',
    ].includes(item.id))
    expect(arithmetic).toHaveLength(6)
    for (const item of arithmetic) {
      expect(item.operation).toMatchObject({ family: 'elementwise', shape_behavior: 'preserve', variadic: true })
      expect(item.implementations.keras?.supported).toBe(true)
      expect(item.implementations.tensorflow?.supported).toBe(true)
      expect(item.implementations.pytorch?.supported).toBe(true)
      expect(item.implementations.jax?.supported).toBe(true)
    }
    const average = merge.find((item) => item.id === 'merge.average')!
    expect(average.implementations.keras).toMatchObject({ supported: true, adapter: 'average' })
    expect(average.implementations.pytorch).toMatchObject({ supported: true, adapter: 'average' })

    await page.goto('/builder')
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
    await page.getByRole('button', { name: /Merge 14 nodes/ }).click()
    for (const item of merge) {
      await expect(page.getByRole('button', { name: new RegExp(`${item.display_name} ${item.id} ·`) })).toBeEnabled()
    }
    await mkdir(evidenceDir, { recursive: true })
    await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-08-merge-catalog-rendered.png'), fullPage: true })
    await writeFile(path.join(evidenceDir, 'nw-t3-08-live-catalog.json'), JSON.stringify({
      schema_version: catalog.schema_version,
      catalog_count: catalog.nodes.length,
      catalog_fingerprint_sha256: createHash('sha256').update(JSON.stringify(catalog)).digest('hex'),
      merge,
      arithmetic_ids: arithmetic.map((item) => item.id),
    }, null, 2))
  })

  for (const candidate of arithmeticCases) {
    test(`${candidate.name} builds, persists, and completes exact-profile functional validation`, async ({ page, request }, testInfo) => {
      const fixture = await createProject(
        request,
        `Tier 3 arithmetic merge ${candidate.definitionId} ${candidate.kerasEnabled ? 'keras' : 'native'} ${Date.now()}`,
        graphShell([
          node('left-input', 'input.tensor', 'Left input', { x: 48, y: 80 }, { shape: [4], dtype: 'float32' }),
          node('middle-input', 'input.tensor', 'Middle input', { x: 48, y: 240 }, { shape: [4], dtype: 'float32' }),
          node('right-input', 'input.tensor', 'Right input', { x: 48, y: 400 }, { shape: [4], dtype: 'float32' }),
          node('merged-output', 'output.custom', 'Merged output', { x: 1040, y: 240 }, {}),
        ]),
        candidate.kerasEnabled,
      )
      const browserErrors: string[] = []
      const failedRequests: string[] = []
      page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
      page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
      page.on('requestfailed', (event) => failedRequests.push(`${event.method()} ${event.url()} ${event.failure()?.errorText ?? 'unknown'}`))
      page.on('response', (event) => {
        if (event.status() >= 400 && event.url().includes('/api/')) failedRequests.push(`${event.status()} ${event.request().method()} ${event.url()}`)
      })
      try {
        await page.goto(`/builder/${fixture.id}`)
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        if (!candidate.kerasEnabled) await expect(page.getByLabel('Enable Keras', { exact: true })).not.toBeChecked()
        const mergeNodeId = await addCatalogNode(page, candidate.displayName, candidate.definitionId)
        await validateSaveAndRun(page, request, fixture, candidate, mergeNodeId, browserErrors, failedRequests, testInfo)
      } finally {
        await deleteProject(request, fixture.id)
      }
    })
  }

  test('incompatible fan-in shapes remain blocking and fail closed', async ({ page, request }, testInfo) => {
    const fixture = await createProject(
      request,
      `Tier 3 arithmetic merge mismatch ${Date.now()}`,
      fanInGraph('merge.add', true),
      true,
    )
    const browserErrors: string[] = []
    const failedRequests: string[] = []
    page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
    page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
    page.on('requestfailed', (event) => failedRequests.push(`${event.method()} ${event.url()} ${event.failure()?.errorText ?? 'unknown'}`))
    try {
      await page.goto(`/builder/${fixture.id}`)
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
      const validationResponsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Validate', exact: true }).click()
      const validationResponse = await validationResponsePromise
      expect(validationResponse.ok()).toBeTruthy()
      const validation = await validationResponse.json() as ValidationPayload
      expect(validation.valid).toBe(false)
      expect(validation.diagnostics.some((item) => item.message?.includes('compatible shapes'))).toBe(true)
      await expect(page.locator('.validation-summary')).toContainText(/Blocked/i)
      await mkdir(evidenceDir, { recursive: true })
      await page.screenshot({ path: testInfo.outputPath('nw-t3-08-incompatible-fan-in.png'), fullPage: true })
      await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-08-incompatible-fan-in.png'), fullPage: true })
      await writeFile(path.join(evidenceDir, 'nw-t3-08-incompatible-fan-in.json'), JSON.stringify({
        case: 'nw-t3-08-incompatible-fan-in',
        project_id: fixture.id,
        definition_id: 'merge.add',
        authored_shapes: { left: [4], middle: [5], right: [4] },
        validation,
        browser_errors: browserErrors,
        failed_requests: failedRequests,
        execution_job_submissions: 0,
      }, null, 2))
      expect(browserErrors).toEqual([])
      expect(failedRequests).toEqual([])
    } finally {
      await deleteProject(request, fixture.id)
    }
  })
})
