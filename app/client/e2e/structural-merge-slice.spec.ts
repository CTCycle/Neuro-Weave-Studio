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
type StructuralCase = {
  name: string
  displayName: string
  definitionId: string
  kerasEnabled: boolean
  inputShapes: number[][]
  expectedStaticShape: Array<number | null>
  expectedRuntimeShape: number[]
  outputPortId?: string
  inputTargetLabel?: 'Input' | 'Inputs'
  parameterOverrides?: Record<string, unknown>
  worker: { api: string; backend: string; device_id: string }
}

const evidenceDir = path.resolve(
  process.env.NW_VALIDATION_EVIDENCE_DIR
    ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260929', 'evidence', 'nw-t3-09-structural-merge'),
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

function graphForCase(candidate: StructuralCase) {
  const nodes = candidate.inputShapes.map((shape, index) => node(
    `input-${index + 1}`,
    'input.tensor',
    `Input ${index + 1}`,
    { x: 48, y: 80 + index * 160 },
    { shape, dtype: 'float32' },
  ))
  nodes.push(node('merged-output', 'output.custom', 'Merged output', { x: 1040, y: 240 }, {}))
  return graphShell(nodes)
}

function invalidGraph(definitionId: string, inputShapes: number[][], parameters: Record<string, unknown> = {}) {
  const nodes = inputShapes.map((shape, index) => node(
    `input-${index + 1}`,
    'input.tensor',
    `Input ${index + 1}`,
    { x: 48, y: 80 + index * 160 },
    { shape, dtype: 'float32' },
  ))
  nodes.push(node('merge', definitionId, 'Merge operation', { x: 560, y: 240 }, parameters))
  nodes.push(node('merged-output', 'output.custom', 'Merged output', { x: 1040, y: 240 }, {}))
  const edges: GraphEdge[] = inputShapes.map((_, index) => ({
    id: `input-${index + 1}-merge`,
    source_node_id: `input-${index + 1}`,
    source_port_id: 'output',
    target_node_id: 'merge',
    target_port_id: inputShapes.length === 1 ? 'input' : 'inputs',
  }))
  edges.push({
    id: 'merge-output',
    source_node_id: 'merge',
    source_port_id: parameters.sections ? 'output_0' : 'output',
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
      description: 'NW-T3-09 structural merge validation fixture',
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

async function configureParameters(page: Page, candidate: StructuralCase): Promise<void> {
  const sections = candidate.parameterOverrides?.sections
  if (sections === undefined) return
  const field = page.locator('label.inspector-field').filter({ hasText: /^Sections/ }).locator('input').first()
  await expect(field).toBeVisible()
  await field.fill(String(sections))
  await field.press('Tab')
  await expect(field).toHaveValue(String(sections))
  await expect(page.getByRole('button', { name: /Output 2 \(output\)/ })).toBeVisible()
}

async function connectByPorts(
  page: Page,
  sourceLabel: string,
  targetLabel: string,
  targetPort: string,
  expectedConnectionCount: number,
  sourcePort = 'Output',
  sourcePortId = 'output',
): Promise<void> {
  const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
  const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
  const source = sourceNode.getByRole('button', { name: new RegExp(`^Port ${sourcePort} \\(${sourcePortId}\\)`) })
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
  expect(payload.valid, `${sourceLabel} → ${targetLabel}.${targetPort} should be accepted: ${JSON.stringify(payload)}`).toBe(true)
  await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
}

async function validateSaveAndRun(
  page: Page,
  request: APIRequestContext,
  fixture: CreatedProject,
  candidate: StructuralCase,
  mergeNodeId: string,
  browserErrors: string[],
  failedRequests: string[],
  testInfo: TestInfo,
): Promise<void> {
  const targetPort = candidate.inputTargetLabel ?? 'Inputs'
  for (let index = 0; index < candidate.inputShapes.length; index += 1) {
    await connectByPorts(page, `Input ${index + 1}`, candidate.displayName, targetPort, index + 1)
  }
  const outputPortId = candidate.outputPortId ?? 'output'
  const outputPort = outputPortId === 'output' ? 'Output' : `Output ${outputPortId.replace('output_', '')}`
  await connectByPorts(page, candidate.displayName, 'Merged output', 'Input', candidate.inputShapes.length + 1, outputPort, 'output')

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
  expect(summary?.outputs[outputPortId]?.shape).toEqual(candidate.expectedStaticShape)
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
  expect(saved.graph.nodes).toHaveLength(candidate.inputShapes.length + 2)
  expect(saved.graph.edges).toHaveLength(candidate.inputShapes.length + 1)
  const savedMerge = saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)
  expect(savedMerge).toBeTruthy()
  if (candidate.parameterOverrides) expect(savedMerge?.parameters).toMatchObject(candidate.parameterOverrides)

  await page.reload()
  await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('[data-node-id]')).toHaveCount(candidate.inputShapes.length + 2)
  await expect(page.getByRole('group', { name: `Node ${candidate.displayName}`, exact: true })).toBeVisible()
  if (candidate.outputPortId) {
    await expect(page.getByRole('button', { name: /^Port Output 0 \(output\)/ })).toBeVisible()
  }
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
  expect(finalJob.result?.outputs[0]?.shape).toEqual(candidate.expectedRuntimeShape)
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
      static_output_shape: candidate.expectedStaticShape,
      runtime_output_shape: candidate.expectedRuntimeShape,
      parameter_count: 0,
      output_port_id: outputPortId,
    },
    validation,
    submitted_job_id: submitted.id,
    final_job: finalJob,
    browser_errors: browserErrors,
    failed_requests: failedRequests,
    cleanup: 'project deleted in finally block',
  }, null, 2))
  await writeFile(path.join(evidenceDir, `${candidate.name}-browser-errors.json`), JSON.stringify({ browser_errors: browserErrors, failed_requests: failedRequests }, null, 2))
}

const structuralCases: StructuralCase[] = [
  { name: 'nw-t3-09-concatenate-keras-pytorch', displayName: 'Concatenate', definitionId: 'merge.concatenate', kerasEnabled: true, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 6], expectedRuntimeShape: [1, 2, 6], worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-09-concatenate-native-pytorch', displayName: 'Concatenate', definitionId: 'merge.concatenate', kerasEnabled: false, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 6], expectedRuntimeShape: [1, 2, 6], worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-09-stack-keras-pytorch', displayName: 'Stack', definitionId: 'merge.stack', kerasEnabled: true, inputShapes: [[4], [4]], expectedStaticShape: [2, null, 4], expectedRuntimeShape: [2, 1, 4], worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-09-stack-native-pytorch', displayName: 'Stack', definitionId: 'merge.stack', kerasEnabled: false, inputShapes: [[4], [4]], expectedStaticShape: [2, null, 4], expectedRuntimeShape: [2, 1, 4], worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-09-split-keras-pytorch', displayName: 'Split', definitionId: 'merge.split', kerasEnabled: true, inputShapes: [[6]], expectedStaticShape: [null, 2], expectedRuntimeShape: [1, 2], outputPortId: 'output_0', inputTargetLabel: 'Input', parameterOverrides: { sections: 3 }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-09-split-native-pytorch', displayName: 'Split', definitionId: 'merge.split', kerasEnabled: false, inputShapes: [[6]], expectedStaticShape: [null, 2], expectedRuntimeShape: [1, 2], outputPortId: 'output_0', inputTargetLabel: 'Input', parameterOverrides: { sections: 3 }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-09-dot-keras-pytorch', displayName: 'Dot Product', definitionId: 'merge.dot', kerasEnabled: true, inputShapes: [[2, 3], [3, 4]], expectedStaticShape: [null, 2, 4], expectedRuntimeShape: [1, 2, 4], worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-09-matmul-keras-pytorch', displayName: 'Matrix Multiplication', definitionId: 'merge.matmul', kerasEnabled: true, inputShapes: [[2, 3], [3, 4]], expectedStaticShape: [null, 2, 4], expectedRuntimeShape: [1, 2, 4], worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-09-matmul-native-pytorch', displayName: 'Matrix Multiplication', definitionId: 'merge.matmul', kerasEnabled: false, inputShapes: [[2, 3], [3, 4]], expectedStaticShape: [null, 2, 4], expectedRuntimeShape: [1, 2, 4], worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-09-residual-keras-pytorch', displayName: 'Residual Connection', definitionId: 'merge.residual', kerasEnabled: true, inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-09-residual-native-pytorch', displayName: 'Residual Connection', definitionId: 'merge.residual', kerasEnabled: false, inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-09-skip-keras-pytorch', displayName: 'Skip Connection', definitionId: 'merge.skip', kerasEnabled: true, inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-09-skip-native-pytorch', displayName: 'Skip Connection', definitionId: 'merge.skip', kerasEnabled: false, inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
]

test.describe('NW-T3-09 structural merge catalog slice', () => {
  test.describe.configure({ mode: 'serial' })
  test.setTimeout(240_000)

  test('live catalog renders structural merge definitions and verified mappings', async ({ page, request }) => {
    const response = await request.get('/api/nodes/catalog')
    expect(response.ok()).toBeTruthy()
    const catalog = await response.json() as {
      schema_version: number
      nodes: Array<{
        id: string
        display_name: string
        category: string
        operation?: { family?: string; shape_behavior?: string; arity?: number; variadic?: boolean } | null
        implementations: Record<string, { supported?: boolean; adapter?: string | null; symbol?: string | null }>
      }>
    }
    const merge = catalog.nodes.filter((item) => item.category === 'merge').sort((left, right) => left.id.localeCompare(right.id))
    const expectedIds = [
      'merge.add', 'merge.average', 'merge.concatenate', 'merge.divide', 'merge.dot', 'merge.matmul',
      'merge.maximum', 'merge.minimum', 'merge.multiply', 'merge.residual', 'merge.skip', 'merge.split',
      'merge.stack', 'merge.subtract',
    ]
    expect(merge.map((item) => item.id)).toEqual(expectedIds)
    const expectedContracts: Record<string, { family: string; shape_behavior: string; arity: number; variadic: boolean }> = {
      'merge.concatenate': { family: 'tensor', shape_behavior: 'concatenate', arity: 1, variadic: true },
      'merge.dot': { family: 'binary', shape_behavior: 'matmul', arity: 2, variadic: false },
      'merge.matmul': { family: 'binary', shape_behavior: 'matmul', arity: 2, variadic: false },
      'merge.stack': { family: 'tensor', shape_behavior: 'stack', arity: 1, variadic: true },
    }
    for (const [id, contract] of Object.entries(expectedContracts)) {
      expect(merge.find((item) => item.id === id)?.operation).toMatchObject(contract)
    }
    for (const id of ['merge.residual', 'merge.skip', 'merge.split']) {
      const item = merge.find((entry) => entry.id === id)!
      expect(item.operation).toBeNull()
      expect(item.implementations.keras?.supported).toBe(true)
      expect(item.implementations.pytorch?.supported).toBe(true)
    }
    expect(merge.find((item) => item.id === 'merge.residual')?.implementations).toMatchObject({ keras: { adapter: 'keras_merge' }, pytorch: { adapter: 'add' } })
    expect(merge.find((item) => item.id === 'merge.skip')?.implementations).toMatchObject({ keras: { adapter: 'keras_merge' }, pytorch: { adapter: 'add' } })
    expect(merge.find((item) => item.id === 'merge.split')?.implementations).toMatchObject({ keras: { adapter: 'split' }, pytorch: { adapter: 'split' } })
    for (const item of merge) {
      expect(item.implementations.keras?.supported).toBe(true)
      expect(item.implementations.pytorch?.supported).toBe(true)
    }

    await page.goto('/builder')
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
    await page.getByRole('button', { name: /Merge 14 nodes/ }).click()
    for (const item of merge) {
      await expect(page.getByRole('button', { name: new RegExp(`${item.display_name} ${item.id} ·`) })).toBeEnabled()
    }
    await mkdir(evidenceDir, { recursive: true })
    await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-09-structural-merge-catalog-rendered.png'), fullPage: true })
    await writeFile(path.join(evidenceDir, 'nw-t3-09-live-catalog.json'), JSON.stringify({
      schema_version: catalog.schema_version,
      catalog_count: catalog.nodes.length,
      catalog_fingerprint_sha256: createHash('sha256').update(JSON.stringify(catalog)).digest('hex'),
      merge,
      structural_ids: Object.keys(expectedContracts).concat(['merge.residual', 'merge.skip', 'merge.split']),
    }, null, 2))
  })

  for (const candidate of structuralCases) {
    test(`${candidate.name} builds, persists, and completes exact-profile functional validation`, async ({ page, request }, testInfo) => {
      const fixture = await createProject(
        request,
        `Tier 3 structural merge ${candidate.definitionId} ${candidate.kerasEnabled ? 'keras' : 'native'} ${Date.now()}`,
        graphForCase(candidate),
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
        await configureParameters(page, candidate)
        await validateSaveAndRun(page, request, fixture, candidate, mergeNodeId, browserErrors, failedRequests, testInfo)
      } finally {
        await deleteProject(request, fixture.id)
      }
    })
  }

  test('matmul and split shape contracts remain fail-closed for invalid graphs', async ({ page, request }, testInfo) => {
    const cases = [
      {
        name: 'nw-t3-09-invalid-matmul',
        definitionId: 'merge.matmul',
        graph: invalidGraph('merge.matmul', [[2, 3], [5, 4]]),
        message: /inner|compatible|matmul|shape/i,
      },
      {
        name: 'nw-t3-09-invalid-split',
        definitionId: 'merge.split',
        graph: invalidGraph('merge.split', [[5]], { sections: 2, axis: -1 }),
        message: /split|divis|shape/i,
      },
    ]
    for (const candidate of cases) {
      const fixture = await createProject(request, `Tier 3 invalid structural merge ${candidate.name} ${Date.now()}`, candidate.graph, true)
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
        expect(validation.diagnostics).toEqual(expect.arrayContaining([
          expect.objectContaining({ message: expect.stringMatching(candidate.message) }),
        ]))
        await expect(page.locator('.validation-summary')).toContainText(/Blocked/i)
        await mkdir(evidenceDir, { recursive: true })
        await page.screenshot({ path: testInfo.outputPath(`${candidate.name}.png`), fullPage: true })
        await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}.png`), fullPage: true })
        await writeFile(path.join(evidenceDir, `${candidate.name}.json`), JSON.stringify({
          case: candidate.name,
          project_id: fixture.id,
          definition_id: candidate.definitionId,
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
    }
  })
})
