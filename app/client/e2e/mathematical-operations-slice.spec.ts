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
type TensorSummary = { shape: Array<number | null>; dtype: string }
type ShapeSummary = {
  node_id: string
  parameter_count: number
  outputs: Record<string, TensorSummary>
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
  result?: {
    status: string
    outputs: Array<{ shape: number[]; dtype: string }>
    parameter_count: number
  }
}
type SavedProject = {
  current_revision: number
  validation_status: string
  graph: { nodes: GraphNode[]; edges: GraphEdge[] }
}
type MathematicalCase = {
  name: string
  displayName: string
  definitionId: string
  kerasEnabled: boolean
  inputShapes: number[][]
  expectedStaticShape: Array<number | null>
  expectedRuntimeShape: number[]
  expectedDtype: string
  parameterOverrides?: Record<string, unknown>
  worker: { api: string; backend: string; device_id: string }
}

const evidenceDir = path.resolve(
  process.env.NW_VALIDATION_EVIDENCE_DIR
    ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260929', 'evidence', 'nw-t3-10-mathematical-operations'),
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

function graphForCase(candidate: MathematicalCase) {
  const nodes = candidate.inputShapes.map((shape, index) => node(
    `input-${index + 1}`,
    'input.tensor',
    `Input ${index + 1}`,
    { x: 48, y: 80 + index * 160 },
    { shape, dtype: 'float32' },
  ))
  nodes.push(node('operation-output', 'output.custom', 'Operation output', { x: 1040, y: 240 }, {}))
  return graphShell(nodes)
}

function prebuiltGraph(
  definitionId: string,
  inputShapes: number[][],
  parameters: Record<string, unknown>,
) {
  const nodes = inputShapes.map((shape, index) => node(
    `input-${index + 1}`,
    'input.tensor',
    `Input ${index + 1}`,
    { x: 48, y: 80 + index * 160 },
    { shape, dtype: 'float32' },
  ))
  nodes.push(node('operation', definitionId, 'Mathematical operation', { x: 560, y: 240 }, parameters))
  nodes.push(node('operation-output', 'output.custom', 'Operation output', { x: 1040, y: 240 }, {}))
  const targetPortId = inputShapes.length > 1 ? 'inputs' : 'input'
  const edges: GraphEdge[] = inputShapes.map((_, index) => ({
    id: `input-${index + 1}-operation`,
    source_node_id: `input-${index + 1}`,
    source_port_id: 'output',
    target_node_id: 'operation',
    target_port_id: targetPortId,
  }))
  edges.push({
    id: 'operation-output',
    source_node_id: 'operation',
    source_port_id: 'output',
    target_node_id: 'operation-output',
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
      description: 'NW-T3-10 mathematical operation validation fixture',
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

function parameterField(page: Page, displayName: string) {
  const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
  return inspector.locator('label.inspector-field').filter({ hasText: displayName }).first()
}

async function configureParameters(page: Page, candidate: MathematicalCase): Promise<void> {
  const displayNames: Record<string, string> = {
    axis: 'Axis',
    axes: 'Axes',
    dtype: 'Data Type',
    exponent: 'Exponent',
    expression: 'Expression',
    index: 'Index',
    keepdims: 'Keepdims',
    multiples: 'Multiples',
    operation: 'Operation',
    repeats: 'Repeats',
    target_shape: 'Target Shape',
  }
  for (const [parameterId, value] of Object.entries(candidate.parameterOverrides ?? {})) {
    const displayName = candidate.definitionId === 'math.slice' && parameterId === 'expression'
      ? 'Slice expression'
      : displayNames[parameterId] ?? parameterId
    const field = parameterField(page, displayName)
    if (!(await field.isVisible())) {
      const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
      const group = inspector.locator('details.parameter-group').filter({ hasText: displayName }).first()
      await group.locator('summary').click()
    }
    await expect(field, `${displayName} should be rendered in the operation inspector`).toBeVisible()
    const select = field.locator('select').first()
    if (await select.count()) {
      await select.selectOption(String(value))
      continue
    }
    const checkbox = field.locator('input[type="checkbox"]').first()
    if (await checkbox.count()) {
      if (value) await checkbox.check()
      else await checkbox.uncheck()
      continue
    }
    const control = field.locator('textarea, input').first()
    await expect(control).toBeVisible()
    await control.fill(typeof value === 'string' ? value : JSON.stringify(value))
    await control.press('Tab')
  }
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
  candidate: MathematicalCase,
  operationNodeId: string,
  browserErrors: string[],
  failedRequests: string[],
  testInfo: TestInfo,
): Promise<void> {
  const targetPort = candidate.inputShapes.length > 1 ? 'Inputs' : 'Input'
  for (let index = 0; index < candidate.inputShapes.length; index += 1) {
    await connectByPorts(page, `Input ${index + 1}`, candidate.displayName, targetPort, index + 1)
  }
  await connectByPorts(page, candidate.displayName, 'Operation output', 'Input', candidate.inputShapes.length + 1)

  const validationResponsePromise = page.waitForResponse(
    (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Validate', exact: true }).click()
  const validationResponse = await validationResponsePromise
  expect(validationResponse.ok()).toBeTruthy()
  const validation = await validationResponse.json() as ValidationPayload
  expect(validation.valid).toBe(true)
  expect(validation.diagnostics).toEqual([])
  const summary = validation.shape_summaries.find((item) => item.node_id === operationNodeId)
  expect(summary?.outputs.output).toMatchObject({
    shape: candidate.expectedStaticShape,
    dtype: candidate.expectedDtype,
  })
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
  const savedOperation = saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)
  expect(savedOperation).toBeTruthy()
  if (candidate.parameterOverrides) expect(savedOperation?.parameters).toMatchObject(candidate.parameterOverrides)

  await page.reload()
  await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('[data-node-id]')).toHaveCount(candidate.inputShapes.length + 2)
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
  expect(finalJob.result?.outputs[0]).toMatchObject({ shape: candidate.expectedRuntimeShape, dtype: candidate.expectedDtype })
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
      output_dtype: candidate.expectedDtype,
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
}

const mathematicalCases: MathematicalCase[] = [
  { name: 'nw-t3-10-abs-keras-pytorch', displayName: 'Absolute Value', definitionId: 'math.abs', kerasEnabled: true, inputShapes: [[4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'float32', worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-abs-native-pytorch', displayName: 'Absolute Value', definitionId: 'math.abs', kerasEnabled: false, inputShapes: [[4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'float32', worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-power-keras-pytorch', displayName: 'Power', definitionId: 'math.power', kerasEnabled: true, inputShapes: [[4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'float32', parameterOverrides: { exponent: 3 }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-power-native-pytorch', displayName: 'Power', definitionId: 'math.power', kerasEnabled: false, inputShapes: [[4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'float32', parameterOverrides: { exponent: 3 }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-softmax-keras-pytorch', displayName: 'Softmax', definitionId: 'math.softmax', kerasEnabled: true, inputShapes: [[2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', parameterOverrides: { axis: -1 }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-softmax-native-pytorch', displayName: 'Softmax', definitionId: 'math.softmax', kerasEnabled: false, inputShapes: [[2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', parameterOverrides: { axis: -1 }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-add-keras-pytorch', displayName: 'Add', definitionId: 'math.add', kerasEnabled: true, inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'float32', worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-add-native-pytorch', displayName: 'Add', definitionId: 'math.add', kerasEnabled: false, inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'float32', worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-matmul-keras-pytorch', displayName: 'Matrix Multiplication', definitionId: 'math.matmul', kerasEnabled: true, inputShapes: [[2, 3], [3, 2]], expectedStaticShape: [null, 2, 2], expectedRuntimeShape: [1, 2, 2], expectedDtype: 'float32', worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-matmul-native-pytorch', displayName: 'Matrix Multiplication', definitionId: 'math.matmul', kerasEnabled: false, inputShapes: [[2, 3], [3, 2]], expectedStaticShape: [null, 2, 2], expectedRuntimeShape: [1, 2, 2], expectedDtype: 'float32', worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-argmin-keras-pytorch', displayName: 'Argmin', definitionId: 'math.argmin', kerasEnabled: true, inputShapes: [[2, 3]], expectedStaticShape: [null, 2], expectedRuntimeShape: [1, 2], expectedDtype: 'int64', parameterOverrides: { axis: -1 }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-argmin-native-pytorch', displayName: 'Argmin', definitionId: 'math.argmin', kerasEnabled: false, inputShapes: [[2, 3]], expectedStaticShape: [null, 2], expectedRuntimeShape: [1, 2], expectedDtype: 'int64', parameterOverrides: { axis: -1 }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-reduce-keras-pytorch', displayName: 'Reduction Operations', definitionId: 'math.reduce', kerasEnabled: true, inputShapes: [[2, 3]], expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'float32', parameterOverrides: { operation: 'mean', axis: 1, keepdims: true }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-reduce-native-pytorch', displayName: 'Reduction Operations', definitionId: 'math.reduce', kerasEnabled: false, inputShapes: [[2, 3]], expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'float32', parameterOverrides: { operation: 'mean', axis: 1, keepdims: true }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-cast-keras-pytorch', displayName: 'Cast', definitionId: 'math.cast', kerasEnabled: true, inputShapes: [[4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'int32', parameterOverrides: { dtype: 'int32' }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-cast-native-pytorch', displayName: 'Cast', definitionId: 'math.cast', kerasEnabled: false, inputShapes: [[4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'int32', parameterOverrides: { dtype: 'int32' }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-transpose-keras-pytorch', displayName: 'Transpose', definitionId: 'math.transpose', kerasEnabled: true, inputShapes: [[2, 3]], expectedStaticShape: [null, 3, 2], expectedRuntimeShape: [1, 3, 2], expectedDtype: 'float32', parameterOverrides: { axes: [0, 2, 1] }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-transpose-native-pytorch', displayName: 'Transpose', definitionId: 'math.transpose', kerasEnabled: false, inputShapes: [[2, 3]], expectedStaticShape: [null, 3, 2], expectedRuntimeShape: [1, 3, 2], expectedDtype: 'float32', parameterOverrides: { axes: [0, 2, 1] }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-expand-dims-keras-pytorch', displayName: 'Expand Dimensions', definitionId: 'math.expand_dims', kerasEnabled: true, inputShapes: [[2, 3]], expectedStaticShape: [null, 1, 2, 3], expectedRuntimeShape: [1, 1, 2, 3], expectedDtype: 'float32', parameterOverrides: { axis: 1 }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-expand-dims-native-pytorch', displayName: 'Expand Dimensions', definitionId: 'math.expand_dims', kerasEnabled: false, inputShapes: [[2, 3]], expectedStaticShape: [null, 1, 2, 3], expectedRuntimeShape: [1, 1, 2, 3], expectedDtype: 'float32', parameterOverrides: { axis: 1 }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-expression-keras-pytorch', displayName: 'Safe Mathematical Expression', definitionId: 'math.expression', kerasEnabled: true, inputShapes: [[4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'float32', parameterOverrides: { expression: 'x * 2 + 1' }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-expression-native-pytorch', displayName: 'Safe Mathematical Expression', definitionId: 'math.expression', kerasEnabled: false, inputShapes: [[4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'float32', parameterOverrides: { expression: 'x * 2 + 1' }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-slice-keras-pytorch', displayName: 'Tensor Slicing', definitionId: 'math.slice', kerasEnabled: true, inputShapes: [[4, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', parameterOverrides: { expression: 'x[:, 1:3]' }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-slice-native-pytorch', displayName: 'Tensor Slicing', definitionId: 'math.slice', kerasEnabled: false, inputShapes: [[4, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', parameterOverrides: { expression: 'x[:, 1:3]' }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-index-keras-pytorch', displayName: 'Tensor Indexing', definitionId: 'math.index', kerasEnabled: true, inputShapes: [[4, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', parameterOverrides: { expression: 'x[:, 1:3]' }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-10-index-native-pytorch', displayName: 'Tensor Indexing', definitionId: 'math.index', kerasEnabled: false, inputShapes: [[4, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', parameterOverrides: { expression: 'x[:, 1:3]' }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-15-divide-keras-pytorch', displayName: 'Divide', definitionId: 'math.divide', kerasEnabled: true, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-15-divide-native-pytorch', displayName: 'Divide', definitionId: 'math.divide', kerasEnabled: false, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-15-dot-keras-pytorch', displayName: 'Dot Product', definitionId: 'math.dot', kerasEnabled: true, inputShapes: [[2, 3], [3, 2]], expectedStaticShape: [null, 2, 2], expectedRuntimeShape: [1, 2, 2], expectedDtype: 'float32', worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-15-dot-native-pytorch', displayName: 'Dot Product', definitionId: 'math.dot', kerasEnabled: false, inputShapes: [[2, 3], [3, 2]], expectedStaticShape: [null, 2, 2], expectedRuntimeShape: [1, 2, 2], expectedDtype: 'float32', worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-15-maximum-keras-pytorch', displayName: 'Maximum', definitionId: 'math.maximum', kerasEnabled: true, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-15-maximum-native-pytorch', displayName: 'Maximum', definitionId: 'math.maximum', kerasEnabled: false, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-15-minimum-keras-pytorch', displayName: 'Minimum', definitionId: 'math.minimum', kerasEnabled: true, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-15-minimum-native-pytorch', displayName: 'Minimum', definitionId: 'math.minimum', kerasEnabled: false, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-15-multiply-keras-pytorch', displayName: 'Multiply', definitionId: 'math.multiply', kerasEnabled: true, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-15-multiply-native-pytorch', displayName: 'Multiply', definitionId: 'math.multiply', kerasEnabled: false, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-15-subtract-keras-pytorch', displayName: 'Subtract', definitionId: 'math.subtract', kerasEnabled: true, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-15-subtract-native-pytorch', displayName: 'Subtract', definitionId: 'math.subtract', kerasEnabled: false, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-concatenate-keras-pytorch', displayName: 'Concatenate', definitionId: 'math.concatenate', kerasEnabled: true, inputShapes: [[2, 3], [2, 4]], expectedStaticShape: [null, 2, 7], expectedRuntimeShape: [1, 2, 7], expectedDtype: 'float32', parameterOverrides: { axis: -1 }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-concatenate-native-pytorch', displayName: 'Concatenate', definitionId: 'math.concatenate', kerasEnabled: false, inputShapes: [[2, 3], [2, 4]], expectedStaticShape: [null, 2, 7], expectedRuntimeShape: [1, 2, 7], expectedDtype: 'float32', parameterOverrides: { axis: -1 }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-flatten-keras-pytorch', displayName: 'Flatten', definitionId: 'math.flatten', kerasEnabled: true, inputShapes: [[2, 3]], expectedStaticShape: [null, 6], expectedRuntimeShape: [1, 6], expectedDtype: 'float32', worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-flatten-native-pytorch', displayName: 'Flatten', definitionId: 'math.flatten', kerasEnabled: false, inputShapes: [[2, 3]], expectedStaticShape: [null, 6], expectedRuntimeShape: [1, 6], expectedDtype: 'float32', worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-ravel-keras-pytorch', displayName: 'Ravel', definitionId: 'math.ravel', kerasEnabled: true, inputShapes: [[2, 3]], expectedStaticShape: [null, 6], expectedRuntimeShape: [1, 6], expectedDtype: 'float32', worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-ravel-native-pytorch', displayName: 'Ravel', definitionId: 'math.ravel', kerasEnabled: false, inputShapes: [[2, 3]], expectedStaticShape: [null, 6], expectedRuntimeShape: [1, 6], expectedDtype: 'float32', worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-repeat-keras-pytorch', displayName: 'Repeat', definitionId: 'math.repeat', kerasEnabled: true, inputShapes: [[2, 3]], expectedStaticShape: [null, 2, 6], expectedRuntimeShape: [1, 2, 6], expectedDtype: 'float32', parameterOverrides: { repeats: 2, axis: 2 }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-repeat-native-pytorch', displayName: 'Repeat', definitionId: 'math.repeat', kerasEnabled: false, inputShapes: [[2, 3]], expectedStaticShape: [null, 2, 6], expectedRuntimeShape: [1, 2, 6], expectedDtype: 'float32', parameterOverrides: { repeats: 2, axis: 2 }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-reshape-keras-pytorch', displayName: 'Reshape', definitionId: 'math.reshape', kerasEnabled: true, inputShapes: [[2, 3]], expectedStaticShape: [null, 3, 2], expectedRuntimeShape: [1, 3, 2], expectedDtype: 'float32', parameterOverrides: { target_shape: [3, 2] }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-reshape-native-pytorch', displayName: 'Reshape', definitionId: 'math.reshape', kerasEnabled: false, inputShapes: [[2, 3]], expectedStaticShape: [null, 3, 2], expectedRuntimeShape: [1, 3, 2], expectedDtype: 'float32', parameterOverrides: { target_shape: [3, 2] }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-squeeze-keras-pytorch', displayName: 'Squeeze Dimensions', definitionId: 'math.squeeze', kerasEnabled: true, inputShapes: [[1, 2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', parameterOverrides: { axis: 1 }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-squeeze-native-pytorch', displayName: 'Squeeze Dimensions', definitionId: 'math.squeeze', kerasEnabled: false, inputShapes: [[1, 2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', parameterOverrides: { axis: 1 }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-stack-keras-pytorch', displayName: 'Stack', definitionId: 'math.stack', kerasEnabled: true, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [2, null, 2, 3], expectedRuntimeShape: [2, 1, 2, 3], expectedDtype: 'float32', parameterOverrides: { axis: 0 }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-stack-native-pytorch', displayName: 'Stack', definitionId: 'math.stack', kerasEnabled: false, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [2, null, 2, 3], expectedRuntimeShape: [2, 1, 2, 3], expectedDtype: 'float32', parameterOverrides: { axis: 0 }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-tile-keras-pytorch', displayName: 'Tile', definitionId: 'math.tile', kerasEnabled: true, inputShapes: [[2, 3]], expectedStaticShape: [null, 4, 9], expectedRuntimeShape: [1, 4, 9], expectedDtype: 'float32', parameterOverrides: { multiples: [1, 2, 3] }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-16-tile-native-pytorch', displayName: 'Tile', definitionId: 'math.tile', kerasEnabled: false, inputShapes: [[2, 3]], expectedStaticShape: [null, 4, 9], expectedRuntimeShape: [1, 4, 9], expectedDtype: 'float32', parameterOverrides: { multiples: [1, 2, 3] }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-17-stop-gradient-keras-pytorch', displayName: 'Stop Gradient', definitionId: 'math.stop_gradient', kerasEnabled: true, inputShapes: [[2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-17-stop-gradient-native-pytorch', displayName: 'Stop Gradient', definitionId: 'math.stop_gradient', kerasEnabled: false, inputShapes: [[2, 3]], expectedStaticShape: [null, 2, 3], expectedRuntimeShape: [1, 2, 3], expectedDtype: 'float32', worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-17-take-keras-pytorch', displayName: 'Take', definitionId: 'math.take', kerasEnabled: true, inputShapes: [[2, 3]], expectedStaticShape: [null, 2], expectedRuntimeShape: [1, 2], expectedDtype: 'float32', parameterOverrides: { index: 1, axis: 2 }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
  { name: 'nw-t3-17-take-native-pytorch', displayName: 'Take', definitionId: 'math.take', kerasEnabled: false, inputShapes: [[2, 3]], expectedStaticShape: [null, 2], expectedRuntimeShape: [1, 2], expectedDtype: 'float32', parameterOverrides: { index: 1, axis: 2 }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
]

test.describe('NW-T3-10 mathematical operation catalog slice', () => {
  test.describe.configure({ mode: 'serial' })
  test.setTimeout(240_000)

  test('live catalog renders the selected mathematical definitions and typed contracts', async ({ page, request }) => {
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
    const selectedIds = ['math.abs', 'math.power', 'math.softmax', 'math.add', 'math.argmin', 'math.reduce', 'math.cast', 'math.transpose', 'math.expand_dims', 'math.expression', 'math.slice', 'math.index']
    const selected = selectedIds.map((id) => catalog.nodes.find((item) => item.id === id))
    expect(selected.every(Boolean)).toBe(true)
    for (const item of selected) {
      expect(item?.category).toBe('mathematical')
      expect(item?.implementations.keras?.supported).toBe(true)
      expect(item?.implementations.pytorch?.supported).toBe(true)
      if (item?.id === 'math.index') expect(item.operation).toBeNull()
      else expect(item?.operation).toBeTruthy()
    }
    expect(selected.find((item) => item?.id === 'math.add')?.operation).toMatchObject({ family: 'elementwise', variadic: true })
    expect(selected.find((item) => item?.id === 'math.argmin')?.operation).toMatchObject({ family: 'reduction', shape_behavior: 'reduce' })
    expect(selected.find((item) => item?.id === 'math.transpose')?.operation).toMatchObject({ family: 'tensor', shape_behavior: 'transpose' })
    expect(selected.find((item) => item?.id === 'math.expression')?.implementations.pytorch).toMatchObject({ adapter: 'safe_expression' })

    await page.goto('/builder')
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
    await page.getByRole('button', { name: /Mathematical \d+ nodes/ }).click()
    for (const item of selected) {
      await expect(page.getByRole('button', { name: new RegExp(`${item!.display_name} ${item!.id} ·`) })).toBeEnabled()
    }
    await mkdir(evidenceDir, { recursive: true })
    await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-10-mathematical-catalog-rendered.png'), fullPage: true })
    await writeFile(path.join(evidenceDir, 'nw-t3-10-live-catalog.json'), JSON.stringify({
      schema_version: catalog.schema_version,
      catalog_count: catalog.nodes.length,
      catalog_fingerprint_sha256: createHash('sha256').update(JSON.stringify(catalog)).digest('hex'),
      selected,
    }, null, 2))
  })

  for (const candidate of mathematicalCases) {
    test(`${candidate.name} builds, persists, and completes exact-profile functional validation`, async ({ page, request }, testInfo) => {
      const fixture = await createProject(
        request,
        `Tier 3 mathematical ${candidate.definitionId} ${candidate.kerasEnabled ? 'keras' : 'native'} ${Date.now()}`,
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
        const operationNodeId = await addCatalogNode(page, candidate.displayName, candidate.definitionId)
        await configureParameters(page, candidate)
        await validateSaveAndRun(page, request, fixture, candidate, operationNodeId, browserErrors, failedRequests, testInfo)
      } finally {
        await deleteProject(request, fixture.id)
      }
    })
  }

  test('unsafe mathematical expressions remain blocked before execution', async ({ page, request }, testInfo) => {
    const fixture = await createProject(
      request,
      `Tier 3 invalid mathematical expression ${Date.now()}`,
      prebuiltGraph('math.expression', [[4]], { expression: "__import__('os')" }),
      true,
    )
    const browserErrors: string[] = []
    const failedRequests: string[] = []
    let executionJobSubmissions = 0
    page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
    page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
    page.on('requestfailed', (event) => failedRequests.push(`${event.method()} ${event.url()} ${event.failure()?.errorText ?? 'unknown'}`))
    page.on('request', (event) => { if (event.url().includes('/api/execution/jobs') && event.method() === 'POST') executionJobSubmissions += 1 })
    page.on('response', (event) => {
      if (event.status() >= 400 && event.url().includes('/api/')) failedRequests.push(`${event.status()} ${event.request().method()} ${event.url()}`)
    })
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
        expect.objectContaining({ message: expect.stringMatching(/safe|expression|allowed/i) }),
      ]))
      await expect(page.locator('.validation-summary')).toContainText(/Blocked/i)
      await mkdir(evidenceDir, { recursive: true })
      await page.screenshot({ path: testInfo.outputPath('nw-t3-10-invalid-expression.png'), fullPage: true })
      await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-10-invalid-expression.png'), fullPage: true })
      await writeFile(path.join(evidenceDir, 'nw-t3-10-invalid-expression.json'), JSON.stringify({
        case: 'nw-t3-10-invalid-expression',
        project_id: fixture.id,
        definition_id: 'math.expression',
        validation,
        browser_errors: browserErrors,
        failed_requests: failedRequests,
        execution_job_submissions: executionJobSubmissions,
      }, null, 2))
      expect(browserErrors).toEqual([])
      expect(failedRequests).toEqual([])
      expect(executionJobSubmissions).toBe(0)
    } finally {
      await deleteProject(request, fixture.id)
    }
  })
})
