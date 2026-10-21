// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { mkdir, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
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
type GraphEdge = {
  id: string
  source_node_id: string
  source_port_id: string
  target_node_id: string
  target_port_id: string
}
type TensorSummary = { shape: Array<number | null>; dtype: string }
type ShapeSummary = { node_id: string; parameter_count: number; outputs: Record<string, TensorSummary> }
type ValidationPayload = {
  valid: boolean
  diagnostics: Array<{ code?: string; message?: string; node_id?: string | null }>
  shape_summaries: ShapeSummary[]
  total_parameters: number
}
type RuntimeJob = {
  state: string
  worker?: { api: string; backend: string; device_id: string }
  result?: { status: string; outputs: Array<{ shape: number[]; dtype: string }>; parameter_count: number }
}
type SavedProject = {
  current_revision: number
  validation_status: string
  graph: { nodes: GraphNode[]; edges: GraphEdge[] }
}
type LiveNode = {
  id: string
  version: number
  display_name: string
  category: string
  operation?: { family?: string; shape_behavior?: string; output_dtype?: string; arity?: number; variadic?: boolean } | null
  implementations: Record<string, { supported?: boolean; status?: string; symbol?: string | null }>
}
type LiveCatalog = { schema_version: number; nodes: LiveNode[] }
type RuntimeDiscoveryStatus = { backend: string; api: string; state: string; available: boolean; reason_code?: string | null }
type Slice = 'nw-t3-13' | 'nw-t3-14'
type OperationDefinition = {
  slice: Slice
  slug: string
  displayName: string
  definitionId: string
  inputDtype: 'float32' | 'bool'
  inputShapes: number[][]
  parameterOverrides?: Record<string, unknown>
  expectedStaticShape: Array<number | null>
  expectedRuntimeShape: number[]
  expectedDtype: string
}
type OperationCase = OperationDefinition & {
  name: string
  kerasEnabled: boolean
  worker: { api: 'keras' | 'native'; backend: string; device_id: string }
}

const evidenceDir = path.resolve(
  process.env.NW_VALIDATION_EVIDENCE_DIR
    ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260929', 'evidence', 'nw-t3-13-14-comparison-reduction'),
)

const selectedDefinitions: OperationDefinition[] = [
  { slice: 'nw-t3-13', slug: 'equal', displayName: 'Equal', definitionId: 'math.equal', inputDtype: 'float32', inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'bool' },
  { slice: 'nw-t3-13', slug: 'not-equal', displayName: 'Not Equal', definitionId: 'math.not_equal', inputDtype: 'float32', inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'bool' },
  { slice: 'nw-t3-13', slug: 'greater', displayName: 'Greater Than', definitionId: 'math.greater', inputDtype: 'float32', inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'bool' },
  { slice: 'nw-t3-13', slug: 'greater-equal', displayName: 'Greater or Equal', definitionId: 'math.greater_equal', inputDtype: 'float32', inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'bool' },
  { slice: 'nw-t3-13', slug: 'less', displayName: 'Less Than', definitionId: 'math.less', inputDtype: 'float32', inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'bool' },
  { slice: 'nw-t3-13', slug: 'less-equal', displayName: 'Less or Equal', definitionId: 'math.less_equal', inputDtype: 'float32', inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'bool' },
  { slice: 'nw-t3-13', slug: 'logical-and', displayName: 'Logical AND', definitionId: 'math.logical_and', inputDtype: 'bool', inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'bool' },
  { slice: 'nw-t3-13', slug: 'logical-or', displayName: 'Logical OR', definitionId: 'math.logical_or', inputDtype: 'bool', inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'bool' },
  { slice: 'nw-t3-13', slug: 'logical-xor', displayName: 'Logical XOR', definitionId: 'math.logical_xor', inputDtype: 'bool', inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], expectedDtype: 'bool' },
  { slice: 'nw-t3-14', slug: 'argmax', displayName: 'Argmax', definitionId: 'math.argmax', inputDtype: 'float32', inputShapes: [[2, 3]], parameterOverrides: { axis: 1 }, expectedStaticShape: [null, 3], expectedRuntimeShape: [1, 3], expectedDtype: 'int64' },
  { slice: 'nw-t3-14', slug: 'sum', displayName: 'Sum', definitionId: 'math.sum', inputDtype: 'float32', inputShapes: [[2, 3]], parameterOverrides: { axis: 1, keepdims: true }, expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'float32' },
  { slice: 'nw-t3-14', slug: 'mean', displayName: 'Mean', definitionId: 'math.mean', inputDtype: 'float32', inputShapes: [[2, 3]], parameterOverrides: { axis: 1, keepdims: true }, expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'float32' },
  { slice: 'nw-t3-14', slug: 'min', displayName: 'Minimum Reduction', definitionId: 'math.min', inputDtype: 'float32', inputShapes: [[2, 3]], parameterOverrides: { axis: 1, keepdims: true }, expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'float32' },
  { slice: 'nw-t3-14', slug: 'max', displayName: 'Maximum Reduction', definitionId: 'math.max', inputDtype: 'float32', inputShapes: [[2, 3]], parameterOverrides: { axis: 1, keepdims: true }, expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'float32' },
  { slice: 'nw-t3-14', slug: 'prod', displayName: 'Product Reduction', definitionId: 'math.prod', inputDtype: 'float32', inputShapes: [[2, 3]], parameterOverrides: { axis: 1, keepdims: true }, expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'float32' },
  { slice: 'nw-t3-14', slug: 'all', displayName: 'All', definitionId: 'math.all', inputDtype: 'bool', inputShapes: [[2, 3]], parameterOverrides: { axis: 1, keepdims: true }, expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'bool' },
  { slice: 'nw-t3-14', slug: 'any', displayName: 'Any', definitionId: 'math.any', inputDtype: 'bool', inputShapes: [[2, 3]], parameterOverrides: { axis: 1, keepdims: true }, expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'bool' },
  { slice: 'nw-t3-14', slug: 'variance', displayName: 'Variance', definitionId: 'math.variance', inputDtype: 'float32', inputShapes: [[2, 3]], parameterOverrides: { axis: 1, keepdims: true }, expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'float32' },
  { slice: 'nw-t3-14', slug: 'stddev', displayName: 'Standard Deviation', definitionId: 'math.stddev', inputDtype: 'float32', inputShapes: [[2, 3]], parameterOverrides: { axis: 1, keepdims: true }, expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'float32' },
  { slice: 'nw-t3-14', slug: 'logsumexp', displayName: 'Log-Sum-Exp', definitionId: 'math.logsumexp', inputDtype: 'float32', inputShapes: [[2, 3]], parameterOverrides: { axis: 1, keepdims: true }, expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'float32' },
  { slice: 'nw-t3-14', slug: 'generic-sum', displayName: 'Reduction Operations', definitionId: 'math.reduce', inputDtype: 'float32', inputShapes: [[2, 3]], parameterOverrides: { operation: 'sum', axis: 1, keepdims: true }, expectedStaticShape: [null, 1, 3], expectedRuntimeShape: [1, 1, 3], expectedDtype: 'float32' },
]

const expectedSymbols: Record<string, Record<string, string>> = {
  'math.equal': { keras: 'keras.ops.equal', tensorflow: 'tf.equal', pytorch: 'torch.eq', jax: 'jnp.equal' },
  'math.not_equal': { keras: 'keras.ops.not_equal', tensorflow: 'tf.not_equal', pytorch: 'torch.ne', jax: 'jnp.not_equal' },
  'math.greater': { keras: 'keras.ops.greater', tensorflow: 'tf.greater', pytorch: 'torch.gt', jax: 'jnp.greater' },
  'math.greater_equal': { keras: 'keras.ops.greater_equal', tensorflow: 'tf.greater_equal', pytorch: 'torch.ge', jax: 'jnp.greater_equal' },
  'math.less': { keras: 'keras.ops.less', tensorflow: 'tf.less', pytorch: 'torch.lt', jax: 'jnp.less' },
  'math.less_equal': { keras: 'keras.ops.less_equal', tensorflow: 'tf.less_equal', pytorch: 'torch.le', jax: 'jnp.less_equal' },
  'math.logical_and': { keras: 'keras.ops.logical_and', tensorflow: 'tf.logical_and', pytorch: 'torch.logical_and', jax: 'jnp.logical_and' },
  'math.logical_or': { keras: 'keras.ops.logical_or', tensorflow: 'tf.logical_or', pytorch: 'torch.logical_or', jax: 'jnp.logical_or' },
  'math.logical_xor': { keras: 'keras.ops.logical_xor', tensorflow: 'tf.math.logical_xor', pytorch: 'torch.logical_xor', jax: 'jnp.logical_xor' },
  'math.argmax': { keras: 'keras.ops.argmax', tensorflow: 'tf.argmax', pytorch: 'torch.argmax', jax: 'jnp.argmax' },
  'math.sum': { keras: 'keras.ops.sum', tensorflow: 'tf.reduce_sum', pytorch: 'torch.sum', jax: 'jnp.sum' },
  'math.mean': { keras: 'keras.ops.mean', tensorflow: 'tf.reduce_mean', pytorch: 'torch.mean', jax: 'jnp.mean' },
  'math.min': { keras: 'keras.ops.min', tensorflow: 'tf.reduce_min', pytorch: 'torch.amin', jax: 'jnp.min' },
  'math.max': { keras: 'keras.ops.max', tensorflow: 'tf.reduce_max', pytorch: 'torch.amax', jax: 'jnp.max' },
  'math.prod': { keras: 'keras.ops.prod', tensorflow: 'tf.reduce_prod', pytorch: 'torch.prod', jax: 'jnp.prod' },
  'math.all': { keras: 'keras.ops.all', tensorflow: 'tf.reduce_all', pytorch: 'torch.all', jax: 'jnp.all' },
  'math.any': { keras: 'keras.ops.any', tensorflow: 'tf.reduce_any', pytorch: 'torch.any', jax: 'jnp.any' },
  'math.variance': { keras: 'keras.ops.var', tensorflow: 'tf.math.reduce_variance', pytorch: 'torch.var', jax: 'jnp.var' },
  'math.stddev': { keras: 'keras.ops.std', tensorflow: 'tf.math.reduce_std', pytorch: 'torch.std', jax: 'jnp.std' },
  'math.logsumexp': { keras: 'keras.ops.logsumexp', tensorflow: 'tf.math.reduce_logsumexp', pytorch: 'torch.logsumexp', jax: 'jax.scipy.special.logsumexp' },
  'math.reduce': { keras: 'keras.ops.sum', tensorflow: 'tf.reduce_sum', pytorch: 'torch.sum', jax: 'jnp.sum' },
}

const operationCases: OperationCase[] = selectedDefinitions.flatMap((definition) => (
  [
    { api: 'keras' as const, kerasEnabled: true, profileSlug: 'keras-pytorch' },
    { api: 'native' as const, kerasEnabled: false, profileSlug: 'native-pytorch' },
  ].map((profile) => ({
    ...definition,
    name: `${definition.slice}-${definition.slug}-${profile.profileSlug}`,
    kerasEnabled: profile.kerasEnabled,
    worker: { api: profile.api, backend: 'pytorch', device_id: 'cpu' },
  }))
))

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map((item) => stableValue(item))
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>
    return Object.fromEntries(Object.keys(record).sort().map((key) => [key, stableValue(record[key])]))
  }
  return value
}

function catalogFingerprint(catalog: LiveCatalog): string {
  return createHash('sha256').update(JSON.stringify(stableValue(catalog))).digest('hex')
}

function graphShell(nodes: GraphNode[], edges: GraphEdge[] = []) {
  return {
    schema_version: 6,
    nodes,
    edges,
    model_configuration: {},
    export_settings: { include_comments: true, include_shape_assertions: true, preferred_binary_format: 'keras' },
    future_training_configuration: {},
  }
}

function node(id: string, definitionId: string, label: string, position: { x: number; y: number }, parameters: Record<string, unknown>): GraphNode {
  return { id, definition_id: definitionId, definition_version: 1, label, position, parameters }
}

function graphForCase(candidate: OperationCase) {
  const nodes = candidate.inputShapes.map((shape, index) => node(
    `input-${index + 1}`,
    'input.tensor',
    `Input ${index + 1}`,
    { x: 48, y: 80 + index * 160 },
    { shape, dtype: candidate.inputDtype },
  ))
  nodes.push(node('operation-output', 'output.custom', 'Operation output', { x: 1040, y: 240 }, {}))
  return graphShell(nodes)
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

async function createProject(request: APIRequestContext, name: string, graph: ReturnType<typeof graphShell>, kerasEnabled: boolean): Promise<CreatedProject> {
  const response = await request.post('/api/projects', { data: { name, description: 'NW-T3-13/NW-T3-14 comparison and reduction validation fixture', execution_backend: 'pytorch', keras_enabled: kerasEnabled, graph } })
  expect(response.ok()).toBeTruthy()
  return response.json() as Promise<CreatedProject>
}

async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
  const response = await request.delete(`/api/projects/${id}`)
  expect([204, 404]).toContain(response.status())
}

async function addCatalogNode(page: Page, displayName: string, definitionId: string): Promise<string> {
  const nodes = page.locator('[data-node-id]')
  const before = new Set(await nodes.evaluateAll((elements) => elements.map((element) => element.getAttribute('data-node-id')).filter(Boolean) as string[]))
  await page.getByRole('textbox', { name: 'Search nodes' }).fill(definitionId)
  const item = page.getByRole('button', { name: new RegExp(`${escapeRegExp(displayName)}\\s+${escapeRegExp(definitionId)}\\s·`) }).first()
  await expect(item, `${definitionId} should be available in the live catalog`).toBeEnabled()
  await item.click()
  await expect(nodes).toHaveCount(before.size + 1)
  const after = await nodes.evaluateAll((elements) => elements.map((element) => element.getAttribute('data-node-id')).filter(Boolean) as string[])
  const created = after.find((nodeId) => !before.has(nodeId))
  expect(created, `Builder should materialize ${definitionId}`).toBeTruthy()
  await page.getByRole('textbox', { name: 'Search nodes' }).fill('')
  return created!
}

function parameterField(page: Page, displayName: string) {
  return page.getByRole('complementary', { name: 'Properties inspector' }).locator('label.inspector-field').filter({ hasText: displayName }).first()
}

async function configureParameters(page: Page, candidate: OperationCase): Promise<void> {
  const displayNames: Record<string, string> = { axis: 'Axis', keepdims: 'Keepdims', operation: 'Operation' }
  for (const [parameterId, value] of Object.entries(candidate.parameterOverrides ?? {})) {
    const field = parameterField(page, displayNames[parameterId] ?? parameterId)
    await expect(field, `${parameterId} should be rendered in the operation inspector`).toBeVisible()
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

async function connectByPorts(page: Page, sourceLabel: string, targetLabel: string, targetPort: string, expectedConnectionCount: number): Promise<void> {
  const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
  const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
  const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
  const target = targetNode.getByRole('button', { name: new RegExp(`^Port ${targetPort} \\(input\\)`) })
  await expect(source).toBeVisible()
  await expect(target).toBeVisible()
  await source.press('Enter')
  await expect(source).toHaveAttribute('aria-pressed', 'true')
  const responsePromise = page.waitForResponse((response) => response.url().endsWith('/api/validation/connection') && response.request().method() === 'POST')
  await target.press('Enter')
  const response = await responsePromise
  expect(response.ok()).toBeTruthy()
  const payload = await response.json() as { valid: boolean }
  expect(payload.valid, `${sourceLabel} → ${targetLabel}.${targetPort} should be accepted`).toBe(true)
  await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
}

async function ensureRuntimeLane(request: APIRequestContext, api: 'keras' | 'native'): Promise<RuntimeDiscoveryStatus> {
  const started = await request.post('/api/execution/discovery', { data: { backend: 'pytorch', api, force_refresh: true } })
  expect(started.ok()).toBeTruthy()
  let status: RuntimeDiscoveryStatus | undefined
  const deadline = Date.now() + 90_000
  while (Date.now() < deadline) {
    const response = await request.get(`/api/execution/discovery?backend=pytorch&api=${api}`)
    expect(response.ok()).toBeTruthy()
    status = await response.json() as RuntimeDiscoveryStatus
    if (status.state !== 'checking') break
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  expect(status).toMatchObject({ backend: 'pytorch', api, state: 'verified', available: true })
  return status!
}

async function validateSaveAndRun(page: Page, request: APIRequestContext, fixture: CreatedProject, candidate: OperationCase, operationNodeId: string, browserErrors: string[], failedRequests: string[]): Promise<void> {
  const targetPort = candidate.inputShapes.length > 1 ? 'Inputs' : 'Input'
  for (let index = 0; index < candidate.inputShapes.length; index += 1) await connectByPorts(page, `Input ${index + 1}`, candidate.displayName, targetPort, index + 1)
  await connectByPorts(page, candidate.displayName, 'Operation output', 'Input', candidate.inputShapes.length + 1)

  const validationResponsePromise = page.waitForResponse((response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Validate', exact: true }).click()
  const validationResponse = await validationResponsePromise
  expect(validationResponse.ok()).toBeTruthy()
  const validation = await validationResponse.json() as ValidationPayload
  expect(validation.valid).toBe(true)
  expect(validation.diagnostics).toEqual([])
  const summary = validation.shape_summaries.find((item) => item.node_id === operationNodeId)
  expect(summary?.outputs.output).toMatchObject({ shape: candidate.expectedStaticShape, dtype: candidate.expectedDtype })
  expect(summary?.parameter_count).toBe(0)
  expect(validation.total_parameters).toBe(0)
  await expect(page.locator('.validation-summary')).toContainText('Ready')

  const saveResponsePromise = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT')
  await page.getByRole('button', { name: 'Save model', exact: true }).click()
  const saveResponse = await saveResponsePromise
  expect(saveResponse.ok()).toBeTruthy()
  const saved = await saveResponse.json() as SavedProject
  expect(saved.validation_status).toBe('valid')
  expect(saved.graph.nodes).toHaveLength(candidate.inputShapes.length + 2)
  expect(saved.graph.edges).toHaveLength(candidate.inputShapes.length + 1)
  expect(saved.graph.nodes.filter((item) => item.definition_id === candidate.definitionId)).toHaveLength(1)
  expect(saved.graph.nodes.find((item) => item.id === operationNodeId)?.parameters).toMatchObject(candidate.parameterOverrides ?? {})

  await page.reload()
  await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('[data-node-id]')).toHaveCount(candidate.inputShapes.length + 2)
  await expect(page.getByRole('group', { name: `Node ${candidate.displayName}`, exact: true })).toBeVisible()
  const reloadedResponse = await request.get(`/api/projects/${fixture.id}`)
  expect(reloadedResponse.ok()).toBeTruthy()
  const reloaded = await reloadedResponse.json() as SavedProject
  expect(reloaded.graph).toEqual(saved.graph)
  expect(reloaded.current_revision).toBe(saved.current_revision)
  await mkdir(evidenceDir, { recursive: true })
  await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-builder-ready.png`), fullPage: true })

  await page.goto(`/models/${fixture.id}/validation`)
  await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText(candidate.kerasEnabled ? /keras.*pytorch/i : /native.*pytorch/i).first()).toBeVisible({ timeout: 75_000 })
  const submitResponsePromise = page.waitForResponse((response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST')
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
  await mkdir(evidenceDir, { recursive: true })
  await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-runtime-succeeded.png`), fullPage: true })
  await writeFile(path.join(evidenceDir, `${candidate.name}.json`), JSON.stringify({ case: candidate.name, slice: candidate.slice, project_id: fixture.id, definition_id: candidate.definitionId, input_dtype: candidate.inputDtype, profile: { execution_backend: 'pytorch', keras_enabled: candidate.kerasEnabled, worker: finalJob.worker }, saved_revision: saved.current_revision, graph: { node_count: saved.graph.nodes.length, edge_count: saved.graph.edges.length, exact_reload_match: true }, expected: { static_output_shape: candidate.expectedStaticShape, runtime_output_shape: candidate.expectedRuntimeShape, output_dtype: candidate.expectedDtype, parameter_count: 0 }, validation, submitted_job_id: submitted.id, final_job: finalJob, browser_errors: browserErrors, failed_requests: failedRequests, cleanup: 'project deleted in finally block' }, null, 2))
  await writeFile(path.join(evidenceDir, `${candidate.name}-browser-errors.json`), JSON.stringify({ browser_errors: browserErrors, failed_requests: failedRequests }, null, 2))
}

test.describe('NW-T3-13/NW-T3-14 comparison, logical, and reduction catalog slices', () => {
  test.describe.configure({ mode: 'serial' })
  test.setTimeout(240_000)

  test('live catalog crosswalk and Mathematical category render cover comparison and reduction families', async ({ page, request }) => {
    const response = await request.get('/api/nodes/catalog')
    expect(response.ok()).toBeTruthy()
    const catalog = await response.json() as LiveCatalog
    const discovery = [await ensureRuntimeLane(request, 'keras'), await ensureRuntimeLane(request, 'native')]
    const mathNodes = catalog.nodes.filter((item) => item.category === 'mathematical').sort((left, right) => left.id.localeCompare(right.id))
    const selectedIds = selectedDefinitions.map((definition) => definition.definitionId)
    const carryForwardIds = ['math.abs', 'math.power', 'math.softmax', 'math.add', 'math.matmul', 'math.argmin', 'math.reduce', 'math.cast', 'math.transpose', 'math.expand_dims', 'math.expression', 'math.slice', 'math.index']
    const selected = selectedIds.map((id) => {
      const item = catalog.nodes.find((candidate) => candidate.id === id)
      expect(item, `${id} must exist in the live catalog`).toBeTruthy()
      return item!
    })
    for (const item of selected) {
      const definition = selectedDefinitions.find((candidate) => candidate.definitionId === item.id)!
      expect(item.operation).toMatchObject({ family: definition.slice === 'nw-t3-13' ? 'elementwise' : 'reduction', shape_behavior: definition.slice === 'nw-t3-13' ? 'preserve' : 'reduce', arity: definition.slice === 'nw-t3-13' ? 2 : 1 })
      if (definition.slice === 'nw-t3-13') expect(item.operation?.output_dtype).toBe('bool')
      for (const [implementation, symbol] of Object.entries(expectedSymbols[item.id])) {
        expect(item.implementations[implementation]?.supported, `${item.id} ${implementation} mapping should be enabled`).toBe(true)
        expect(item.implementations[implementation]?.symbol).toBe(symbol)
      }
    }
    const crosswalk = mathNodes.map((item) => ({ id: item.id, display_name: item.display_name, status: selectedIds.includes(item.id) ? 'primary-nw-t3-13-14' : carryForwardIds.includes(item.id) ? 'carry-forward' : 'untested-outside-current-scope', operation: item.operation ?? null, implementations: item.implementations }))

    await page.goto('/builder')
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
    await page.getByRole('button', { name: /Mathematical \d+ nodes/ }).click()
    for (const item of selected) await expect(page.getByRole('button', { name: new RegExp(`${escapeRegExp(item.display_name)} ${escapeRegExp(item.id)} ·`) })).toBeEnabled()
    await mkdir(evidenceDir, { recursive: true })
    await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-13-14-mathematical-comparison-reduction-catalog-rendered.png'), fullPage: true })
    await writeFile(path.join(evidenceDir, 'nw-t3-13-14-live-catalog.json'), JSON.stringify({ schema_version: catalog.schema_version, catalog_count: catalog.nodes.length, mathematical_count: mathNodes.length, normalized_catalog_fingerprint_sha256: catalogFingerprint(catalog), selected_ids: selectedIds, carry_forward_ids: carryForwardIds, crosswalk, runtime_discovery: discovery }, null, 2))
  })

  for (const candidate of operationCases) {
    test(`${candidate.name} materializes, persists, and completes exact-profile functional validation`, async ({ page, request }) => {
      const fixture = await createProject(request, `Tier 3 ${candidate.slice} ${candidate.definitionId} ${candidate.kerasEnabled ? 'keras' : 'native'} ${Date.now()}`, graphForCase(candidate), candidate.kerasEnabled)
      const browserErrors: string[] = []
      const failedRequests: string[] = []
      page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
      page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
      page.on('requestfailed', (event) => failedRequests.push(`${event.method()} ${event.url()} ${event.failure()?.errorText ?? 'unknown'}`))
      page.on('response', (event) => { if (event.status() >= 400 && event.url().includes('/api/')) failedRequests.push(`${event.status()} ${event.request().method()} ${event.url()}`) })
      try {
        await page.goto(`/builder/${fixture.id}`)
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        if (!candidate.kerasEnabled) await expect(page.getByLabel('Enable Keras', { exact: true })).not.toBeChecked()
        const operationNodeId = await addCatalogNode(page, candidate.displayName, candidate.definitionId)
        await configureParameters(page, candidate)
        await validateSaveAndRun(page, request, fixture, candidate, operationNodeId, browserErrors, failedRequests)
      } finally {
        await deleteProject(request, fixture.id)
      }
    })
  }

  test('all NW-T3-13/NW-T3-14 disposable projects are deleted after the serial suite', async ({ request }) => {
    const response = await request.get('/api/projects')
    expect(response.ok()).toBeTruthy()
    const payload = await response.json() as { projects: Array<{ id: string; name: string }> }
    expect(payload.projects).toEqual([])
    await mkdir(evidenceDir, { recursive: true })
    await writeFile(path.join(evidenceDir, 'nw-t3-13-14-project-cleanup.json'), JSON.stringify({ projects_remaining_after_run: payload.projects.length, cleanup: 'PASS' }, null, 2))
  })
})
