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
type ShapeSummary = {
  node_id: string
  parameter_count: number
  outputs: Record<string, TensorSummary>
}
type ValidationPayload = {
  valid: boolean
  diagnostics: Array<{ code?: string; message?: string; node_id?: string | null }>
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
type LiveNode = {
  id: string
  version: number
  display_name: string
  category: string
  inputs: Array<{ id: string; display_name: string; kind: string }>
  outputs: Array<{ id: string; display_name: string; kind: string }>
  parameters: Array<{ id: string; display_name: string }>
  operation?: {
    family?: string
    shape_behavior?: string
    output_dtype?: string
    arity?: number
    variadic?: boolean
  } | null
  implementations: Record<string, { supported?: boolean; status?: string; symbol?: string | null }>
}
type LiveCatalog = { schema_version: number; nodes: LiveNode[] }
type RuntimeDiscoveryStatus = {
  backend: string
  api: string
  state: string
  available: boolean
  reason_code?: string | null
  source?: string
  checked_at?: string | null
}
type UnaryDefinition = {
  slice: 'nw-t3-11' | 'nw-t3-12'
  slug: string
  displayName: string
  definitionId: string
  parameterOverrides?: Record<string, unknown>
}
type UnaryCase = UnaryDefinition & {
  name: string
  kerasEnabled: boolean
  inputShapes: number[][]
  expectedStaticShape: Array<number | null>
  expectedRuntimeShape: number[]
  expectedDtype: string
  worker: { api: string; backend: string; device_id: string }
}

const evidenceDir = path.resolve(
  process.env.NW_VALIDATION_EVIDENCE_DIR
    ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260929', 'evidence', 'nw-t3-11-12-unary-mathematics'),
)

const selectedDefinitions: UnaryDefinition[] = [
  { slice: 'nw-t3-11', slug: 'ceil', displayName: 'Ceiling', definitionId: 'math.ceil' },
  { slice: 'nw-t3-11', slug: 'floor', displayName: 'Floor', definitionId: 'math.floor' },
  { slice: 'nw-t3-11', slug: 'round', displayName: 'Round', definitionId: 'math.round' },
  { slice: 'nw-t3-11', slug: 'sin', displayName: 'Sine', definitionId: 'math.sin' },
  { slice: 'nw-t3-11', slug: 'cos', displayName: 'Cosine', definitionId: 'math.cos' },
  { slice: 'nw-t3-11', slug: 'tan', displayName: 'Tangent', definitionId: 'math.tan' },
  { slice: 'nw-t3-11', slug: 'exp', displayName: 'Exponential', definitionId: 'math.exp' },
  { slice: 'nw-t3-11', slug: 'log', displayName: 'Logarithm', definitionId: 'math.log' },
  { slice: 'nw-t3-11', slug: 'log1p', displayName: 'Logarithm 1p', definitionId: 'math.log1p' },
  { slice: 'nw-t3-11', slug: 'erf', displayName: 'Error Function', definitionId: 'math.erf' },
  { slice: 'nw-t3-11', slug: 'sqrt', displayName: 'Square Root', definitionId: 'math.sqrt' },
  { slice: 'nw-t3-11', slug: 'rsqrt', displayName: 'Reciprocal Square Root', definitionId: 'math.rsqrt' },
  { slice: 'nw-t3-11', slug: 'square', displayName: 'Square', definitionId: 'math.square' },
  { slice: 'nw-t3-11', slug: 'negative', displayName: 'Negative', definitionId: 'math.negative' },
  { slice: 'nw-t3-11', slug: 'reciprocal', displayName: 'Reciprocal', definitionId: 'math.reciprocal' },
  { slice: 'nw-t3-11', slug: 'sign', displayName: 'Sign', definitionId: 'math.sign' },
  { slice: 'nw-t3-11', slug: 'clip', displayName: 'Clip', definitionId: 'math.clip', parameterOverrides: { min_value: -1, max_value: 1 } },
  { slice: 'nw-t3-12', slug: 'sinh', displayName: 'Hyperbolic Sine', definitionId: 'math.sinh' },
  { slice: 'nw-t3-12', slug: 'cosh', displayName: 'Hyperbolic Cosine', definitionId: 'math.cosh' },
  { slice: 'nw-t3-12', slug: 'tanh', displayName: 'Hyperbolic Tangent', definitionId: 'math.tanh' },
  { slice: 'nw-t3-12', slug: 'relu', displayName: 'ReLU', definitionId: 'math.relu' },
  { slice: 'nw-t3-12', slug: 'gelu', displayName: 'GELU', definitionId: 'math.gelu' },
  { slice: 'nw-t3-12', slug: 'sigmoid', displayName: 'Sigmoid', definitionId: 'math.sigmoid' },
  { slice: 'nw-t3-12', slug: 'softplus', displayName: 'Softplus', definitionId: 'math.softplus' },
]

const expectedSymbols: Record<string, Record<string, string>> = {
  'math.ceil': { keras: 'keras.ops.ceil', tensorflow: 'tf.math.ceil', pytorch: 'torch.ceil', jax: 'jnp.ceil' },
  'math.floor': { keras: 'keras.ops.floor', tensorflow: 'tf.math.floor', pytorch: 'torch.floor', jax: 'jnp.floor' },
  'math.round': { keras: 'keras.ops.round', tensorflow: 'tf.math.round', pytorch: 'torch.round', jax: 'jnp.round' },
  'math.sin': { keras: 'keras.ops.sin', tensorflow: 'tf.math.sin', pytorch: 'torch.sin', jax: 'jnp.sin' },
  'math.cos': { keras: 'keras.ops.cos', tensorflow: 'tf.math.cos', pytorch: 'torch.cos', jax: 'jnp.cos' },
  'math.tan': { keras: 'keras.ops.tan', tensorflow: 'tf.math.tan', pytorch: 'torch.tan', jax: 'jnp.tan' },
  'math.exp': { keras: 'keras.ops.exp', tensorflow: 'tf.math.exp', pytorch: 'torch.exp', jax: 'jnp.exp' },
  'math.log': { keras: 'keras.ops.log', tensorflow: 'tf.math.log', pytorch: 'torch.log', jax: 'jnp.log' },
  'math.log1p': { keras: 'keras.ops.log1p', tensorflow: 'tf.math.log1p', pytorch: 'torch.log1p', jax: 'jnp.log1p' },
  'math.erf': { keras: 'keras.ops.erf', tensorflow: 'tf.math.erf', pytorch: 'torch.erf', jax: 'jax.scipy.special.erf' },
  'math.sqrt': { keras: 'keras.ops.sqrt', tensorflow: 'tf.math.sqrt', pytorch: 'torch.sqrt', jax: 'jnp.sqrt' },
  'math.rsqrt': { keras: 'keras.ops.rsqrt', tensorflow: 'tf.math.rsqrt', pytorch: 'torch.rsqrt', jax: 'jax.lax.rsqrt' },
  'math.square': { keras: 'keras.ops.square', tensorflow: 'tf.math.square', pytorch: 'torch.square', jax: 'jnp.square' },
  'math.negative': { keras: 'keras.ops.negative', tensorflow: 'tf.math.negative', pytorch: 'torch.neg', jax: 'jnp.negative' },
  'math.reciprocal': { keras: 'keras.ops.reciprocal', tensorflow: 'tf.math.reciprocal', pytorch: 'torch.reciprocal', jax: 'jnp.reciprocal' },
  'math.sign': { keras: 'keras.ops.sign', tensorflow: 'tf.math.sign', pytorch: 'torch.sign', jax: 'jnp.sign' },
  'math.clip': { keras: 'keras.ops.clip', tensorflow: 'tf.clip_by_value', pytorch: 'torch.clamp', jax: 'jnp.clip' },
  'math.sinh': { keras: 'keras.ops.sinh', tensorflow: 'tf.math.sinh', pytorch: 'torch.sinh', jax: 'jnp.sinh' },
  'math.cosh': { keras: 'keras.ops.cosh', tensorflow: 'tf.math.cosh', pytorch: 'torch.cosh', jax: 'jnp.cosh' },
  'math.tanh': { keras: 'keras.ops.tanh', tensorflow: 'tf.math.tanh', pytorch: 'torch.tanh', jax: 'jnp.tanh' },
  'math.relu': { keras: 'keras.ops.relu', tensorflow: 'tf.nn.relu', pytorch: 'torch.relu', jax: 'jax.nn.relu' },
  'math.gelu': { keras: 'keras.ops.gelu', tensorflow: 'tf.nn.gelu', pytorch: 'torch.nn.functional.gelu', jax: 'jax.nn.gelu' },
  'math.sigmoid': { keras: 'keras.ops.sigmoid', tensorflow: 'tf.nn.sigmoid', pytorch: 'torch.sigmoid', jax: 'jax.nn.sigmoid' },
  'math.softplus': { keras: 'keras.ops.softplus', tensorflow: 'tf.nn.softplus', pytorch: 'torch.nn.functional.softplus', jax: 'jax.nn.softplus' },
}

const mathematicalCases: UnaryCase[] = selectedDefinitions.flatMap((definition) => (
  [
    { api: 'keras', kerasEnabled: true, profileSlug: 'keras-pytorch' },
    { api: 'native', kerasEnabled: false, profileSlug: 'native-pytorch' },
  ].map((profile) => ({
    ...definition,
    name: `${definition.slice}-${definition.slug}-${profile.profileSlug}`,
    kerasEnabled: profile.kerasEnabled,
    inputShapes: [[4]],
    expectedStaticShape: [null, 4],
    expectedRuntimeShape: [1, 4],
    expectedDtype: 'float32',
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

function node(
  id: string,
  definitionId: string,
  label: string,
  position: { x: number; y: number },
  parameters: Record<string, unknown>,
): GraphNode {
  return { id, definition_id: definitionId, definition_version: 1, label, position, parameters }
}

function graphForCase(candidate: UnaryCase) {
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

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

async function createProject(
  request: APIRequestContext,
  name: string,
  graph: ReturnType<typeof graphShell>,
  kerasEnabled: boolean,
): Promise<CreatedProject> {
  const response = await request.post('/api/projects', {
    data: { name, description: 'NW-T3-11/NW-T3-12 unary mathematical validation fixture', execution_backend: 'pytorch', keras_enabled: kerasEnabled, graph },
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
  const before = new Set(await nodes.evaluateAll((elements) => elements.map((element) => element.getAttribute('data-node-id')).filter(Boolean) as string[]))
  await page.getByRole('textbox', { name: 'Search nodes' }).fill(definitionId)
  const item = page.locator('.catalog-item').filter({
    hasText: new RegExp(`${escapeRegExp(displayName)}${escapeRegExp(definitionId)}\\s·`),
  }).first()
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
  const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
  return inspector.locator('label.inspector-field').filter({ hasText: displayName }).first()
}

async function configureParameters(page: Page, candidate: UnaryCase): Promise<void> {
  const displayNames: Record<string, string> = { min_value: 'Minimum', max_value: 'Maximum' }
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
  const responsePromise = page.waitForResponse((response) => response.url().endsWith('/api/validation/connection') && response.request().method() === 'POST')
  await target.press('Enter')
  const response = await responsePromise
  expect(response.ok()).toBeTruthy()
  const payload = await response.json() as { valid: boolean }
  expect(payload.valid, `${sourceLabel} → ${targetLabel}.${targetPort} should be accepted`).toBe(true)
  await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
}

async function ensureRuntimeLane(request: APIRequestContext, api: 'keras' | 'native'): Promise<RuntimeDiscoveryStatus> {
  const started = await request.post('/api/execution/discovery', {
    data: { backend: 'pytorch', api, force_refresh: true },
  })
  expect(started.ok(), `start exact PyTorch/${api} runtime discovery`).toBeTruthy()

  let status: RuntimeDiscoveryStatus | undefined
  const deadline = Date.now() + 90_000
  while (Date.now() < deadline) {
    const response = await request.get(`/api/execution/discovery?backend=pytorch&api=${api}`)
    expect(response.ok(), `read exact PyTorch/${api} runtime discovery`).toBeTruthy()
    status = await response.json() as RuntimeDiscoveryStatus
    if (status.state !== 'checking') break
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  expect(status, `PyTorch/${api} discovery status`).toMatchObject({
    backend: 'pytorch',
    api,
    state: 'verified',
    available: true,
  })
  return status!
}

async function validateSaveAndRun(
  page: Page,
  request: APIRequestContext,
  fixture: CreatedProject,
  candidate: UnaryCase,
  operationNodeId: string,
  browserErrors: string[],
  failedRequests: string[],
): Promise<void> {
  await expect(page.getByRole('group', { name: `Node ${candidate.displayName}`, exact: true }).getByRole('button', { name: /^Port Input \(input\)/ })).toBeVisible()
  await expect(page.getByRole('group', { name: `Node ${candidate.displayName}`, exact: true }).getByRole('button', { name: /^Port Output \(output\)/ })).toBeVisible()
  await connectByPorts(page, 'Input 1', candidate.displayName, 'Input', 1)
  await connectByPorts(page, candidate.displayName, 'Operation output', 'Input', 2)

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
  expect(saved.graph.nodes).toHaveLength(3)
  expect(saved.graph.edges).toHaveLength(2)
  expect(saved.graph.nodes.filter((item) => item.definition_id === candidate.definitionId)).toHaveLength(1)
  expect(saved.graph.nodes.find((item) => item.id === operationNodeId)?.parameters).toMatchObject(candidate.parameterOverrides ?? {})

  await page.reload()
  await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('[data-node-id]')).toHaveCount(3)
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
  await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-runtime-succeeded.png`), fullPage: true })
  await writeFile(path.join(evidenceDir, `${candidate.name}.json`), JSON.stringify({
    case: candidate.name,
    slice: candidate.slice,
    project_id: fixture.id,
    definition_id: candidate.definitionId,
    profile: { execution_backend: 'pytorch', keras_enabled: candidate.kerasEnabled, worker: finalJob.worker },
    saved_revision: saved.current_revision,
    graph: { node_count: saved.graph.nodes.length, edge_count: saved.graph.edges.length, exact_reload_match: true },
    input_policy: 'Functional Validation uses deterministic non-zero dummy inputs after the NW-T3-11/NW-T3-12 fixture correction.',
    expected: { static_output_shape: candidate.expectedStaticShape, runtime_output_shape: candidate.expectedRuntimeShape, output_dtype: candidate.expectedDtype, parameter_count: 0 },
    validation,
    submitted_job_id: submitted.id,
    final_job: finalJob,
    browser_errors: browserErrors,
    failed_requests: failedRequests,
    cleanup: 'project deleted in finally block',
  }, null, 2))
  await writeFile(path.join(evidenceDir, `${candidate.name}-browser-errors.json`), JSON.stringify({ browser_errors: browserErrors, failed_requests: failedRequests }, null, 2))
}

test.describe('NW-T3-11/NW-T3-12 unary mathematical family catalog slice', () => {
  test.describe.configure({ mode: 'serial' })
  test.setTimeout(240_000)

  test('live catalog crosswalk and Mathematical category render cover the selected unary families', async ({ page, request }) => {
    const response = await request.get('/api/nodes/catalog')
    expect(response.ok()).toBeTruthy()
    const catalog = await response.json() as LiveCatalog
    const metadataResponse = await request.get('/api/metadata')
    expect(metadataResponse.ok()).toBeTruthy()
    const metadataBeforeDiscovery = await metadataResponse.json() as Record<string, unknown>
    const discovery = [
      await ensureRuntimeLane(request, 'keras'),
      await ensureRuntimeLane(request, 'native'),
    ]
    const refreshedMetadataResponse = await request.get('/api/metadata')
    expect(refreshedMetadataResponse.ok()).toBeTruthy()
    const metadata = await refreshedMetadataResponse.json() as Record<string, unknown>
    const mathNodes = catalog.nodes.filter((item) => item.category === 'mathematical').sort((left, right) => left.id.localeCompare(right.id))
    const selectedIds = selectedDefinitions.map((definition) => definition.definitionId)
    const carryForwardIds = ['math.abs', 'math.power', 'math.softmax', 'math.add', 'math.matmul', 'math.argmin', 'math.reduce', 'math.cast', 'math.transpose', 'math.expand_dims', 'math.expression', 'math.slice', 'math.index']
    const selected = selectedIds.map((id) => {
      const item = catalog.nodes.find((candidate) => candidate.id === id)
      expect(item, `${id} must exist in the live catalog`).toBeTruthy()
      return item!
    })
    for (const item of selected) {
      expect(item.category).toBe('mathematical')
      expect(item.operation).toMatchObject({ family: 'unary', shape_behavior: 'preserve', output_dtype: 'same', arity: 1 })
      for (const [implementation, symbol] of Object.entries(expectedSymbols[item.id])) {
        expect(item.implementations[implementation]?.supported, `${item.id} ${implementation} mapping should be enabled`).toBe(true)
        expect(item.implementations[implementation]?.symbol).toBe(symbol)
      }
    }
    const crosswalk = mathNodes.map((item) => ({
      id: item.id,
      display_name: item.display_name,
      status: selectedIds.includes(item.id)
        ? 'primary-nw-t3-11-12'
        : carryForwardIds.includes(item.id) ? 'carry-forward-nw-t3-10' : 'untested-outside-current-scope',
      operation: item.operation ?? null,
      implementations: item.implementations,
    }))

    await page.goto('/builder')
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
    await page.getByRole('button', { name: /Mathematical \d+ nodes/ }).click()
    for (const item of selected) await expect(page.getByRole('button', { name: new RegExp(`${item.display_name} ${item.id} ·`) })).toBeEnabled()
    await mkdir(evidenceDir, { recursive: true })
    await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-11-12-mathematical-unary-catalog-rendered.png'), fullPage: true })
    await writeFile(path.join(evidenceDir, 'nw-t3-11-12-live-catalog.json'), JSON.stringify({
      schema_version: catalog.schema_version,
      catalog_count: catalog.nodes.length,
      mathematical_count: mathNodes.length,
      normalized_catalog_fingerprint_sha256: catalogFingerprint(catalog),
      selected_ids: selectedIds,
      carry_forward_ids: carryForwardIds,
      crosswalk,
      runtime_discovery: discovery,
      metadata_before_discovery: metadataBeforeDiscovery,
      metadata,
    }, null, 2))
  })

  for (const candidate of mathematicalCases) {
    test(`${candidate.name} materializes, persists, and completes exact-profile functional validation`, async ({ page, request }) => {
      const fixture = await createProject(request, `Tier 3 unary ${candidate.definitionId} ${candidate.kerasEnabled ? 'keras' : 'native'} ${Date.now()}`, graphForCase(candidate), candidate.kerasEnabled)
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

  test('all NW-T3-11/NW-T3-12 disposable projects are deleted after the serial suite', async ({ request }) => {
    const response = await request.get('/api/projects')
    expect(response.ok()).toBeTruthy()
    const payload = await response.json() as { projects: Array<{ id: string; name: string }> }
    const projects = payload.projects
    expect(projects).toEqual([])
    await mkdir(evidenceDir, { recursive: true })
    await writeFile(path.join(evidenceDir, 'nw-t3-11-12-project-cleanup.json'), JSON.stringify({ projects_remaining_after_run: projects.length, cleanup: 'PASS' }, null, 2))
  })
})
