import { mkdir, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Locator, type Page } from '@playwright/test'

type CreatedProject = { id: string; current_revision: number; name: string }
type GraphNode = {
  id: string
  definition_id: string
  definition_version: number
  label: string
  position: { x: number; y: number }
  parameters: Record<string, unknown>
}
type ValidationDiagnostic = {
  code?: string
  message?: string
  node_id?: string | null
  parameter_id?: string | null
}
type ShapeSummary = {
  node_id: string
  parameter_count: number
  outputs: Record<string, { shape: Array<number | null> }>
}
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
    outputs: Array<{ shape: number[] }>
    parameter_count: number
  }
}
type SavedProject = {
  current_revision: number
  validation_status: string
  graph: { nodes: GraphNode[]; edges: unknown[] }
}
type TransformerKind = 'single' | 'decoder' | 'residual' | 'mask'
type TransformerCase = {
  name: string
  catalogNode: string
  definitionId: string
  kind: TransformerKind
  parameters: Record<string, unknown>
  staticOutputShape: Array<number | null>
  runtimeOutputShape: number[]
  parameterCount: number
  nodeCount: number
  edgeCount: number
  kerasEnabled: boolean
  worker: { api: string; backend: string; device_id: string }
}

const evidenceDir = path.resolve(
  process.env.NW_VALIDATION_EVIDENCE_DIR
    ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260929', 'evidence', 'nw-t3-07-transformer'),
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

function initialGraph(kind: TransformerKind) {
  const nodes = kind === 'decoder'
    ? [
        node('query-input', 'input.sequence', 'Query Sequence', { x: 48, y: 120 }, { shape: [5, 8], dtype: 'float32' }),
        node('context-input', 'input.sequence', 'Context Sequence', { x: 48, y: 360 }, { shape: [3, 8], dtype: 'float32' }),
        node('custom-output', 'output.custom', 'Custom Output', { x: 1120, y: 240 }, {}),
      ]
    : kind === 'residual'
      ? [
          node('left-input', 'input.sequence', 'Left Sequence', { x: 48, y: 120 }, { shape: [5, 8], dtype: 'float32' }),
          node('right-input', 'input.sequence', 'Right Sequence', { x: 48, y: 360 }, { shape: [5, 8], dtype: 'float32' }),
          node('custom-output', 'output.custom', 'Custom Output', { x: 1120, y: 240 }, {}),
        ]
      : kind === 'mask'
        ? [
            node('mask-input', 'input.mask', 'Mask Input', { x: 48, y: 240 }, { shape: [5, 5], dtype: 'bool' }),
            node('custom-output', 'output.custom', 'Custom Output', { x: 1120, y: 240 }, {}),
          ]
        : [
            node('sequence-input', 'input.sequence', 'Input Sequence', { x: 48, y: 240 }, { shape: [5, 8], dtype: 'float32' }),
            node('custom-output', 'output.custom', 'Custom Output', { x: 1120, y: 240 }, {}),
          ]
  return graphShell(nodes)
}

async function createProject(
  request: APIRequestContext,
  name: string,
  kind: TransformerKind,
  kerasEnabled: boolean,
): Promise<CreatedProject> {
  const response = await request.post('/api/projects', {
    data: {
      name,
      description: 'NW-T3-07 Transformer block validation fixture',
      execution_backend: 'pytorch',
      keras_enabled: kerasEnabled,
      graph: initialGraph(kind),
    },
  })
  expect(response.ok()).toBeTruthy()
  return response.json() as Promise<CreatedProject>
}

async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
  const response = await request.delete('/api/projects/' + id)
  expect([204, 404]).toContain(response.status())
}

async function addCatalogNode(page: Page, displayName: string, definitionId: string): Promise<string> {
  const nodes = page.locator('[data-node-id]')
  const before = new Set(
    await nodes.evaluateAll((elements) => elements
      .map((element) => element.getAttribute('data-node-id'))
      .filter(Boolean) as string[]),
  )
  await page.getByRole('textbox', { name: 'Search nodes' }).fill(displayName)
  const item = page.locator('.catalog-item').filter({ hasText: definitionId }).first()
  await expect(item, 'Catalog node ' + displayName + ' should be available').toBeEnabled()
  await item.click()
  await expect(nodes).toHaveCount(before.size + 1)
  const after = await nodes.evaluateAll((elements) => elements
    .map((element) => element.getAttribute('data-node-id'))
    .filter(Boolean) as string[])
  const created = after.find((nodeId) => !before.has(nodeId))
  expect(created, 'Builder should materialize ' + displayName).toBeTruthy()
  await page.getByRole('textbox', { name: 'Search nodes' }).fill('')
  return created!
}

async function connectByPorts(
  page: Page,
  sourceLabel: string,
  targetLabel: string,
  targetPortLabel: string,
  expectedConnectionCount: number,
): Promise<void> {
  const sourceNode = page.getByRole('group', { name: 'Node ' + sourceLabel, exact: true })
  const targetNode = page.getByRole('group', { name: 'Node ' + targetLabel, exact: true })
  const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
  const target = targetNode.getByRole('button', { name: new RegExp('^Port ' + targetPortLabel + ' \\(input\\)') })
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
  expect(payload.valid, sourceLabel + ' → ' + targetLabel + '.' + targetPortLabel + ' should be accepted').toBe(true)
  await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
}

async function rejectConnectionByPorts(
  page: Page,
  sourceLabel: string,
  targetLabel: string,
  targetPortLabel: string,
): Promise<{ valid: boolean; diagnostics?: unknown[] }> {
  const sourceNode = page.getByRole('group', { name: 'Node ' + sourceLabel, exact: true })
  const targetNode = page.getByRole('group', { name: 'Node ' + targetLabel, exact: true })
  const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
  const target = targetNode.getByRole('button', { name: new RegExp('^Port ' + targetPortLabel + ' \\(input\\)') })
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
  const payload = await response.json() as { valid: boolean; diagnostics?: unknown[] }
  expect(payload.valid, sourceLabel + ' → ' + targetLabel + '.' + targetPortLabel + ' should be rejected').toBe(false)
  return payload
}

async function parameterField(inspector: Locator, displayName: string | RegExp): Promise<Locator> {
  const field = inspector.locator('label.inspector-field').filter({ hasText: displayName }).first()
  if (!(await field.isVisible())) {
    const group = inspector.locator('details.parameter-group').filter({ hasText: displayName }).first()
    await group.locator('summary').click()
  }
  await expect(field, displayName.toString() + ' should be visible in the Inspector').toBeVisible()
  return field
}

async function setTransformerParameters(
  page: Page,
  nodeId: string,
  parameters: Record<string, unknown>,
): Promise<void> {
  await page.locator('[data-node-id="' + nodeId + '"]').click()
  const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
  const displayNames: Record<string, string | RegExp> = {
    num_heads: /Num heads/i,
    key_dim: /Key dimension/i,
    dropout: /Dropout/i,
    layers: /Layers/i,
    ffn_multiplier: /Feed-forward multiplier/i,
    pre_norm: /Pre-normalization/i,
  }
  for (const [id, value] of Object.entries(parameters)) {
    const displayName = displayNames[id]
    if (!displayName) continue
    const field = await parameterField(inspector, displayName)
    if (typeof value === 'boolean') {
      await field.locator('input[type="checkbox"]').setChecked(value)
    } else {
      const input = field.locator('input').first()
      if (await input.isEnabled()) await input.fill(String(value))
      await expect(input).toHaveValue(String(value))
    }
  }
}

async function validateGraph(page: Page): Promise<ValidationPayload> {
  const validationResponsePromise = page.waitForResponse(
    (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Validate', exact: true }).click()
  const validationResponse = await validationResponsePromise
  expect(validationResponse.ok()).toBeTruthy()
  return validationResponse.json() as Promise<ValidationPayload>
}

async function connectCase(page: Page, candidate: TransformerCase): Promise<void> {
  if (candidate.kind === 'decoder') {
    await connectByPorts(page, 'Query Sequence', candidate.catalogNode, 'Query', 1)
    await connectByPorts(page, 'Context Sequence', candidate.catalogNode, 'Context', 2)
  } else if (candidate.kind === 'residual') {
    await connectByPorts(page, 'Left Sequence', candidate.catalogNode, 'Inputs', 1)
    await connectByPorts(page, 'Right Sequence', candidate.catalogNode, 'Inputs', 2)
  } else if (candidate.kind === 'mask') {
    await connectByPorts(page, 'Mask Input', candidate.catalogNode, 'Input', 1)
  } else {
    const targetPort = candidate.definitionId === 'transformer.positional_encoding' ? 'Input' : 'Query'
    await connectByPorts(page, 'Input Sequence', candidate.catalogNode, targetPort, 1)
  }
  await connectByPorts(page, candidate.catalogNode, 'Custom Output', 'Input', candidate.edgeCount)
}

async function validateSaveAndRun(
  page: Page,
  request: APIRequestContext,
  fixture: CreatedProject,
  nodeId: string,
  candidate: TransformerCase,
  browserErrors: string[],
  failedRequests: string[],
): Promise<{ validation: ValidationPayload; saved: SavedProject; submittedJobId: string; finalJob: RuntimeJob }> {
  const validation = await validateGraph(page)
  expect(validation.valid, JSON.stringify(validation.diagnostics)).toBe(true)
  expect(validation.diagnostics).toEqual([])
  const summary = validation.shape_summaries.find((item) => item.node_id === nodeId)
  expect(summary?.outputs.output?.shape).toEqual(candidate.staticOutputShape)
  expect(summary?.parameter_count).toBe(candidate.parameterCount)
  expect(validation.total_parameters).toBe(candidate.parameterCount)
  await expect(page.locator('.validation-summary')).toContainText('Ready')
  await mkdir(evidenceDir, { recursive: true })
  await page.screenshot({ path: path.join(evidenceDir, candidate.name + '-builder-ready.png'), fullPage: true })

  const saveResponsePromise = page.waitForResponse(
    (response) => response.url().includes('/api/projects/' + fixture.id) && response.request().method() === 'PUT',
  )
  await page.getByRole('button', { name: 'Save model', exact: true }).click()
  const saveResponse = await saveResponsePromise
  expect(saveResponse.ok()).toBeTruthy()
  const saved = await saveResponse.json() as SavedProject
  expect(saved.validation_status).toBe('valid')
  expect(saved.graph.nodes).toHaveLength(candidate.nodeCount)
  expect(saved.graph.edges).toHaveLength(candidate.edgeCount)
  const savedNode = saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)
  expect(savedNode?.parameters).toMatchObject(candidate.parameters)

  await page.reload()
  await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('[data-node-id]')).toHaveCount(candidate.nodeCount)
  await expect(page.getByRole('group', { name: 'Node ' + candidate.catalogNode, exact: true })).toBeVisible()

  await page.goto('/models/' + fixture.id + '/validation')
  await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
  const profilePattern = candidate.kerasEnabled ? /keras.*pytorch/i : /native.*pytorch/i
  await expect(page.getByText(profilePattern).first()).toBeVisible({ timeout: 75_000 })
  await expect(page.getByRole('button', { name: 'Run functional validation', exact: true })).toBeEnabled()

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
    const jobResponse = await request.get('/api/execution/jobs/' + submitted.id)
    if (!jobResponse.ok()) return 'missing'
    finalJob = await jobResponse.json() as RuntimeJob
    return finalJob.state
  }, { timeout: 90_000 }).toBe('succeeded')

  expect(finalJob.worker).toMatchObject(candidate.worker)
  expect(finalJob.result?.status).toBe('succeeded')
  expect(finalJob.result?.outputs[0]?.shape).toEqual(candidate.runtimeOutputShape)
  expect(finalJob.result?.parameter_count).toBe(candidate.parameterCount)
  await page.screenshot({ path: path.join(evidenceDir, candidate.name + '-runtime-succeeded.png'), fullPage: true })
  expect(browserErrors, candidate.name + ' should not emit browser errors').toEqual([])
  expect(failedRequests, candidate.name + ' should not emit failed product requests').toEqual([])

  await writeFile(path.join(evidenceDir, candidate.name + '.json'), JSON.stringify({
    case: candidate.name,
    project_id: fixture.id,
    definition_id: candidate.definitionId,
    authored_parameters: candidate.parameters,
    profile: {
      execution_backend: 'pytorch',
      keras_enabled: candidate.kerasEnabled,
      worker: finalJob.worker,
    },
    saved_revision: saved.current_revision,
    graph: { node_count: saved.graph.nodes.length, edge_count: saved.graph.edges.length },
    validation,
    submitted_job_id: submitted.id,
    final_job: finalJob,
    browser_errors: browserErrors,
    failed_requests: failedRequests,
    cleanup: 'project deleted in finally block',
  }, null, 2))
  await writeFile(path.join(evidenceDir, candidate.name + '-browser-errors.json'), JSON.stringify({
    browser_errors: browserErrors,
    failed_requests: failedRequests,
  }, null, 2))
  return { validation, saved, submittedJobId: submitted.id, finalJob }
}

const transformerCases: TransformerCase[] = [
  {
    name: 'nw-t3-07-encoder-keras-pytorch',
    catalogNode: 'Transformer Encoder Block',
    definitionId: 'transformer.encoder_block',
    kind: 'single',
    parameters: { num_heads: 2, key_dim: 4, layers: 2, dropout: 0, ffn_multiplier: 2, pre_norm: true },
    staticOutputShape: [null, 5, 8],
    runtimeOutputShape: [1, 5, 8],
    parameterCount: 1200,
    nodeCount: 3,
    edgeCount: 2,
    kerasEnabled: true,
    worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
  },
  {
    name: 'nw-t3-07-decoder-keras-pytorch',
    catalogNode: 'Transformer Decoder Block',
    definitionId: 'transformer.decoder_block',
    kind: 'decoder',
    parameters: { num_heads: 2, key_dim: 4, layers: 1, dropout: 0, ffn_multiplier: 2, pre_norm: true },
    staticOutputShape: [null, 5, 8],
    runtimeOutputShape: [1, 5, 8],
    parameterCount: 904,
    nodeCount: 4,
    edgeCount: 3,
    kerasEnabled: true,
    worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
  },
  {
    name: 'nw-t3-07-feed-forward-keras-pytorch',
    catalogNode: 'Feed-forward Transformer Block',
    definitionId: 'transformer.feed_forward',
    kind: 'single',
    parameters: { ffn_multiplier: 2, dropout: 0 },
    staticOutputShape: [null, 5, 8],
    runtimeOutputShape: [1, 5, 8],
    parameterCount: 280,
    nodeCount: 3,
    edgeCount: 2,
    kerasEnabled: true,
    worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
  },
  {
    name: 'nw-t3-07-stack-keras-pytorch',
    catalogNode: 'Complete Transformer Stack',
    definitionId: 'transformer.stack',
    kind: 'single',
    parameters: { num_heads: 2, key_dim: 4, layers: 2, dropout: 0, ffn_multiplier: 2, pre_norm: false },
    staticOutputShape: [null, 5, 8],
    runtimeOutputShape: [1, 5, 8],
    parameterCount: 1200,
    nodeCount: 3,
    edgeCount: 2,
    kerasEnabled: true,
    worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
  },
  {
    name: 'nw-t3-07-positional-encoding-keras-pytorch',
    catalogNode: 'Positional Encoding',
    definitionId: 'transformer.positional_encoding',
    kind: 'single',
    parameters: {},
    staticOutputShape: [null, 5, 8],
    runtimeOutputShape: [1, 5, 8],
    parameterCount: 0,
    nodeCount: 3,
    edgeCount: 2,
    kerasEnabled: true,
    worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
  },
  {
    name: 'nw-t3-07-residual-keras-pytorch',
    catalogNode: 'Residual Connection',
    definitionId: 'transformer.residual',
    kind: 'residual',
    parameters: {},
    staticOutputShape: [null, 5, 8],
    runtimeOutputShape: [1, 5, 8],
    parameterCount: 0,
    nodeCount: 4,
    edgeCount: 3,
    kerasEnabled: true,
    worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
  },
  {
    name: 'nw-t3-07-attention-mask-keras-pytorch',
    catalogNode: 'Attention Mask',
    definitionId: 'transformer.attention_mask',
    kind: 'mask',
    parameters: {},
    staticOutputShape: [null, 5, 5],
    runtimeOutputShape: [1, 5, 5],
    parameterCount: 0,
    nodeCount: 3,
    edgeCount: 2,
    kerasEnabled: true,
    worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
  },
  {
    name: 'nw-t3-07-encoder-native-pytorch',
    catalogNode: 'Transformer Encoder Block',
    definitionId: 'transformer.encoder_block',
    kind: 'single',
    parameters: { num_heads: 2, key_dim: 4, layers: 2, dropout: 0, ffn_multiplier: 2, pre_norm: true },
    staticOutputShape: [null, 5, 8],
    runtimeOutputShape: [1, 5, 8],
    parameterCount: 1200,
    nodeCount: 3,
    edgeCount: 2,
    kerasEnabled: false,
    worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' },
  },
]

test.describe('NW-T3-07 Transformer block catalog slice', () => {
  test.describe.configure({ mode: 'serial' })
  test.setTimeout(240_000)

  test('live catalog renders every Transformer definition with declared mappings', async ({ page, request }) => {
    const catalogResponse = await request.get('/api/nodes/catalog')
    expect(catalogResponse.ok()).toBeTruthy()
    const catalog = await catalogResponse.json() as {
      schema_version: number
      nodes: Array<{
        id: string
        display_name: string
        category: string
        implementations: Record<string, { supported?: boolean }>
      }>
    }
    const transformer = catalog.nodes
      .filter((item) => item.id.startsWith('transformer.'))
      .sort((left, right) => left.id.localeCompare(right.id))
    expect(transformer.map((item) => item.id)).toEqual([
      'transformer.attention_mask',
      'transformer.decoder_block',
      'transformer.encoder_block',
      'transformer.feed_forward',
      'transformer.positional_encoding',
      'transformer.residual',
      'transformer.stack',
    ])
    for (const item of transformer) {
      expect(item.implementations.keras?.supported).toBe(true)
      expect(item.implementations.tensorflow?.supported).toBe(true)
      expect(item.implementations.pytorch?.supported).toBe(true)
      expect(item.implementations.jax?.supported).toBe(true)
    }

    await page.goto('/builder')
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
    await page.getByRole('button', { name: /Transformer 10 nodes/ }).click()
    for (const candidate of transformerCases.slice(0, 7)) {
      await expect(page.getByRole('button', {
        name: new RegExp(candidate.catalogNode + ' ' + candidate.definitionId + ' · operation'),
      })).toBeEnabled()
    }
    await mkdir(evidenceDir, { recursive: true })
    await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-07-transformer-catalog-rendered.png'), fullPage: true })
    await writeFile(path.join(evidenceDir, 'nw-t3-07-live-catalog.json'), JSON.stringify({
      schema_version: catalog.schema_version,
      catalog_count: catalog.nodes.length,
      category_counts: catalog.nodes.reduce<Record<string, number>>((counts, item) => {
        counts[item.category] = (counts[item.category] ?? 0) + 1
        return counts
      }, {}),
      catalog_fingerprint_sha256: createHash('sha256').update(JSON.stringify(catalog)).digest('hex'),
      transformer,
    }, null, 2))
  })

  for (const candidate of transformerCases.filter((item) => item.kind !== 'mask')) {
    test(candidate.name + ' builds, persists, and completes exact-profile functional validation', async ({ page, request }) => {
      const fixture = await createProject(
        request,
        'Tier 3 Transformer ' + candidate.catalogNode + ' ' + Date.now(),
        candidate.kind,
        candidate.kerasEnabled,
      )
      const browserErrors: string[] = []
      const failedRequests: string[] = []
      page.on('pageerror', (error) => browserErrors.push('pageerror: ' + error.message))
      page.on('console', (message) => { if (message.type() === 'error') browserErrors.push('console: ' + message.text()) })
      page.on('requestfailed', (requestEvent) => failedRequests.push(requestEvent.method() + ' ' + requestEvent.url() + ' ' + (requestEvent.failure()?.errorText ?? 'unknown')))
      page.on('response', (response) => {
        if (response.status() >= 400 && response.url().includes('/api/')) failedRequests.push(response.status() + ' ' + response.request().method() + ' ' + response.url())
      })
      try {
        await page.goto('/builder/' + fixture.id)
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        const transformerNodeId = await addCatalogNode(page, candidate.catalogNode, candidate.definitionId)
        await setTransformerParameters(page, transformerNodeId, candidate.parameters)
        await connectCase(page, candidate)
        await validateSaveAndRun(page, request, fixture, transformerNodeId, candidate, browserErrors, failedRequests)
      } finally {
        await deleteProject(request, fixture.id)
      }
    })
  }

  test('attention mask preserves its mask type and rejects a tensor-only output', async ({ page, request }) => {
    const candidate = transformerCases.find((item) => item.kind === 'mask')!
    const fixture = await createProject(
      request,
      'Tier 3 Transformer Attention Mask boundary ' + Date.now(),
      candidate.kind,
      candidate.kerasEnabled,
    )
    const browserErrors: string[] = []
    const failedRequests: string[] = []
    page.on('pageerror', (error) => browserErrors.push('pageerror: ' + error.message))
    page.on('console', (message) => { if (message.type() === 'error') browserErrors.push('console: ' + message.text()) })
    page.on('requestfailed', (requestEvent) => failedRequests.push(requestEvent.method() + ' ' + requestEvent.url() + ' ' + (requestEvent.failure()?.errorText ?? 'unknown')))
    page.on('response', (response) => {
      if (response.status() >= 400 && response.url().includes('/api/')) failedRequests.push(response.status() + ' ' + response.request().method() + ' ' + response.url())
    })
    try {
      await page.goto('/builder/' + fixture.id)
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
      await addCatalogNode(page, candidate.catalogNode, candidate.definitionId)
      await connectByPorts(page, 'Mask Input', candidate.catalogNode, 'Input', 1)
      const rejected = await rejectConnectionByPorts(page, candidate.catalogNode, 'Custom Output', 'Input')
      await mkdir(evidenceDir, { recursive: true })
      await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-07-attention-mask-type-boundary.png'), fullPage: true })
      await writeFile(path.join(evidenceDir, 'nw-t3-07-attention-mask-type-boundary.json'), JSON.stringify({
        case: 'nw-t3-07-attention-mask-type-boundary',
        project_id: fixture.id,
        definition_id: candidate.definitionId,
        input_connection: 'Mask Input → Attention Mask.Input accepted',
        rejected_connection: 'Attention Mask → Custom Output.Input',
        validation: rejected,
        execution_job_submissions: 0,
        browser_errors: browserErrors,
        failed_requests: failedRequests,
        limitation: 'The current catalog has no mask-capable terminal consumer, so standalone attention-mask runtime execution is not certified in this slice.',
      }, null, 2))
      expect(browserErrors).toEqual([])
      expect(failedRequests).toEqual([])
    } finally {
      await deleteProject(request, fixture.id)
    }
  })

  test('encoder invalid layer count is blocked by the Builder, repaired, persisted, and executed', async ({ page, request }) => {
    const candidate = transformerCases[0]!
    const repaired: TransformerCase = {
      ...candidate,
      name: 'nw-t3-07-encoder-repaired',
      parameters: { ...candidate.parameters, layers: 1 },
      parameterCount: 600,
    }
    const fixture = await createProject(request, 'Tier 3 Transformer repair ' + Date.now(), candidate.kind, true)
    const browserErrors: string[] = []
    const failedRequests: string[] = []
    let jobSubmitCount = 0
    page.on('pageerror', (error) => browserErrors.push('pageerror: ' + error.message))
    page.on('console', (message) => { if (message.type() === 'error') browserErrors.push('console: ' + message.text()) })
    page.on('requestfailed', (requestEvent) => failedRequests.push(requestEvent.method() + ' ' + requestEvent.url() + ' ' + (requestEvent.failure()?.errorText ?? 'unknown')))
    page.on('request', (requestEvent) => {
      if (requestEvent.method() === 'POST' && requestEvent.url().includes('/api/execution/jobs')) jobSubmitCount += 1
    })
    try {
      await page.goto('/builder/' + fixture.id)
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
      const transformerNodeId = await addCatalogNode(page, candidate.catalogNode, candidate.definitionId)
      await setTransformerParameters(page, transformerNodeId, candidate.parameters)
      await connectCase(page, candidate)

      const validBeforeRepair = await validateGraph(page)
      expect(validBeforeRepair.valid).toBe(true)
      const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
      const layers = (await parameterField(inspector, /Layers/i)).locator('input')
      await layers.fill('0')
      await expect(layers).toHaveValue('0')
      await expect(inspector.locator('.parameter-error').filter({ hasText: /valid layers/i })).toBeVisible()
      await expect(page.getByRole('button', { name: 'Validate', exact: true })).toBeDisabled()
      await expect(page.getByRole('button', { name: 'Save model', exact: true })).toBeDisabled()
      await mkdir(evidenceDir, { recursive: true })
      await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-07-invalid-layers.png'), fullPage: true })
      await writeFile(path.join(evidenceDir, 'nw-t3-07-invalid-layers.json'), JSON.stringify({
        case: 'nw-t3-07-invalid-layers',
        project_id: fixture.id,
        authored_parameters: { ...candidate.parameters, layers: 0 },
        ui_parameter_error: 'Enter a valid layers.',
        validate_action: 'disabled',
        save_action: 'disabled',
        server_validation_submitted: false,
        execution_job_submissions: jobSubmitCount,
        browser_errors: browserErrors,
        failed_requests: failedRequests,
      }, null, 2))

      await layers.fill('1')
      await expect(layers).toHaveValue('1')
      await expect(page.getByRole('button', { name: 'Validate', exact: true })).toBeEnabled()
      const completed = await validateSaveAndRun(page, request, fixture, transformerNodeId, repaired, browserErrors, failedRequests)
      expect(completed.saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)?.parameters.layers).toBe(1)
      await writeFile(path.join(evidenceDir, 'nw-t3-07-repaired-layers.json'), JSON.stringify({
        case: 'nw-t3-07-repaired-layers',
        project_id: fixture.id,
        invalid_ui_state: {
          layers: 0,
          validate_action: 'disabled',
          save_action: 'disabled',
        },
        repaired_saved_revision: completed.saved.current_revision,
        repaired_graph: {
          node_count: completed.saved.graph.nodes.length,
          edge_count: completed.saved.graph.edges.length,
          layers: completed.saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)?.parameters.layers,
        },
        submitted_job_id: completed.submittedJobId,
        final_job: completed.finalJob,
        execution_job_submissions: jobSubmitCount,
        browser_errors: browserErrors,
        failed_requests: failedRequests,
      }, null, 2))
      await page.goto('/builder/' + fixture.id)
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
      const persistedNode = page.getByRole('group', { name: 'Node ' + candidate.catalogNode, exact: true })
      await persistedNode.click()
      const persistedInspector = page.getByRole('complementary', { name: 'Properties inspector' })
      await expect((await parameterField(persistedInspector, /Layers/i)).locator('input')).toHaveValue('1')
      expect(jobSubmitCount).toBe(1)
      expect(browserErrors).toEqual([])
      expect(failedRequests).toEqual([])
    } finally {
      await deleteProject(request, fixture.id)
    }
  })
})
