// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

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
type ShapeSummary = {
  node_id: string
  parameter_count: number
  outputs: Record<string, { shape: Array<number | null> }>
}
type ValidationDiagnostic = {
  code?: string
  message?: string
  node_id?: string | null
  parameter_id?: string | null
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
type AttentionCase = {
  name: string
  catalogNode: string
  definitionId: string
  includeContext: boolean
  inputLabels: string[]
  parameters: Record<string, unknown>
  staticOutputShape: Array<number | null>
  runtimeOutputShape: number[]
  parameterCount: number
  kerasEnabled: boolean
  worker: { api: string; backend: string; device_id: string }
}

const evidenceDir = path.resolve(
  process.env.NW_VALIDATION_EVIDENCE_DIR
    ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260929', 'evidence', 'nw-t3-06-attention'),
)

function graphShell(nodes: GraphNode[]) {
  return {
    schema_version: 6,
    nodes,
    edges: [],
    model_configuration: {},
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

function initialGraph(includeContext: boolean) {
  const nodes = includeContext
    ? [
        node('query-input', 'input.sequence', 'Query Sequence', { x: 48, y: 120 }, { shape: [5, 8], dtype: 'float32' }),
        node('context-input', 'input.sequence', 'Context Sequence', { x: 48, y: 360 }, { shape: [3, 8], dtype: 'float32' }),
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
  includeContext: boolean,
  kerasEnabled: boolean,
): Promise<CreatedProject> {
  const response = await request.post('/api/projects', {
    data: {
      name,
      description: 'NW-T3-06 attention primitive validation fixture',
      execution_backend: 'pytorch',
      keras_enabled: kerasEnabled,
      graph: initialGraph(includeContext),
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
  targetPortLabel: string,
  expectedConnectionCount: number,
): Promise<void> {
  const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
  const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
  const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
  const target = targetNode.getByRole('button', { name: new RegExp(`^Port ${targetPortLabel} \\(input\\)`) })
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
  expect(payload.valid, `${sourceLabel} → ${targetLabel}.${targetPortLabel} should be accepted`).toBe(true)
  await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
}

async function parameterField(inspector: Locator, displayName: string): Promise<Locator> {
  const field = inspector.locator('label.inspector-field').filter({ hasText: displayName }).first()
  if (!(await field.isVisible())) {
    const group = inspector.locator('details.parameter-group').filter({ hasText: displayName }).first()
    await group.locator('summary').click()
  }
  await expect(field, `${displayName} should be visible in the Inspector`).toBeVisible()
  return field
}

async function setAttentionParameters(
  page: Page,
  nodeId: string,
  parameters: Record<string, unknown>,
): Promise<void> {
  await page.locator(`[data-node-id="${nodeId}"]`).click()
  const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
  for (const [id, value] of Object.entries(parameters)) {
    const displayName = {
      num_heads: 'Num Heads',
      key_dim: 'Key Dim',
      value_dim: 'Value dim',
      layers: 'Layers',
      dropout: 'Dropout',
      use_bias: 'Use bias',
      output_shape: 'Output shape',
    }[id]
    if (!displayName) continue
    const field = await parameterField(inspector, displayName)
    if (id === 'use_bias') {
      const checkbox = field.locator('input[type="checkbox"]')
      if (value === true) await checkbox.check()
      else await checkbox.uncheck()
    } else if (id === 'output_shape') {
      await field.locator('textarea').fill(JSON.stringify(value))
    } else {
      const input = field.locator('input')
      if (await input.isEnabled()) {
        await input.fill(String(value))
      }
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

async function saveProject(page: Page, fixture: CreatedProject): Promise<SavedProject> {
  const saveResponsePromise = page.waitForResponse(
    (response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT',
  )
  await page.getByRole('button', { name: 'Save model', exact: true }).click()
  const saveResponse = await saveResponsePromise
  expect(saveResponse.ok()).toBeTruthy()
  return saveResponse.json() as Promise<SavedProject>
}

async function validateSaveAndRun(
  page: Page,
  request: APIRequestContext,
  fixture: CreatedProject,
  attentionNodeId: string,
  candidate: AttentionCase,
  browserErrors: string[],
  failedRequests: string[],
): Promise<{ validation: ValidationPayload; saved: SavedProject; submittedJobId: string; finalJob: RuntimeJob }> {
  const validation = await validateGraph(page)
  expect(validation.valid, JSON.stringify(validation.diagnostics)).toBe(true)
  expect(validation.diagnostics).toEqual([])
  const summary = validation.shape_summaries.find((item) => item.node_id === attentionNodeId)
  expect(summary?.outputs.output?.shape).toEqual(candidate.staticOutputShape)
  expect(summary?.parameter_count).toBe(candidate.parameterCount)
  expect(validation.total_parameters).toBe(candidate.parameterCount)
  await expect(page.locator('.validation-summary')).toContainText('Ready')
  await mkdir(evidenceDir, { recursive: true })
  await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-builder-ready.png`), fullPage: true })

  const saved = await saveProject(page, fixture)
  expect(saved.validation_status).toBe('valid')
  expect(saved.graph.nodes).toHaveLength(candidate.includeContext ? 4 : 3)
  expect(saved.graph.edges).toHaveLength(candidate.includeContext ? 3 : 2)
  const savedAttention = saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)
  expect(savedAttention?.parameters).toMatchObject(candidate.parameters)

  await page.reload()
  await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('[data-node-id]')).toHaveCount(candidate.includeContext ? 4 : 3)
  await expect(page.getByRole('group', { name: `Node ${candidate.catalogNode}`, exact: true })).toBeVisible()

  await page.goto(`/models/${fixture.id}/validation`)
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
    const jobResponse = await request.get(`/api/execution/jobs/${submitted.id}`)
    if (!jobResponse.ok()) return 'missing'
    finalJob = await jobResponse.json() as RuntimeJob
    return finalJob.state
  }, { timeout: 90_000 }).toBe('succeeded')

  expect(finalJob.worker).toMatchObject(candidate.worker)
  expect(finalJob.result?.status).toBe('succeeded')
  expect(finalJob.result?.outputs[0]?.shape).toEqual(candidate.runtimeOutputShape)
  expect(finalJob.result?.parameter_count).toBe(candidate.parameterCount)
  await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-runtime-succeeded.png`), fullPage: true })
  expect(browserErrors, `${candidate.name} should not emit browser errors`).toEqual([])
  expect(failedRequests, `${candidate.name} should not emit failed product requests`).toEqual([])

  await writeFile(path.join(evidenceDir, `${candidate.name}.json`), JSON.stringify({
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
  await writeFile(path.join(evidenceDir, `${candidate.name}-browser-errors.json`), JSON.stringify({ browser_errors: browserErrors, failed_requests: failedRequests }, null, 2))
  return { validation, saved, submittedJobId: submitted.id, finalJob }
}

const attentionCases: AttentionCase[] = [
  {
    name: 'nw-t3-06-self-keras-pytorch',
    catalogNode: 'Self-attention',
    definitionId: 'attention.self',
    includeContext: false,
    inputLabels: ['Input Sequence'],
    parameters: { num_heads: 2, key_dim: 4, value_dim: 4, layers: 1, dropout: 0, use_bias: true },
    staticOutputShape: [null, 5, 8],
    runtimeOutputShape: [1, 5, 8],
    parameterCount: 288,
    kerasEnabled: true,
    worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
  },
  {
    name: 'nw-t3-06-cross-keras-pytorch',
    catalogNode: 'Cross-attention',
    definitionId: 'attention.cross',
    includeContext: true,
    inputLabels: ['Query Sequence', 'Context Sequence'],
    parameters: { num_heads: 2, key_dim: 4, value_dim: 4, layers: 1, dropout: 0, use_bias: true },
    staticOutputShape: [null, 5, 8],
    runtimeOutputShape: [1, 5, 8],
    parameterCount: 288,
    kerasEnabled: true,
    worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
  },
  {
    name: 'nw-t3-06-multi-head-output-shape-keras-pytorch',
    catalogNode: 'Multi-head Attention',
    definitionId: 'attention.multi_head',
    includeContext: false,
    inputLabels: ['Input Sequence'],
    parameters: { num_heads: 2, key_dim: 4, value_dim: 4, output_shape: [2, 4], layers: 1 },
    staticOutputShape: [null, 5, 2, 4],
    runtimeOutputShape: [1, 5, 2, 4],
    parameterCount: 288,
    kerasEnabled: true,
    worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
  },
  {
    name: 'nw-t3-06-self-native-pytorch',
    catalogNode: 'Self-attention',
    definitionId: 'attention.self',
    includeContext: false,
    inputLabels: ['Input Sequence'],
    parameters: { num_heads: 2, key_dim: 4, value_dim: 4, layers: 1, dropout: 0, use_bias: true },
    staticOutputShape: [null, 5, 8],
    runtimeOutputShape: [1, 5, 8],
    parameterCount: 288,
    kerasEnabled: false,
    worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' },
  },
]

test.describe('NW-T3-06 attention primitive catalog slice', () => {
  test.describe.configure({ mode: 'serial' })
  test.setTimeout(240_000)

  test('live catalog renders all attention primitives with Keras and native mappings', async ({ page, request }) => {
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
    const metadataResponse = await request.get('/api/metadata')
    expect(metadataResponse.ok()).toBeTruthy()
    const metadata = await metadataResponse.json() as Record<string, unknown>
    const attention = catalog.nodes.filter((item) => item.id.startsWith('attention.'))
    expect(attention.map((item) => item.id).sort()).toEqual(
      ['attention.cross', 'attention.multi_head', 'attention.self'].sort(),
    )
    for (const item of attention) {
      expect(item.implementations.keras?.supported).toBe(true)
      expect(item.implementations.pytorch?.supported).toBe(true)
      expect(item.implementations.tensorflow?.supported).toBe(true)
      expect(item.implementations.jax?.supported).toBe(true)
    }

    await page.goto('/builder')
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
    await page.getByRole('button', { name: /Transformer 10 nodes/ }).click()
    for (const candidate of attentionCases.slice(0, 3)) {
      await expect(page.getByRole('button', {
        name: new RegExp(`${candidate.catalogNode} ${candidate.definitionId} · operation`),
      })).toBeEnabled()
    }
    await mkdir(evidenceDir, { recursive: true })
    await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-06-attention-catalog-rendered.png'), fullPage: true })
    await writeFile(path.join(evidenceDir, 'nw-t3-06-live-catalog.json'), JSON.stringify({
      schema_version: catalog.schema_version,
      catalog_count: catalog.nodes.length,
      category_counts: catalog.nodes.reduce<Record<string, number>>((counts, item) => {
        counts[item.category] = (counts[item.category] ?? 0) + 1
        return counts
      }, {}),
      catalog_fingerprint_sha256: createHash('sha256').update(JSON.stringify(catalog)).digest('hex'),
      attention,
      metadata,
    }, null, 2))
  })

  for (const candidate of attentionCases) {
    test(`${candidate.name} builds, persists, and completes exact-profile functional validation`, async ({ page, request }) => {
      const fixture = await createProject(
        request,
        `Tier 3 attention ${candidate.catalogNode} ${Date.now()}`,
        candidate.includeContext,
        candidate.kerasEnabled,
      )
      const browserErrors: string[] = []
      const failedRequests: string[] = []
      page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
      page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
      page.on('requestfailed', (requestEvent) => failedRequests.push(`${requestEvent.method()} ${requestEvent.url()} ${requestEvent.failure()?.errorText ?? 'unknown'}`))
      page.on('response', (response) => {
        if (response.status() >= 400 && response.url().includes('/api/')) failedRequests.push(`${response.status()} ${response.request().method()} ${response.url()}`)
      })
      try {
        await page.goto(`/builder/${fixture.id}`)
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        const attentionNodeId = await addCatalogNode(page, candidate.catalogNode)
        await setAttentionParameters(page, attentionNodeId, candidate.parameters)
        if (candidate.includeContext) {
          await connectByPorts(page, 'Query Sequence', candidate.catalogNode, 'Query', 1)
          await connectByPorts(page, 'Context Sequence', candidate.catalogNode, 'Context', 2)
        } else {
          await connectByPorts(page, 'Input Sequence', candidate.catalogNode, 'Query', 1)
        }
        await connectByPorts(page, candidate.catalogNode, 'Custom Output', 'Input', candidate.includeContext ? 3 : 2)
        await validateSaveAndRun(page, request, fixture, attentionNodeId, candidate, browserErrors, failedRequests)
      } finally {
        await deleteProject(request, fixture.id)
      }
    })
  }

  test('invalid layers are blocked, repaired, persisted, and executed', async ({ page, request }) => {
    const candidate = attentionCases[0]!
    const fixture = await createProject(request, `Tier 3 attention repair ${Date.now()}`, false, true)
    const browserErrors: string[] = []
    const failedRequests: string[] = []
    let jobSubmitCount = 0
    page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
    page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
    page.on('requestfailed', (requestEvent) => failedRequests.push(`${requestEvent.method()} ${requestEvent.url()} ${requestEvent.failure()?.errorText ?? 'unknown'}`))
    page.on('request', (requestEvent) => {
      if (requestEvent.method() === 'POST' && requestEvent.url().includes('/api/execution/jobs')) jobSubmitCount += 1
    })
    try {
      await page.goto(`/builder/${fixture.id}`)
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
      const attentionNodeId = await addCatalogNode(page, candidate.catalogNode)
      await setAttentionParameters(page, attentionNodeId, candidate.parameters)
      await connectByPorts(page, 'Input Sequence', candidate.catalogNode, 'Query', 1)
      await connectByPorts(page, candidate.catalogNode, 'Custom Output', 'Input', 2)

      const validBeforeRepair = await validateGraph(page)
      expect(validBeforeRepair.valid).toBe(true)
      const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
      const layers = (await parameterField(inspector, 'Layers')).locator('input')
      await layers.fill('2')
      await expect(layers).toHaveValue('2')
      const invalid = await validateGraph(page)
      expect(invalid.valid).toBe(false)
      expect(invalid.diagnostics.some((item) => item.message?.includes('accepts layers=1'))).toBe(true)
      await expect(page.locator('.validation-summary')).toContainText('Blocked')
      await mkdir(evidenceDir, { recursive: true })
      await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-06-invalid-layers.png'), fullPage: true })
      await writeFile(path.join(evidenceDir, 'nw-t3-06-invalid-layers.json'), JSON.stringify({
        case: 'nw-t3-06-invalid-layers',
        project_id: fixture.id,
        authored_parameters: { ...candidate.parameters, layers: 2 },
        validation: invalid,
        execution_job_submissions: jobSubmitCount,
        browser_errors: browserErrors,
        failed_requests: failedRequests,
      }, null, 2))

      await layers.fill('1')
      await expect(layers).toHaveValue('1')
      const repaired = await validateSaveAndRun(page, request, fixture, attentionNodeId, candidate, browserErrors, failedRequests)
      expect(repaired.saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)?.parameters.layers).toBe(1)
      await writeFile(path.join(evidenceDir, 'nw-t3-06-repaired-layers.json'), JSON.stringify({
        case: 'nw-t3-06-repaired-layers',
        project_id: fixture.id,
        invalid_diagnostic: invalid.diagnostics.find((item) => item.parameter_id === 'layers'),
        repaired_saved_revision: repaired.saved.current_revision,
        repaired_graph: {
          node_count: repaired.saved.graph.nodes.length,
          edge_count: repaired.saved.graph.edges.length,
          layers: repaired.saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)?.parameters.layers,
        },
        submitted_job_id: repaired.submittedJobId,
        final_job: repaired.finalJob,
        execution_job_submissions: jobSubmitCount,
        browser_errors: browserErrors,
        failed_requests: failedRequests,
      }, null, 2))
      await page.goto(`/builder/${fixture.id}`)
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
      const persistedNode = page.getByRole('group', { name: `Node ${candidate.catalogNode}`, exact: true })
      await persistedNode.click()
      const persistedInspector = page.getByRole('complementary', { name: 'Properties inspector' })
      await expect((await parameterField(persistedInspector, 'Layers')).locator('input')).toHaveValue('1')
      expect(jobSubmitCount).toBe(1)
      expect(browserErrors).toEqual([])
      expect(failedRequests).toEqual([])
    } finally {
      await deleteProject(request, fixture.id)
    }
  })

  test('native attention visibly fixes unsupported dropout without submitting a job', async ({ page, request }) => {
    const candidate = attentionCases[3]!
    const fixture = await createProject(request, `Tier 3 attention native restriction ${Date.now()}`, false, false)
    const browserErrors: string[] = []
    const failedRequests: string[] = []
    let jobSubmitCount = 0
    page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
    page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
    page.on('requestfailed', (requestEvent) => failedRequests.push(`${requestEvent.method()} ${requestEvent.url()} ${requestEvent.failure()?.errorText ?? 'unknown'}`))
    page.on('request', (requestEvent) => {
      if (requestEvent.method() === 'POST' && requestEvent.url().includes('/api/execution/jobs')) jobSubmitCount += 1
    })
    try {
      await page.goto(`/builder/${fixture.id}`)
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
      const attentionNodeId = await addCatalogNode(page, candidate.catalogNode)
      await setAttentionParameters(page, attentionNodeId, candidate.parameters)
      const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
      const dropout = await parameterField(inspector, 'Dropout')
      await expect(dropout).toContainText('Fixed to 0 by this backend.')
      await expect(dropout.locator('input')).toBeDisabled()
      await expect(dropout.locator('input')).toHaveValue('0')
      await connectByPorts(page, 'Input Sequence', candidate.catalogNode, 'Query', 1)
      await connectByPorts(page, candidate.catalogNode, 'Custom Output', 'Input', 2)
      const validation = await validateGraph(page)
      expect(validation.valid).toBe(true)
      expect(validation.diagnostics).toEqual([])
      await mkdir(evidenceDir, { recursive: true })
      await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-06-native-dropout-fixed.png'), fullPage: true })
      await writeFile(path.join(evidenceDir, 'nw-t3-06-native-dropout-fixed.json'), JSON.stringify({
        case: 'nw-t3-06-native-dropout-fixed',
        project_id: fixture.id,
        profile: candidate.worker,
        authored_parameters: candidate.parameters,
        capability_boundary: {
          parameter_id: 'dropout',
          rendered_value: 0,
          editable: false,
          message: 'Fixed to 0 by this backend.',
          authored_nonzero_value_is_rejected_by_static_contract: true,
        },
        validation,
        execution_job_submissions: jobSubmitCount,
        browser_errors: browserErrors,
        failed_requests: failedRequests,
      }, null, 2))
      expect(jobSubmitCount).toBe(0)
      expect(browserErrors).toEqual([])
      expect(failedRequests).toEqual([])
    } finally {
      await deleteProject(request, fixture.id)
    }
  })
})
