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
type GraphEdge = {
  id: string
  source_node_id: string
  source_port_id: string
  target_node_id: string
  target_port_id: string
}

const evidenceDir = path.resolve(
  process.env.NW_VALIDATION_EVIDENCE_DIR
    ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260927', 'evidence'),
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

async function createProject(
  request: APIRequestContext,
  name: string,
  modelGraph: ReturnType<typeof graphShell>,
): Promise<CreatedProject> {
  const response = await request.post('/api/projects', {
    data: {
      name,
      description: 'Risk-based Tier 3 catalog validation fixture',
      execution_backend: 'pytorch',
      keras_enabled: true,
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
  sourcePort: string,
  targetLabel: string,
  targetPort: string,
  expectedConnectionCount: number,
): Promise<Record<string, unknown>> {
  const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
  const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
  const source = sourceNode.getByRole('button', { name: new RegExp(`^Port ${sourcePort} \\(output\\)`) })
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
  const payload = await response.json() as Record<string, unknown>
  expect(payload['valid'], `${sourceLabel} → ${targetLabel} should be accepted`).toBe(true)
  await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
  return payload
}

async function saveValidateAndRun(
  page: Page,
  request: APIRequestContext,
  fixture: CreatedProject,
  expectedNodes: number,
  expectedEdges: number,
  caseName: string,
  testInfo: TestInfo,
): Promise<Record<string, unknown>> {
  const validationResponsePromise = page.waitForResponse(
    (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Validate', exact: true }).click()
  const validationResponse = await validationResponsePromise
  expect(validationResponse.ok()).toBeTruthy()
  const validationPayload = await validationResponse.json() as Record<string, unknown>
  expect(validationPayload['valid']).toBe(true)
  await expect(page.locator('.validation-summary')).toContainText('Ready')
  await page.screenshot({ path: testInfo.outputPath(`${caseName}-builder-ready.png`), fullPage: true })

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
  expect(saved.graph.nodes).toHaveLength(expectedNodes)
  expect(saved.graph.edges).toHaveLength(expectedEdges)

  await page.reload()
  await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
  await expect(page.locator('[data-node-id]')).toHaveCount(expectedNodes)

  await page.goto(`/models/${fixture.id}/validation`)
  await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText(/keras.*pytorch/i).first()).toBeVisible({ timeout: 75_000 })
  const submitResponsePromise = page.waitForResponse(
    (response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
  const submitResponse = await submitResponsePromise
  expect(submitResponse.ok()).toBeTruthy()
  const submitted = await submitResponse.json() as { id: string }
  await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 90_000 })
  let finalJob: Record<string, unknown> = {}
  await expect.poll(async () => {
    const response = await request.get(`/api/execution/jobs/${submitted.id}`)
    if (!response.ok()) return 'missing'
    finalJob = await response.json() as Record<string, unknown>
    return String(finalJob['state'])
  }, { timeout: 90_000 }).toBe('succeeded')

  await page.screenshot({ path: testInfo.outputPath(`${caseName}-runtime-succeeded.png`), fullPage: true })
  await mkdir(evidenceDir, { recursive: true })
  await writeFile(path.join(evidenceDir, `${caseName}.json`), JSON.stringify({
    case: caseName,
    project_id: fixture.id,
    revision: saved.current_revision,
    validation: validationPayload,
    submitted_job_id: submitted.id,
    final_job: finalJob,
  }, null, 2))
  return { validation: validationPayload, saved, final_job: finalJob }
}

test.describe('NW-T3-01 risk-based catalog slice', () => {
  test.describe.configure({ mode: 'serial' })
  test.setTimeout(180_000)

  test('recurrent sequence/state ports build, persist, and execute from the catalog', async ({ page, request }, testInfo) => {
    const fixture = await createProject(
      request,
      `Tier 3 recurrent catalog ${Date.now()}`,
      graphShell([
        node('sequence-input', 'input.sequence', 'Sequence input', { x: 48, y: 160 }, { shape: [4, 3], dtype: 'float32' }),
        node('sequence-output', 'output.sequence', 'Sequence output', { x: 1040, y: 120 }, {}),
        node('state-output', 'output.custom', 'State output', { x: 1040, y: 360 }, {}),
      ]),
    )
    const browserErrors: string[] = []
    page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
    page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
    try {
      await page.goto(`/builder/${fixture.id}`)
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
      const rnnNodeId = await addCatalogNode(page, 'Simple RNN')
      const rnn = page.locator(`[data-node-id="${rnnNodeId}"]`)
      await rnn.click()
      const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
      await inspector.locator('label.inspector-field').filter({ hasText: 'Units' }).locator('input').fill('8')
      await inspector.locator('label.inspector-field').filter({ hasText: 'Return sequences' }).locator('input[type="checkbox"]').check()
      await inspector.locator('label.inspector-field').filter({ hasText: 'Return state' }).locator('input[type="checkbox"]').check()
      await expect(rnn.getByRole('button', { name: /^Port Hidden state \(output\)/ })).toBeVisible()

      await connectByPorts(page, 'Sequence input', 'Output', 'Simple RNN', 'Input', 1)
      await connectByPorts(page, 'Simple RNN', 'Output', 'Sequence output', 'Input', 2)
      await connectByPorts(page, 'Simple RNN', 'Hidden state', 'State output', 'Input', 3)
      await saveValidateAndRun(page, request, fixture, 4, 3, 'nw-t3-01-recurrent', testInfo)
      expect(browserErrors, 'recurrent catalog workflow should not emit browser errors').toEqual([])
      await writeFile(path.join(evidenceDir, 'nw-t3-01-recurrent-browser-errors.json'), JSON.stringify(browserErrors, null, 2))
    } finally {
      await deleteProject(request, fixture.id)
    }
  })

  test('tensor-list merge fan-in and axis parameter build, persist, and execute from the catalog', async ({ page, request }, testInfo) => {
    const fixture = await createProject(
      request,
      `Tier 3 merge catalog ${Date.now()}`,
      graphShell([
        node('left-input', 'input.tensor', 'Left input', { x: 48, y: 96 }, { shape: [4], dtype: 'float32' }),
        node('right-input', 'input.tensor', 'Right input', { x: 48, y: 320 }, { shape: [4], dtype: 'float32' }),
        node('merge-output', 'output.custom', 'Merged output', { x: 1040, y: 200 }, {}),
      ]),
    )
    const browserErrors: string[] = []
    page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
    page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
    try {
      await page.goto(`/builder/${fixture.id}`)
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
      const mergeNodeId = await addCatalogNode(page, 'Concatenate')
      const merge = page.locator(`[data-node-id="${mergeNodeId}"]`)
      await merge.click()
      const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
      await inspector.locator('label.inspector-field').filter({ hasText: 'Axis' }).locator('input').fill('-1')

      await connectByPorts(page, 'Left input', 'Output', 'Concatenate', 'Inputs', 1)
      await connectByPorts(page, 'Right input', 'Output', 'Concatenate', 'Inputs', 2)
      await connectByPorts(page, 'Concatenate', 'Output', 'Merged output', 'Input', 3)
      const result = await saveValidateAndRun(page, request, fixture, 4, 3, 'nw-t3-01-merge', testInfo)
      const validationPayload = result['validation'] as {
        shape_summaries?: Array<{ node_id: string; outputs: Record<string, { shape: unknown[] }> }>
      }
      const mergeSummary = validationPayload.shape_summaries?.find((summary) => summary.node_id === mergeNodeId)
      expect(mergeSummary?.outputs['output']?.shape).toEqual([null, 8])
      expect(browserErrors, 'merge catalog workflow should not emit browser errors').toEqual([])
      await writeFile(path.join(evidenceDir, 'nw-t3-01-merge-browser-errors.json'), JSON.stringify(browserErrors, null, 2))
    } finally {
      await deleteProject(request, fixture.id)
    }
  })
})
