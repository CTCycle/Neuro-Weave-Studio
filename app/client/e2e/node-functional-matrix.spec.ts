import { expect, test, type APIRequestContext, type Page, type TestInfo } from '@playwright/test'

type CreatedProject = { id: string; current_revision: number; name: string }
type GraphNode = {
  id: string
  definition_id: string
  definition_version: number
  label: string | null
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

type CaseDefinition = {
  name: string
  inputDefinition: 'input.sequence' | 'input.image' | 'input.tensor'
  inputShape: number[]
  catalogNodes: string[]
  kerasEnabled?: boolean
}

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

function inputOutputGraph(inputDefinition: CaseDefinition['inputDefinition'], inputShape: number[]) {
  return graphShell([
    {
      id: 'matrix-input',
      definition_id: inputDefinition,
      definition_version: 1,
      label: 'Matrix input',
      position: { x: 64, y: 180 },
      parameters: { shape: inputShape, dtype: 'float32' },
    },
    {
      id: 'matrix-output',
      definition_id: 'output.custom',
      definition_version: 1,
      label: 'Matrix output',
      position: { x: 1200, y: 180 },
      parameters: {},
    },
  ])
}

async function createProject(
  request: APIRequestContext,
  name: string,
  modelGraph: ReturnType<typeof graphShell>,
  kerasEnabled = true,
): Promise<CreatedProject> {
  const response = await request.post('/api/projects', {
    data: {
      name,
      description: 'Focused CNN and activation browser validation fixture',
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
  const before = new Set(await nodes.evaluateAll((elements) => elements.map((element) => element.getAttribute('data-node-id')).filter(Boolean) as string[]))
  await page.getByRole('textbox', { name: 'Search nodes' }).fill(displayName)
  const item = page.locator('.catalog-item').filter({ hasText: displayName }).first()
  await expect(item, `Catalog node ${displayName} should be available for the selected profile`).toBeEnabled()
  await item.click()
  await expect(nodes).toHaveCount(before.size + 1)
  const after = await nodes.evaluateAll((elements) => elements.map((element) => element.getAttribute('data-node-id')).filter(Boolean) as string[])
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
  const validationResponsePromise = page.waitForResponse(
    (response) => response.url().includes('/api/validation/connection') && response.request().method() === 'POST',
  )
  await target.press('Enter')
  const response = await validationResponsePromise
  expect(response.ok()).toBeTruthy()
  const payload = await response.json() as { valid: boolean; diagnostics: unknown[] }
  console.log(`[NODE_MATRIX_CONNECTION] ${sourceLabel} -> ${targetLabel} ${JSON.stringify(payload)}`)
  expect(payload.valid, `${sourceLabel} → ${targetLabel} should be accepted by connection validation`).toBeTruthy()
  await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
}

async function capture(page: Page, testInfo: TestInfo, name: string): Promise<void> {
  await page.screenshot({ path: testInfo.outputPath(`${name}.png`), fullPage: true })
}

async function validateSaveAndRun(
  page: Page,
  request: APIRequestContext,
  fixture: CreatedProject,
  expectedNodes: number,
  expectedEdges: number,
  caseName: string,
  kerasEnabled: boolean,
  testInfo: TestInfo,
): Promise<void> {
  const validationResponsePromise = page.waitForResponse(
    (response) => response.url().includes('/api/validation') && response.request().method() === 'POST',
  )
  await page.getByRole('button', { name: 'Validate', exact: true }).click()
  const validationResponse = await validationResponsePromise
  expect(validationResponse.ok()).toBeTruthy()
  const validationPayload = await validationResponse.json()
  console.log(`[NODE_MATRIX_STATIC] ${caseName} ${JSON.stringify(validationPayload)}`)
  await expect(page.locator('.validation-summary')).toContainText('Ready')
  await capture(page, testInfo, `${caseName}-builder-ready`)

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

  await page.goto(`/models/${fixture.id}/validation`)
  await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
  const profilePattern = kerasEnabled ? /keras.*pytorch/i : /native.*pytorch/i
  await expect(page.getByText(profilePattern).first()).toBeVisible({ timeout: 75_000 })

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

  console.log(`[NODE_MATRIX_RUNTIME] ${caseName} ${JSON.stringify(finalJob)}`)
  await capture(page, testInfo, `${caseName}-runtime-succeeded`)
}

async function runCnnCase(
  page: Page,
  request: APIRequestContext,
  testInfo: TestInfo,
  definition: CaseDefinition,
): Promise<void> {
  const kerasEnabled = definition.kerasEnabled ?? true
  const fixture = await createProject(
    request,
    `Node matrix ${definition.name} ${Date.now()}`,
    inputOutputGraph(definition.inputDefinition, definition.inputShape),
    kerasEnabled,
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

    for (const displayName of definition.catalogNodes) {
      await addCatalogNode(page, displayName)
    }

    const chainLabels = ['Matrix input', ...definition.catalogNodes, 'Matrix output']
    for (let index = 0; index < chainLabels.length - 1; index += 1) {
      await connectByLabels(page, chainLabels[index]!, chainLabels[index + 1]!, index + 1)
    }

    await validateSaveAndRun(
      page,
      request,
      fixture,
      chainLabels.length,
      chainLabels.length - 1,
      definition.name,
      kerasEnabled,
      testInfo,
    )
    expect(browserErrors, `${definition.name} should not emit browser errors`).toEqual([])
    console.log(`[NODE_MATRIX_BROWSER_ERRORS] ${definition.name} ${JSON.stringify(browserErrors)}`)
  } finally {
    await deleteProject(request, fixture.id)
  }
}

async function runActivationCase(
  page: Page,
  request: APIRequestContext,
  testInfo: TestInfo,
  activation: string,
): Promise<void> {
  const caseName = `activation-${activation}`
  const fixture = await createProject(
    request,
    `Node matrix ${caseName} ${Date.now()}`,
    inputOutputGraph('input.tensor', [8]),
    true,
  )
  const browserErrors: string[] = []
  page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
  page.on('console', (message) => {
    if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`)
  })

  try {
    await page.goto(`/builder/${fixture.id}`)
    const activationNodeId = await addCatalogNode(page, 'Activation')
    await page.locator(`[data-node-id="${activationNodeId}"]`).click()
    const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
    const activationField = inspector.locator('label.inspector-field').filter({ hasText: 'Activation' })
    await expect(activationField.locator('select')).toBeVisible()
    await activationField.locator('select').selectOption(activation)
    await connectByLabels(page, 'Matrix input', 'Activation', 1)
    await connectByLabels(page, 'Activation', 'Matrix output', 2)
    await validateSaveAndRun(page, request, fixture, 3, 2, caseName, true, testInfo)
    expect(browserErrors, `${caseName} should not emit browser errors`).toEqual([])
    console.log(`[NODE_MATRIX_BROWSER_ERRORS] ${caseName} ${JSON.stringify(browserErrors)}`)
  } finally {
    await deleteProject(request, fixture.id)
  }
}

test.describe('focused CNN and activation node functional matrix', () => {
  test.setTimeout(150_000)

  const cnnCases: CaseDefinition[] = [
    {
      name: 'cnn-1d-keras-pytorch',
      inputDefinition: 'input.sequence',
      inputShape: [12, 3],
      catalogNodes: ['Conv1D', 'Average Pooling 1D', 'Global Pooling 1D'],
    },
    {
      name: 'cnn-2d-keras-pytorch',
      inputDefinition: 'input.image',
      inputShape: [16, 16, 3],
      catalogNodes: ['Conv2D', 'Max Pooling 2D', 'Global Pooling 2D'],
    },
    {
      name: 'cnn-3d-keras-pytorch',
      inputDefinition: 'input.tensor',
      inputShape: [8, 8, 8, 1],
      catalogNodes: ['Conv3D', 'Average Pooling 3D', 'Global Pooling 3D'],
    },
    {
      name: 'cnn-2d-native-pytorch',
      inputDefinition: 'input.image',
      inputShape: [16, 16, 3],
      catalogNodes: ['Conv2D', 'Max Pooling 2D', 'Global Pooling 2D'],
      kerasEnabled: false,
    },
  ]

  for (const definition of cnnCases) {
    test(`${definition.name} builds in the Builder and completes a real functional validation`, async ({ page, request }, testInfo) => {
      await runCnnCase(page, request, testInfo, definition)
    })
  }

  for (const activation of ['relu', 'gelu', 'sigmoid', 'tanh', 'softmax', 'linear']) {
    test(`Activation ${activation} builds in the Builder and completes a real functional validation`, async ({ page, request }, testInfo) => {
      await runActivationCase(page, request, testInfo, activation)
    })
  }
})
