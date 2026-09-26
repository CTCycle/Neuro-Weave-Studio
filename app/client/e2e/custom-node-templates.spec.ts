import fs from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

const evidenceDir = path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260926', 'evidence', 'nw-t2-09')

type CustomNodeDefinition = {
  id: string
  display_name: string
  version: number
  spec: {
    type: string
    formula?: string
    expression?: string
    components?: Array<{ definition_id: string; parameters: Record<string, unknown> }>
    edges?: unknown[]
  }
}

async function customDefinitions(request: APIRequestContext): Promise<CustomNodeDefinition[]> {
  const response = await request.get('/api/nodes/custom')
  expect(response.ok()).toBeTruthy()
  const payload = await response.json() as { definitions: CustomNodeDefinition[] }
  return payload.definitions
}

async function ensureTensorFlowKeras(page: Page): Promise<void> {
  await page.goto('/environment')
  const tensorflow = page.locator('.backend-status-card-container').filter({ hasText: 'TensorFlow' }).first()
  if (!(await tensorflow.getByText('Usable', { exact: true }).count())) {
    await tensorflow.getByRole('button', { name: 'Check Keras', exact: true }).click()
  }
  await expect(tensorflow).toContainText('Usable', { timeout: 90_000 })
}

test('authors, reloads, and materializes mathematical and graph templates', async ({ page, request }) => {
  test.setTimeout(120_000)
  await fs.mkdir(evidenceDir, { recursive: true })

  const suffix = Date.now().toString()
  const mathName = `Browser Safe Quadratic ${suffix}`
  const graphName = `Browser Dense GELU ${suffix}`

  await page.goto('/nodes')
  await expect(page.getByRole('heading', { name: 'Nodes', exact: true })).toBeVisible()

  await page.getByRole('button', { name: 'Add custom node', exact: true }).click()
  let dialog = page.getByRole('dialog', { name: 'Choose a node type' })
  await dialog.getByRole('button', { name: /Mathematical operation/ }).click()
  await dialog.getByRole('button', { name: 'Continue', exact: true }).click()
  dialog = page.getByRole('dialog', { name: 'Describe the custom node' })
  await dialog.getByLabel('Name').fill(mathName)
  await dialog.getByLabel('Category').fill('validation')
  await dialog.getByLabel('Description').fill('A safe single-input quadratic expression for browser validation.')
  await dialog.getByRole('button', { name: 'Continue', exact: true }).click()
  dialog = page.getByRole('dialog', { name: 'Define the operation' })
  await dialog.locator('textarea[formcontrolname="formula"]').fill('x^2 + 1')
  const mathCreate = page.waitForResponse((response) => response.url().includes('/api/nodes/custom') && response.request().method() === 'POST')
  await dialog.getByRole('button', { name: 'Add custom node', exact: true }).click()
  expect((await mathCreate).status()).toBe(201)
  await expect(page.getByRole('heading', { name: mathName, exact: true })).toBeVisible()

  await page.getByRole('button', { name: `Edit ${mathName}`, exact: true }).click()
  dialog = page.getByRole('dialog', { name: 'Edit node identity' })
  await dialog.getByRole('button', { name: 'Continue', exact: true }).click()
  dialog = page.getByRole('dialog', { name: 'Define the operation' })
  await dialog.locator('textarea[formcontrolname="formula"]').fill('x^2 - 1')
  const mathUpdate = page.waitForResponse((response) => response.url().includes('/api/nodes/custom/') && response.request().method() === 'PUT')
  await dialog.getByRole('button', { name: 'Save changes', exact: true }).click()
  expect((await mathUpdate).status()).toBe(200)
  await expect(page.locator('.detail-body')).toContainText('Version')
  await expect(page.locator('.detail-body')).toContainText('x**2 - 1')

  await page.reload()
  const mathRow = page.locator('.library-row').filter({ hasText: mathName }).first()
  await expect(mathRow).toBeVisible()
  await mathRow.click()
  await expect(page.getByRole('heading', { name: mathName, exact: true })).toBeVisible()
  await expect(page.locator('.detail-body')).toContainText('x**2 - 1')
  await page.screenshot({ path: path.join(evidenceDir, 'math-template-persisted.png'), fullPage: false })

  await page.getByRole('button', { name: 'Add custom node', exact: true }).click()
  dialog = page.getByRole('dialog', { name: 'Choose a node type' })
  await dialog.getByRole('button', { name: /Graph template/ }).click()
  await dialog.getByRole('button', { name: 'Continue', exact: true }).click()
  dialog = page.getByRole('dialog', { name: 'Describe the custom node' })
  await dialog.getByLabel('Name').fill(graphName)
  await dialog.getByLabel('Category').fill('validation')
  await dialog.getByLabel('Description').fill('A reusable dense projection followed by GELU for browser validation.')
  await dialog.getByRole('button', { name: 'Continue', exact: true }).click()
  dialog = page.getByRole('dialog', { name: 'Build the graph template' })

  const builtInNode = dialog.getByLabel('Built-in node')
  await builtInNode.selectOption({ label: 'Dense / Linear · core' })
  await dialog.getByRole('button', { name: 'Add component', exact: true }).click()
  await dialog.getByRole('spinbutton', { name: /Units/ }).fill('64')
  await dialog.getByRole('combobox', { name: /Activation/ }).first().selectOption({ label: 'gelu' })
  await builtInNode.selectOption({ label: 'Activation · core' })
  await dialog.getByRole('button', { name: 'Add component', exact: true }).click()
  await dialog.getByRole('combobox', { name: /Activation/ }).nth(1).selectOption({ label: 'gelu' })

  const graphCreate = page.waitForResponse((response) => response.url().includes('/api/nodes/custom') && response.request().method() === 'POST')
  await dialog.getByRole('button', { name: 'Add custom node', exact: true }).click()
  expect((await graphCreate).status()).toBe(201)
  await expect(page.getByRole('heading', { name: graphName, exact: true })).toBeVisible()
  await expect(page.locator('.detail-body')).toContainText('2 components · 1 edges')
  await expect(page.locator('.detail-body')).toContainText('Dense / Linear')
  await expect(page.locator('.detail-body')).toContainText('Activation')

  await page.reload()
  const graphRow = page.locator('.library-row').filter({ hasText: graphName }).first()
  await expect(graphRow).toBeVisible()
  await graphRow.click()
  await expect(page.locator('.detail-body')).toContainText('2 components · 1 edges')
  await page.screenshot({ path: path.join(evidenceDir, 'graph-template-persisted.png'), fullPage: false })

  const definitions = await customDefinitions(request)
  const persistedMath = definitions.find((definition) => definition.display_name === mathName)
  const persistedGraph = definitions.find((definition) => definition.display_name === graphName)
  expect(persistedMath).toMatchObject({ version: 2, spec: { type: 'mathematical', formula: 'x^2 - 1', expression: 'x**2 - 1' } })
  expect(persistedGraph).toMatchObject({ version: 1, spec: { type: 'graph_template', edges: [{ source_component_id: 'component_0', target_component_id: 'component_1' }] } })
  expect(persistedGraph?.spec.components).toEqual(expect.arrayContaining([
    expect.objectContaining({ definition_id: 'layer.dense', parameters: expect.objectContaining({ units: 64, activation: 'gelu' }) }),
    expect.objectContaining({ definition_id: 'layer.activation', parameters: expect.objectContaining({ activation: 'gelu' }) }),
  ]))
  await fs.writeFile(path.join(evidenceDir, 'custom-node-template-results.json'), JSON.stringify({ math: persistedMath, graph: persistedGraph }, null, 2), 'utf8')

  await ensureTensorFlowKeras(page)
  await page.goto('/builder')
  await expect(page.getByRole('textbox', { name: 'Model name' })).toBeVisible()
  const catalogToggle = page.getByRole('button', { name: 'Catalog', exact: true })
  if (await catalogToggle.count() && await catalogToggle.isVisible()) await catalogToggle.click()
  await page.getByRole('button', { name: new RegExp(`${graphName} Graph template`) }).click()
  await expect(page.getByRole('group', { name: 'Node Dense / Linear', exact: true })).toBeVisible()
  await expect(page.getByRole('group', { name: 'Node Activation', exact: true })).toBeVisible()
  const connection = page.getByRole('group', { name: /Connection from .*layer-dense.* to .*layer-activation/ })
  await expect(connection).toHaveCount(1)
  await expect(connection).toHaveAttribute('aria-label', /Connection from .*layer-dense.* to .*layer-activation/)
  await expect(page.locator('.nw-flow-node')).toHaveCount(2)
  await page.screenshot({ path: path.join(evidenceDir, 'graph-template-materialized.png'), fullPage: false })
})
