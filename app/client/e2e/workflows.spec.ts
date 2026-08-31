import { expect, test, type APIRequestContext } from '@playwright/test'

type CreatedProject = { id: string; current_revision: number; name: string }

function graph() {
  return {
    schema_version: 3,
    nodes: [
      {
        id: 'input-tabular-e2e',
        definition_id: 'input.tabular',
        definition_version: 1,
        label: 'Features',
        position: { x: 48, y: 144 },
        parameters: { shape: [null, 8], dtype: 'float32' },
      },
      {
        id: 'layer-dense-e2e-1',
        definition_id: 'layer.dense',
        definition_version: 1,
        label: 'Hidden layer',
        position: { x: 352, y: 144 },
        parameters: {
          units: 16,
          activation: 'relu',
          use_bias: true,
          kernel_initializer: 'glorot_uniform',
          bias_initializer: 'zeros',
          kernel_regularizer: 'none',
          bias_regularizer: 'none',
          activity_regularizer: 'none',
          kernel_constraint: 'none',
          bias_constraint: 'none',
        },
      },
      {
        id: 'layer-dense-e2e-2',
        definition_id: 'layer.dense',
        definition_version: 1,
        label: 'Classifier',
        position: { x: 656, y: 144 },
        parameters: {
          units: 1,
          activation: 'sigmoid',
          use_bias: true,
          kernel_initializer: 'glorot_uniform',
          bias_initializer: 'zeros',
          kernel_regularizer: 'none',
          bias_regularizer: 'none',
          activity_regularizer: 'none',
          kernel_constraint: 'none',
          bias_constraint: 'none',
        },
      },
      {
        id: 'output-classification-e2e',
        definition_id: 'output.classification',
        definition_version: 1,
        label: 'Prediction',
        position: { x: 960, y: 144 },
        parameters: {},
      },
    ],
    edges: [
      { id: 'edge-e2e-1', source_node_id: 'input-tabular-e2e', source_port_id: 'output', target_node_id: 'layer-dense-e2e-1', target_port_id: 'input' },
      { id: 'edge-e2e-2', source_node_id: 'layer-dense-e2e-1', source_port_id: 'output', target_node_id: 'layer-dense-e2e-2', target_port_id: 'input' },
      { id: 'edge-e2e-3', source_node_id: 'layer-dense-e2e-2', source_port_id: 'output', target_node_id: 'output-classification-e2e', target_port_id: 'input' },
    ],
    model_configuration: {
      keras: {
        optimizer: 'adam',
        optimizer_parameters: { learning_rate: 0.001 },
        losses: { __default__: 'binary_crossentropy' },
        metrics: { __default__: ['binary_accuracy', 'auc'] },
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
          optimizer_parameters: { learning_rate: 0.0005, weight_decay: 0.004, beta_1: 0.9, beta_2: 0.999, epsilon: 1e-7 },
          losses: { __default__: 'binary_crossentropy' },
          metrics: { __default__: ['binary_accuracy', 'auc'] },
          compile_enabled: false,
          mode: 'default',
          fullgraph: false,
          dynamic: false,
        },
        jax: { jit_enabled: false },
      },
    },
    export_settings: { include_comments: true, include_shape_assertions: true, preferred_binary_format: 'keras' },
    future_training_configuration: {},
  }
}

async function createProject(request: APIRequestContext, name: string): Promise<CreatedProject> {
  const response = await request.post('/api/projects', {
    data: { name, description: 'Browser regression fixture', execution_backend: 'pytorch', keras_enabled: true, graph: graph() },
  })
  expect(response.ok()).toBeTruthy()
  return response.json() as Promise<CreatedProject>
}

async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
  const response = await request.delete(`/api/projects/${id}`)
  expect([204, 404]).toContain(response.status())
}

test('covers builder editing, history, focus mode, and dirty navigation', async ({ page, request }) => {
  const fixture = await createProject(request, `Angular builder ${Date.now()}`)
  try {
    await page.goto('/builder')
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()

    await page.getByRole('textbox', { name: 'Search nodes' }).fill('Dense')
    await page.getByRole('button', { name: /Dense \/ Linear/ }).click()
    await expect(page.locator('.nw-flow-node')).toHaveCount(1)
    await expect(page.getByRole('heading', { name: 'Properties' })).toBeVisible()

    const label = page.getByLabel('Label')
    await label.fill('Regression layer')
    await expect(label).toHaveValue('Regression layer')
    await page.locator('.nw-flow-node').click()
    await page.keyboard.press('Control+c')
    await page.keyboard.press('Control+v')
    await expect(page.locator('.nw-flow-node')).toHaveCount(2)
    await page.getByRole('button', { name: 'Undo' }).click()
    await expect(page.locator('.nw-flow-node')).toHaveCount(1)

    await page.getByRole('button', { name: 'Enter full canvas' }).click()
    await expect(page.getByRole('button', { name: 'Exit full canvas' })).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(page.getByRole('button', { name: 'Enter full canvas' })).toBeVisible()

    await page.getByRole('link', { name: 'Models', exact: true }).click()
    await expect(page.getByRole('dialog')).toContainText('Leave the Model Builder?')
    await page.getByRole('button', { name: 'Stay' }).click()
    await expect(page).toHaveURL(/\/builder$/)
    await page.getByRole('link', { name: 'Models', exact: true }).click()
    await page.getByRole('button', { name: 'Discard changes' }).click()
    await expect(page).toHaveURL(/\/models$/)
    await expect(page.getByRole('heading', { name: 'Saved models' })).toBeVisible()
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('covers responsive drawers and saved-model workflow routes', async ({ page, request }) => {
  const fixture = await createProject(request, `Angular workflow ${Date.now()}`)
  let duplicateId: string | undefined
  try {
    await page.setViewportSize({ width: 1023, height: 768 })
    await page.goto('/builder')
    await expect(page.getByRole('button', { name: 'Catalog', exact: true })).toBeVisible()
    await expect(page.locator('.builder-catalog')).toBeHidden()
    await page.getByRole('button', { name: 'Catalog', exact: true }).click()
    await expect(page.locator('.builder-catalog.is-open')).toBeVisible()
    await page.getByRole('textbox', { name: 'Search nodes' }).fill('Dense')
    await page.getByRole('button', { name: /Dense \/ Linear/ }).click()
    await page.getByRole('button', { name: 'Inspector', exact: true }).click()
    await expect(page.locator('.builder-inspector.is-open')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Properties' })).toBeVisible()

    await page.goto(`/models/${fixture.id}`)
    await expect(page.getByRole('heading', { name: fixture.name })).toBeVisible()
    await page.getByRole('link', { name: 'Compilation', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Configure and generate source' })).toBeVisible()
    await page.getByRole('button', { name: 'Configure compilation' }).click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    await dialog.getByRole('button', { name: 'Apply configuration' }).click()
    await expect(page.getByRole('button', { name: /Saved compilation settings/ })).toBeVisible()

    await page.getByRole('link', { name: 'Export', exact: true }).click()
    await expect(page.getByRole('heading', { name: `Export ${fixture.name}` })).toBeVisible()
    await expect(page.getByRole('button', { name: /Project bundle/ })).toBeEnabled()
    const download = page.waitForEvent('download')
    await page.getByRole('button', { name: /Export artifact/ }).click()
    await expect((await download).suggestedFilename()).toMatch(/\.json$/)

    await page.getByRole('link', { name: 'Inference', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible()

    await page.getByRole('link', { name: 'Overview', exact: true }).click()
    await page.getByRole('button', { name: 'Duplicate' }).click()
    await expect(page).toHaveURL(/\/models\/[0-9a-f-]+$/)
    duplicateId = page.url().split('/').pop()
    await expect(page.getByRole('heading', { name: `${fixture.name} copy` })).toBeVisible()
    await page.getByRole('button', { name: 'Delete model' }).click()
    await page.getByRole('button', { name: 'Delete permanently' }).click()
    await expect(page).toHaveURL(/\/models$/)
    duplicateId = undefined
  } finally {
    if (duplicateId) await deleteProject(request, duplicateId)
    await deleteProject(request, fixture.id)
    await page.setViewportSize({ width: 1280, height: 800 })
  }
})
