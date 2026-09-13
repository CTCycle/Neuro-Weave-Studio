import { mkdir } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext } from '@playwright/test'

const visualEvidenceDir = path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'model-builder-evaluation-20260902', 'screenshots')
const kerasImportFixture = Buffer.from('UEsDBBQAAAAIALVcKl3fELsi1QAAAL8BAAALAAAAY29uZmlnLmpzb26dj8FqwzAMhl/F+JxLu512K/RSGIOxYylBjZVUzLFTS14pIe9eqxAoaU+7iV+f9EmjbTww1wF6tB/G/uA5YxACbytjmxha6ko+2hn43phdP8Qk6MwXyiWmXyU9XDFxAfbjYuMuDFk+tb3YeARpTjWfYFBuH7L3lXk/FMjJ9Z7Z1keQt7UO8gCJNWzBM5YgQdehewhm4RlqUqedpsosr9liKFtev1YG3dzOgUTfWZcaGqE/EIpBqYQ+3wnG+kigkKSM/5DFLHrmg231ZGPq+kjupfAwTTdQSwMEFAAAAAgAtVwqXafnk7AYAAAAFgAAAA0AAABtZXRhZGF0YS5qc29uq1bKTi1KLI4vSy0qzszPU7JSUDJWqgUAUEsBAhQAFAAAAAgAtVwqXd8QuyLVAAAAvwEAAAsAAAAAAAAAAAAAAIABAAAAAGNvbmZpZy5qc29uUEsBAhQAFAAAAAgAtVwqXafnk7AYAAAAFgAAAA0AAAAAAAAAAAAAAIAB/gAAAG1ldGFkYXRhLmpzb25QSwUGAAAAAAIAAgB0AAAAQQEAAAAA', 'base64')

type CreatedProject = { id: string; current_revision: number; name: string }

function graph() {
  return {
    schema_version: 5,
    nodes: [
      {
        id: 'input-tabular-e2e',
        definition_id: 'input.tabular',
        definition_version: 1,
        label: 'Features',
        position: { x: 48, y: 144 },
        parameters: { shape: [8], dtype: 'float32' },
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

function fanOutGraph() {
  const base = graph()
  const denseParameters = (units: number, activation: string) => ({
    units,
    activation,
    use_bias: true,
    kernel_initializer: 'glorot_uniform',
    bias_initializer: 'zeros',
    kernel_regularizer: 'none',
    bias_regularizer: 'none',
    activity_regularizer: 'none',
    kernel_constraint: 'none',
    bias_constraint: 'none',
  })
  return {
    ...base,
    nodes: [
      { ...base.nodes[0], id: 'input-branch-e2e', label: 'Branch source', position: { x: 48, y: 160 } },
      { ...base.nodes[1], id: 'dense-primary-e2e', label: 'Primary branch', position: { x: 352, y: 80 }, parameters: denseParameters(16, 'relu') },
      { ...base.nodes[2], id: 'merge-add-e2e', definition_id: 'merge.add', label: 'Merge', position: { x: 656, y: 160 }, parameters: {} },
      { ...base.nodes[1], id: 'dense-downstream-e2e', label: 'Downstream', position: { x: 960, y: 80 }, parameters: denseParameters(16, 'relu') },
      { ...base.nodes[3], id: 'output-branch-e2e', label: 'Prediction', position: { x: 1264, y: 80 } },
      { ...base.nodes[1], id: 'dense-fanout-e2e', label: 'Fan-out target', position: { x: 352, y: 360 }, parameters: denseParameters(8, 'tanh') },
      { ...base.nodes[2], id: 'mask-target-e2e', definition_id: 'transformer.attention_mask', label: 'Mask target', position: { x: 656, y: 440 }, parameters: {} },
    ],
    edges: [
      { id: 'edge-branch-1', source_node_id: 'input-branch-e2e', source_port_id: 'output', target_node_id: 'dense-primary-e2e', target_port_id: 'input' },
      { id: 'edge-branch-2', source_node_id: 'dense-primary-e2e', source_port_id: 'output', target_node_id: 'merge-add-e2e', target_port_id: 'inputs' },
      { id: 'edge-branch-3', source_node_id: 'merge-add-e2e', source_port_id: 'output', target_node_id: 'dense-downstream-e2e', target_port_id: 'input' },
      { id: 'edge-branch-4', source_node_id: 'dense-downstream-e2e', source_port_id: 'output', target_node_id: 'output-branch-e2e', target_port_id: 'input' },
    ],
  }
}

async function createProject(request: APIRequestContext, name: string, modelGraph = graph()): Promise<CreatedProject> {
  const response = await request.post('/api/projects', {
    data: { name, description: 'Browser regression fixture', execution_backend: 'pytorch', keras_enabled: true, graph: modelGraph },
  })
  expect(response.ok()).toBeTruthy()
  return response.json() as Promise<CreatedProject>
}

async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
  const response = await request.delete(`/api/projects/${id}`)
  expect([204, 404]).toContain(response.status())
}

async function dragConnection(page: import('@playwright/test').Page, sourceId: string, targetId: string): Promise<void> {
  const source = page.locator(`[data-f-connector-id="${sourceId}"]`)
  const target = page.locator(`[data-f-connector-id="${targetId}"]`)
  const sourceBounds = await source.boundingBox()
  const targetBounds = await target.boundingBox()
  expect(sourceBounds).not.toBeNull()
  expect(targetBounds).not.toBeNull()
  await page.mouse.move(sourceBounds!.x + sourceBounds!.width / 2, sourceBounds!.y + sourceBounds!.height / 2)
  await page.mouse.down()
  await page.mouse.move(targetBounds!.x + targetBounds!.width / 2, targetBounds!.y + targetBounds!.height / 2, { steps: 8 })
}

function backendCompatibilityGraph() {
  const base = graph()
  return {
    ...base,
    nodes: [
      { ...base.nodes[0], id: 'input-sequence-compat-e2e', definition_id: 'input.sequence', label: 'Sequence input', position: { x: 48, y: 160 }, parameters: { shape: [4, 3], dtype: 'float32' } },
      { ...base.nodes[1], id: 'simple-rnn-compat-e2e', definition_id: 'recurrent.simple_rnn', label: 'Simple RNN', position: { x: 400, y: 160 }, parameters: { units: 8, return_sequences: true } },
      { ...base.nodes[3], id: 'output-sequence-compat-e2e', definition_id: 'output.sequence', label: 'Sequence output', position: { x: 752, y: 160 }, parameters: {} },
    ],
    edges: [
      { id: 'edge-compat-e2e-1', source_node_id: 'input-sequence-compat-e2e', source_port_id: 'output', target_node_id: 'simple-rnn-compat-e2e', target_port_id: 'input' },
      { id: 'edge-compat-e2e-2', source_node_id: 'simple-rnn-compat-e2e', source_port_id: 'output', target_node_id: 'output-sequence-compat-e2e', target_port_id: 'input' },
    ],
  }
}

async function waitForCanvasFit(page: import('@playwright/test').Page): Promise<void> {
  await expect.poll(async () => {
    const style = await page.locator('f-canvas').getAttribute('style')
    return Boolean(style && !style.includes('matrix(1, 0, 0, 1, 0, 0)'))
  }, { timeout: 5_000 }).toBeTruthy()
}

test('imports a Keras architecture through the browser file input', async ({ page, request }) => {
  let projectId: string | undefined
  try {
    await page.goto('/models')
    const importResponse = page.waitForResponse((response) => response.url().includes('/api/imports/keras-architecture') && response.request().method() === 'POST')
    await page.locator('input[type="file"][accept*=".keras"]').setInputFiles({
      name: 'browser-import.keras',
      mimeType: 'application/vnd.keras.zip',
      buffer: kerasImportFixture,
    })
    const response = await importResponse
    expect(response.status()).toBe(201)
    const imported = await response.json() as { project: { id: string; name: string; graph: { nodes: unknown[] } } }
    projectId = imported.project.id

    await expect(page).toHaveURL(new RegExp(`/models/${projectId}$`))
    await expect(page.getByRole('heading', { name: 'browser import' })).toBeVisible()
    expect(imported.project.graph.nodes.length).toBe(4)
    const persisted = await (await request.get(`/api/projects/${projectId}`)).json() as { name: string; graph: { nodes: unknown[] } }
    expect(persisted.name).toBe('browser import')
    expect(persisted.graph.nodes).toHaveLength(4)
  } finally {
    if (projectId) await deleteProject(request, projectId)
  }
})

test('deletes a model from its catalog card without navigating', async ({ page, request }) => {
  const fixture = await createProject(request, `Angular catalog delete ${Date.now()}`)
  try {
    await page.goto('/models')
    const card = page.locator('.model-card').filter({ hasText: fixture.name })
    await expect(card).toBeVisible()
    await expect(card.getByRole('link')).toHaveAttribute('href', `/models/${fixture.id}`)

    await card.getByRole('button', { name: `Delete model ${fixture.name}` }).click()
    const dialog = page.getByRole('alertdialog')
    await expect(dialog).toContainText(`Delete “${fixture.name}”?`)
    await expect(page).toHaveURL(/\/models$/)
    await dialog.getByRole('button', { name: 'Cancel' }).click()
    await expect(card).toBeVisible()

    await card.getByRole('button', { name: `Delete model ${fixture.name}` }).click()
    const deleteResponse = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'DELETE')
    await page.getByRole('button', { name: 'Delete permanently' }).click()
    expect((await deleteResponse).status()).toBe(204)
    await expect(card).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Dismiss catalog notice' })).toContainText(`Deleted “${fixture.name}”.`)
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('keeps catalog deletion cancellable and reports deletion errors', async ({ page, request }) => {
  const fixture = await createProject(request, `Angular catalog error ${Date.now()}`)
  let releaseDelete: (() => void) | undefined
  const deleteGate = new Promise<void>((resolve) => { releaseDelete = resolve })
  try {
    await page.route(`**/api/projects/${fixture.id}`, async (route) => {
      if (route.request().method() !== 'DELETE') { await route.continue(); return }
      await deleteGate
      await route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'Catalog delete regression' }) })
    })
    await page.goto('/models')
    const card = page.locator('.model-card').filter({ hasText: fixture.name })
    await card.getByRole('button', { name: `Delete model ${fixture.name}` }).click()
    const dialog = page.getByRole('alertdialog')
    const confirm = dialog.locator('button.danger-action')
    const cancel = dialog.getByRole('button', { name: 'Cancel' })
    const deleteResponse = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'DELETE')
    await confirm.click()
    await expect(confirm).toBeDisabled()
    await expect(cancel).toBeDisabled()
    releaseDelete?.()
    expect((await deleteResponse).status()).toBe(500)
    await expect(dialog.locator('.delete-dialog-error')).toContainText('Delete failed')
    await expect(card).toBeVisible()
    await cancel.click()
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('keeps saved builder profile, validation, and API payload synchronized', async ({ page, request }) => {
  const fixture = await createProject(request, `Angular state ${Date.now()}`)
  try {
    await page.goto(`/builder/${fixture.id}`)
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
    await expect(page.getByRole('combobox', { name: 'Execution backend' })).toHaveAttribute('data-value', 'pytorch')
    await expect(page.getByLabel('Enable Keras')).toBeChecked()
    await expect(page.locator('.nw-flow-node')).toHaveCount(4)
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(3)
    await expect(page.locator('f-connection path.f-connection-selection').first()).toHaveAttribute('d', /L/)

    const validationResponse = page.waitForResponse((response) => response.url().includes('/api/validation') && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Validate', exact: true }).click()
    expect((await validationResponse).ok()).toBeTruthy()
    await expect(page.locator('.validation-summary')).toContainText('Ready')
    await page.locator('.builder-notice').click()
    await page.locator('.validation-summary').click()
    await expect(page.getByText('No validation findings.')).toBeVisible()
    await page.screenshot({ path: path.join(visualEvidenceDir, 'final-builder-valid.png'), fullPage: true })

    await page.getByRole('button', { name: 'Description', exact: true }).click()
    const descriptionDialog = page.getByRole('dialog', { name: 'Model description editor' })
    await descriptionDialog.getByLabel('Model description').fill('Persisted from the Angular builder.')
    await descriptionDialog.getByRole('button', { name: 'Save', exact: true }).click()
    const updatedName = `${fixture.name} synced`
    await page.getByLabel('Model name').fill(updatedName)
    const updateResponse = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT')
    await page.getByRole('button', { name: 'Save model', exact: true }).click()
    const saved = await (await updateResponse).json() as { name: string; description: string; execution_backend: string; keras_enabled: boolean; current_revision: number; graph: { schema_version: number; nodes: unknown[]; edges: unknown[] } }
    expect(saved.name).toBe(updatedName)
    expect(saved.description).toBe('Persisted from the Angular builder.')
    expect(saved.execution_backend).toBe('pytorch')
    expect(saved.keras_enabled).toBe(true)
    expect(saved.graph).toMatchObject({ schema_version: 5 })
    expect(saved.graph.nodes).toHaveLength(4)
    expect(saved.graph.edges).toHaveLength(3)
    await expect(page.getByText(`Saved revision ${saved.current_revision}`)).toBeVisible()
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('keeps Inspector controls contract-driven and Backend picker dark and keyboard accessible', async ({ page, request }) => {
  const fixture = await createProject(request, `Angular Inspector ${Date.now()}`)
  try {
    await page.goto(`/builder/${fixture.id}`)
    const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
    await page.locator('.nw-flow-node').nth(1).click()
    await expect(inspector.locator('.inspector-node-header')).toBeVisible()
    await expect(inspector.locator('.inspector-node-header')).toContainText('Hidden layer')
    await expect(inspector.locator('.port-spec-row')).toHaveCount(2)
    await expect(inspector.locator('.port-spec-section')).toContainText('Connected')
    const initializer = inspector.locator('label.inspector-field').filter({ hasText: 'Kernel initializer' })
    await expect(initializer.locator('select')).toHaveCount(1)
    await expect(initializer.locator('textarea')).toHaveCount(0)
    const useBias = inspector.locator('label.inspector-field').filter({ hasText: 'Use bias' })
    await expect(useBias.locator('input[type="checkbox"]')).toHaveCount(1)

    const backend = page.getByRole('combobox', { name: 'Execution backend' })
    await backend.click()
    const listbox = page.getByRole('listbox', { name: 'Execution backend' })
    await expect(listbox).toBeVisible()
    await expect(listbox.locator('[role="option"]')).toHaveCount(3)
    const menuBackground = await listbox.evaluate((element) => getComputedStyle(element).backgroundColor)
    expect(menuBackground).not.toBe('rgb(255, 255, 255)')
    await backend.press('ArrowDown')
    await backend.press('Enter')
    await expect(backend).toHaveAttribute('data-value', /^(tensorflow|jax)$/)
    await backend.press('Escape')
    await expect(listbox).toBeHidden()
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('renders stable connection previews, fan-out connections, and cycle rejection feedback', async ({ page, request }) => {
  const branchGraph = fanOutGraph()
  const fixture = await createProject(request, `Angular graph UX ${Date.now()}`, branchGraph)
  try {
    await page.goto(`/builder/${fixture.id}`)
    await expect(page.locator('.nw-flow-node')).toHaveCount(7)
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(4)
    await waitForCanvasFit(page)

    await dragConnection(page, 'input-branch-e2e|output|output', 'dense-fanout-e2e|input|input')
    const preview = page.locator('f-connection-for-create path').first()
    await expect(preview).toBeVisible()
    await expect(preview).toHaveAttribute('d', /L/)
    await expect(page.locator('[data-f-connector-id="dense-fanout-e2e|input|input"]')).toHaveClass(/f-connector-connectable/)
    await page.mouse.up()
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(5)

    const saveResponse = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT')
    await page.getByRole('button', { name: 'Save model', exact: true }).click()
    const saved = await (await saveResponse).json() as { graph: { edges: unknown[] } }
    expect(saved.graph.edges).toHaveLength(5)

    await dragConnection(page, 'dense-downstream-e2e|output|output', 'merge-add-e2e|input|inputs')
    await expect(page.locator('[data-f-connector-id="merge-add-e2e|input|inputs"]')).toHaveClass(/f-connector-connectable/)
    await page.mouse.up()
    await expect(page.getByRole('status')).toContainText('cycle')
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(5)
    await page.screenshot({ path: path.join(visualEvidenceDir, 'final-builder-validation.png'), fullPage: true })

    await dragConnection(page, 'input-branch-e2e|output|output', 'mask-target-e2e|input|input')
    await expect(page.locator('[data-f-connector-id="mask-target-e2e|input|input"]')).not.toHaveClass(/f-connector-connectable/)
    await page.mouse.up()
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(5)
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('retains incompatible drafts while disabling catalog, compilation, and export actions', async ({ page, request }) => {
  test.setTimeout(90_000)
  const fixture = await createProject(request, `Angular compatibility ${Date.now()}`, backendCompatibilityGraph())
  const evidenceDir = path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'backend-compatibility-consolidation')
  await mkdir(evidenceDir, { recursive: true })
  try {
    await page.goto(`/builder/${fixture.id}`)
    const kerasToggle = page.getByLabel('Enable Keras', { exact: true })
    await expect(kerasToggle).toBeChecked()
    await kerasToggle.uncheck()

    const incompatibleNode = page.locator('[data-node-id="simple-rnn-compat-e2e"]')
    await expect(incompatibleNode).toHaveClass(/is-profile-incompatible/)
    await expect(incompatibleNode).toHaveAttribute('aria-label', /Repair required/)

    await page.getByRole('textbox', { name: 'Search nodes' }).fill('Simple RNN')
    const catalogItem = page.locator('.catalog-item').filter({ hasText: 'Simple RNN' }).first()
    await expect(catalogItem).toBeDisabled()
    await expect(catalogItem).toHaveAttribute('title', /not supported by Native PyTorch/i)

    await incompatibleNode.click()
    const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
    await expect(inspector.locator('label.inspector-field').filter({ hasText: 'Units' }).locator('input')).toBeDisabled()
    await expect(inspector.locator('label.inspector-field').filter({ hasText: 'Return Sequences' }).locator('input')).toBeDisabled()
    await page.screenshot({ path: path.join(evidenceDir, 'profile-incompatible-builder.png'), fullPage: true })

    const saveResponse = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT')
    await page.getByRole('button', { name: 'Save model', exact: true }).click()
    const saved = await (await saveResponse).json() as { current_revision: number; keras_enabled: boolean; validation_status: string; graph: { nodes: unknown[] } }
    expect(saved.keras_enabled).toBe(false)
    expect(saved.validation_status).toBe('invalid')
    expect(saved.graph.nodes).toHaveLength(3)
    await expect(page.locator('.builder-badge')).toContainText('Saved')

    const persisted = await (await request.get(`/api/projects/${fixture.id}`)).json() as { keras_enabled: boolean; validation_status: string; graph: { nodes: Array<{ id: string }> } }
    expect(persisted).toMatchObject({ keras_enabled: false, validation_status: 'invalid' })
    expect(persisted.graph.nodes.map((node) => node.id)).toContain('simple-rnn-compat-e2e')

    await page.goto(`/models/${fixture.id}/compilation`)
    await expect(page.getByRole('button', { name: 'Compile source', exact: true })).toBeDisabled()
    await page.getByRole('button', { name: 'Validate profile', exact: true }).click()
    await expect(page.getByText(/Simple RNN is not supported by native pytorch/i)).toBeVisible()
    await page.screenshot({ path: path.join(evidenceDir, 'invalid-compilation.png'), fullPage: true })

    await page.goto(`/models/${fixture.id}/export`)
    await expect(page.getByRole('button', { name: /Project bundle/ })).toBeDisabled()
    await expect(page.getByRole('button', { name: /Project bundle/ })).toContainText('Fix the validation diagnostics')
    await expect(page.getByRole('button', { name: /PyTorch ExportedProgram/ })).toContainText('Unavailable for this profile')
    await page.screenshot({ path: path.join(evidenceDir, 'invalid-export-capabilities.png'), fullPage: true })

    await page.goto(`/builder/${fixture.id}`)
    await expect(page.getByLabel('Enable Keras', { exact: true })).not.toBeChecked()
    await page.getByLabel('Enable Keras', { exact: true }).check()
    await expect(page.locator('[data-node-id="simple-rnn-compat-e2e"]')).not.toHaveClass(/is-profile-incompatible/)
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('polls a compatible smoke-test job to a verified result', async ({ page, request }) => {
  test.setTimeout(120_000)
  const fixture = await createProject(request, `Angular compatibility smoke ${Date.now()}`)
  const evidenceDir = path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'backend-compatibility-consolidation')
  await mkdir(evidenceDir, { recursive: true })
  try {
    await page.goto(`/models/${fixture.id}/inference`)
    await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
    await expect(page.getByText(/keras · pytorch ·/i)).toBeVisible({ timeout: 75_000 })
    const submitResponse = page.waitForResponse((response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Run smoke test', exact: true }).click()
    const submitted = await submitResponse
    expect(submitted.ok()).toBeTruthy()
    const job = await submitted.json() as { id: string }
    await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 75_000 })
    await expect.poll(async () => {
      const response = await request.get(`/api/execution/jobs/${job.id}`)
      return response.ok() ? (await response.json() as { state: string }).state : 'missing'
    }, { timeout: 75_000 }).toBe('succeeded')
    await page.screenshot({ path: path.join(evidenceDir, 'smoke-job-succeeded.png'), fullPage: true })
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('exposes keyboard-traversable graph nodes and ports', async ({ page, request }) => {
  const fixture = await createProject(request, `Angular graph accessibility ${Date.now()}`, fanOutGraph())
  try {
    await page.goto(`/builder/${fixture.id}`)
    const nodes = page.locator('[data-node-id]')
    const ports = page.locator('.nw-flow-port[role="button"]')
    await expect(nodes).toHaveCount(7)
    await expect(nodes.first()).toHaveAttribute('tabindex', '0')
    await expect(nodes.first()).toHaveAttribute('role', 'group')
    await expect(ports).not.toHaveCount(0)

    const source = page.locator('[data-node-id="input-branch-e2e"] .nw-flow-port.output')
    const target = page.locator('[data-node-id="dense-fanout-e2e"] .nw-flow-port.input')
    await source.focus()
    await source.press('Enter')
    await expect(source).toHaveAttribute('aria-pressed', 'true')
    await expect(page.getByRole('status')).toContainText('Choose an input port')

    await target.focus()
    const validationResponse = page.waitForResponse((response) => response.url().includes('/api/validation/connection') && response.request().method() === 'POST')
    await target.press('Enter')
    expect((await validationResponse).ok()).toBeTruthy()
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(5)
    await expect(source).toHaveAttribute('aria-pressed', 'false')
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('keeps edge selection, inspector details, deletion, and undo synchronized', async ({ page, request }) => {
  const fixture = await createProject(request, `Angular edge state ${Date.now()}`)
  try {
    await page.goto(`/builder/${fixture.id}`)
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(3)
    await page.locator('f-connection path.f-connection-selection').first().click({ force: true })
    await expect(page.locator('f-connection.is-selected')).toHaveCount(1)
    await expect(page.getByRole('heading', { name: 'Connection' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Delete connection' })).toBeVisible()
    await expect(page.getByRole('complementary', { name: 'Properties inspector' })).toContainText('Features')
    await expect(page.getByRole('complementary', { name: 'Properties inspector' })).toContainText('Hidden layer')

    await page.locator('body').press('Delete')
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(2)
    await expect(page.getByRole('heading', { name: 'No node selected' })).toBeVisible()
    await page.getByRole('button', { name: 'Undo', exact: true }).click()
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(3)

    const persisted = await (await request.get(`/api/projects/${fixture.id}`)).json() as { graph: { edges: unknown[] } }
    expect(persisted.graph.edges).toHaveLength(3)
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('isolates builder tabs and exposes canvas and validation controls', async ({ page, request }) => {
  const fixture = await createProject(request, `Angular tabs ${Date.now()}`)
  try {
    await page.goto(`/builder/${fixture.id}`)
    const savedTab = page.locator('button[role="tab"]').filter({ hasText: fixture.name })
    await expect(savedTab).toBeVisible()
    await page.getByRole('button', { name: 'New model', exact: true }).click()
    const draftTab = page.locator('button[role="tab"]').filter({ hasText: 'Untitled model' })
    await expect(draftTab).toBeVisible()
    await page.getByRole('textbox', { name: 'Search nodes' }).fill('Dense')
    await page.getByRole('button', { name: /Dense \/ Linear/ }).click()
    await expect(page.locator('.nw-flow-node')).toHaveCount(1)

    await savedTab.click()
    await expect(page.locator('.nw-flow-node')).toHaveCount(4)
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(3)
    await page.getByRole('button', { name: 'Validate', exact: true }).click()
    await expect(page.locator('.validation-summary')).toContainText('Ready')
    await page.locator('.builder-notice').click()
    await page.locator('.validation-summary').click()
    await expect(page.getByRole('separator', { name: 'Resize validation panel' })).toBeVisible()

    await draftTab.click()
    await expect(page.locator('.nw-flow-node')).toHaveCount(1)
    await page.getByRole('button', { name: 'Hide minimap' }).click()
    await expect(page.getByRole('button', { name: 'Show minimap' })).toBeVisible()
    await page.getByRole('button', { name: 'Enter full canvas' }).click()
    await expect(page.getByRole('button', { name: 'Exit full canvas' })).toBeVisible()
    await page.getByRole('button', { name: 'Exit full canvas' }).click()
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('covers builder editing, history, focus mode, and dirty navigation', async ({ page, request }) => {
  const fixture = await createProject(request, `Angular builder ${Date.now()}`)
  try {
    await page.goto('/builder')
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
    await page.screenshot({ path: path.join(visualEvidenceDir, 'final-builder-empty.png'), fullPage: true })

    await page.getByRole('textbox', { name: 'Search nodes' }).fill('Dense')
    await page.getByRole('button', { name: /Dense \/ Linear/ }).click()
    await expect(page.locator('.nw-flow-node')).toHaveCount(1)
    await page.locator('.nw-flow-node').first().click()
    await expect(page.getByRole('heading', { name: 'Dense / Linear' })).toBeVisible()

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
    await page.getByRole('button', { name: 'Close node catalog', exact: true }).click()
    await expect(page.locator('.builder-catalog')).toBeHidden()
    await expect(page.getByRole('button', { name: 'Catalog', exact: true })).toBeFocused()
    await page.getByRole('button', { name: 'Catalog', exact: true }).click()
    await page.getByRole('textbox', { name: 'Search nodes' }).fill('Dense')
    await page.getByRole('button', { name: /Dense \/ Linear/ }).click()
    await page.getByRole('button', { name: 'Close node catalog', exact: true }).click()
    await page.locator('.nw-flow-node').first().click()
    await page.getByRole('button', { name: 'Inspector', exact: true }).click()
    await expect(page.locator('.builder-inspector.is-open')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Dense / Linear' })).toBeVisible()
    await page.getByRole('button', { name: 'Close properties inspector', exact: true }).click()
    await expect(page.locator('#builder-inspector-drawer')).toBeHidden()
    await expect(page.getByRole('button', { name: 'Inspector', exact: true })).toBeFocused()

    await page.goto(`/models/${fixture.id}`)
    await expect(page.locator('.workspace-actions-section')).toHaveCount(0)
    const overviewBounds = await page.evaluate(() => ({
      documentOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
      contentOverflow: (document.querySelector('.app-content')?.scrollWidth ?? 0) > (document.querySelector('.app-content')?.clientWidth ?? 0) + 1,
    }))
    expect(overviewBounds.documentOverflow).toBeFalsy()
    expect(overviewBounds.contentOverflow).toBeFalsy()

    await expect(page.getByRole('heading', { name: fixture.name })).toBeVisible()
    await page.getByRole('link', { name: 'Compilation', exact: true }).click()
    await expect(page.getByRole('heading', { name: /Compile revision/ })).toBeVisible()
    await page.getByRole('button', { name: 'Configure', exact: true }).click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    await expect(dialog.getByLabel('Loss')).toHaveValue('binary_crossentropy')
    await dialog.getByRole('button', { name: 'Apply configuration' }).click()
    await expect(page.getByRole('button', { name: /Saved validated compilation settings/ })).toBeVisible()
    const sourceResponse = page.waitForResponse((response) => response.url().includes(`/api/exports/${fixture.id}/source`) && response.request().method() === 'GET')
    await page.getByRole('button', { name: 'Compile source', exact: true }).click()
    expect((await sourceResponse).ok()).toBeTruthy()
    await expect(page.getByRole('heading', { name: /Python · revision/ })).toBeVisible()
    await expect(page.locator('.generated-code code')).toContainText('keras')

    await page.getByRole('link', { name: 'Export', exact: true }).click()
    await expect(page.getByRole('heading', { name: `Export ${fixture.name}` })).toBeVisible()
    await expect(page.getByRole('button', { name: /Project bundle/ })).toBeEnabled()
    const download = page.waitForEvent('download')
    await page.getByRole('button', { name: /Export artifact/ }).click()
    await expect((await download).suggestedFilename()).toMatch(/\.json$/)

    await page.goto(`/models/${fixture.id}/inference`)
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

test('keeps read-only previews, compact Builder controls, and modal bounds usable', async ({ page, request }) => {
  test.setTimeout(60_000)
  const fixture = await createProject(request, `Angular layout ${Date.now()}`)
  try {
    await page.setViewportSize({ width: 1280, height: 800 })
    await page.goto(`/models/${fixture.id}`)
    await expect(page.locator('.readonly-graph-preview')).toBeVisible()
    await expect(page.locator('.readonly-graph-preview .canvas-editor-tools')).toHaveCount(0)
    await expect(page.locator('.readonly-graph-preview .canvas-hint')).toHaveCount(0)
    await expect(page.getByText('Drop a node to build the model', { exact: false })).toHaveCount(0)

    const overviewGeometry = await page.evaluate(() => {
      const preview = document.querySelector<HTMLElement>('.overview-preview')?.getBoundingClientRect()
      const summary = document.querySelector<HTMLElement>('.overview-summary > .summary-card')?.getBoundingClientRect()
      const readinessRows = [...document.querySelectorAll<HTMLElement>('.readiness-stack > div')]
      const hasRowOverlap = readinessRows.some((row) => {
        const children = [...row.children].map((child) => child.getBoundingClientRect())
        return children.some((left, index) => children.slice(index + 1).some((right) => left.right > right.left + 0.5 && right.right > left.left + 0.5 && left.bottom > right.top + 0.5 && right.bottom > left.top + 0.5))
      })
      const wrappingRegions = [...document.querySelectorAll<HTMLElement>('.detail-list dd, .readiness-stack, .readiness-stack > div')]
      return {
        alignedAtTop: Boolean(preview && summary && Math.abs(preview.top - summary.top) <= 1),
        noDocumentOverflow: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
        noReadinessOverflow: wrappingRegions.every((element) => element.scrollWidth <= element.clientWidth + 1),
        noReadinessOverlap: !hasRowOverlap,
        modelActionsRemoved: !document.querySelector('.workspace-actions-section'),
      }
    })
    expect(overviewGeometry.alignedAtTop).toBeTruthy()
    expect(overviewGeometry.noDocumentOverflow).toBeTruthy()
    expect(overviewGeometry.noReadinessOverflow).toBeTruthy()
    expect(overviewGeometry.noReadinessOverlap).toBeTruthy()
    expect(overviewGeometry.modelActionsRemoved).toBeTruthy()

    await page.goto(`/builder/${fixture.id}`)
    await expect(page.locator('.nw-flow-node')).toHaveCount(4)
    await expect(page.getByRole('button', { name: 'Lock canvas' })).toBeVisible()
    const builderLayout = await page.evaluate(() => {
      const rect = (selector: string): DOMRect => document.querySelector(selector)?.getBoundingClientRect() ?? new DOMRect()
      const catalog = rect('.catalog-panel')
      const center = rect('.builder-center-workspace')
      const inspector = rect('nw-properties-inspector > aside')
      const canvas = rect('.canvas-shell')
      const validation = rect('.validation-panel')
      return {
        sidePanelsStretch: Math.abs(catalog.height - center.height) <= 1 && Math.abs(inspector.height - center.height) <= 1,
        validationSpansCanvas: Math.abs(validation.width - canvas.width) <= 1 && Math.abs(validation.left - canvas.left) <= 1,
      }
    })
    expect(builderLayout.sidePanelsStretch).toBeTruthy()
    expect(builderLayout.validationSpansCanvas).toBeTruthy()

    const toolbarMetrics = await page.locator('.builder-toolbar').evaluate((element) => ({
      height: element.getBoundingClientRect().height,
      buttonFontSize: Number.parseFloat(getComputedStyle(element.querySelector('button')!).fontSize),
    }))
    expect(toolbarMetrics.height).toBeLessThanOrEqual(50)
    expect(toolbarMetrics.buttonFontSize).toBeGreaterThanOrEqual(11)

    const canvasBounds = await page.locator('.canvas-shell').evaluate((element) => {
      const bounds = element.getBoundingClientRect()
      return { left: bounds.left, top: bounds.top, width: bounds.width, height: bounds.height }
    })
    const dragStart = { x: canvasBounds.left + canvasBounds.width / 2, y: canvasBounds.top + canvasBounds.height - 20 }
    const readCanvasTransform = () => page.locator('f-canvas').evaluate((element) => `${element.getAttribute('style') ?? ''}:${getComputedStyle(element).transform}`)
    await waitForCanvasFit(page)
    const lockedTransform = await readCanvasTransform()
    await page.getByRole('button', { name: 'Lock canvas' }).click()
    const unlockButton = page.getByRole('button', { name: 'Unlock canvas' })
    await expect(unlockButton).toHaveAttribute('aria-pressed', 'true')
    await expect(page.getByRole('button', { name: 'Zoom in' })).toBeDisabled()
    await page.mouse.move(dragStart.x, dragStart.y)
    await page.mouse.down()
    await page.mouse.move(dragStart.x + 150, dragStart.y - 80, { steps: 5 })
    await page.mouse.up()
    await page.mouse.wheel(0, 500)
    expect(await readCanvasTransform()).toBe(lockedTransform)
    await page.keyboard.press('Control+l')
    expect(await readCanvasTransform()).toBe(lockedTransform)
    await expect(page.getByRole('heading', { name: 'No node selected' })).toBeVisible()

    await unlockButton.click()
    await expect(page.getByRole('button', { name: 'Lock canvas' })).toHaveAttribute('aria-pressed', 'false')
    await page.mouse.move(dragStart.x, dragStart.y)
    await page.mouse.down()
    await page.mouse.move(dragStart.x + 150, dragStart.y - 80, { steps: 5 })
    await page.mouse.up()
    expect(await readCanvasTransform()).not.toBe(lockedTransform)
    await page.locator('.nw-flow-node').first().click()
    await expect(page.getByRole('heading', { name: 'Features' })).toBeVisible()

    await page.setViewportSize({ width: 560, height: 800 })
    await page.goto(`/builder/${fixture.id}`)
    await expect(page.getByRole('button', { name: 'Validate', exact: true })).toBeVisible()
    const toolbarBounds = await page.locator('.builder-toolbar').evaluate((element) => ({
      scrollWidth: element.scrollWidth,
      clientWidth: element.clientWidth,
      visibleActions: [...element.querySelectorAll<HTMLButtonElement>('.builder-actions button')].every((button) => button.getBoundingClientRect().right <= element.getBoundingClientRect().right + 1),
    }))
    expect(toolbarBounds.scrollWidth).toBeLessThanOrEqual(toolbarBounds.clientWidth + 1)
    expect(toolbarBounds.visibleActions).toBeTruthy()

    await page.goto(`/models/${fixture.id}/compilation`)
    await page.getByRole('button', { name: 'Configure', exact: true }).click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    const dialogBounds = await dialog.evaluate((element) => {
      const bounds = element.getBoundingClientRect()
      return { right: bounds.right, bottom: bounds.bottom, viewportWidth: window.innerWidth, viewportHeight: window.innerHeight }
    })
    expect(dialogBounds.right).toBeLessThanOrEqual(dialogBounds.viewportWidth + 1)
    expect(dialogBounds.bottom).toBeLessThanOrEqual(dialogBounds.viewportHeight + 1)
    await expect(dialog.getByRole('button', { name: 'Apply configuration' })).toBeVisible()
  } finally {
    await deleteProject(request, fixture.id)
    await page.setViewportSize({ width: 1280, height: 800 })
  }
})
