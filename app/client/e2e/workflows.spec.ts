// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

const defaultCampaignEvidenceDir = path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260923', 'evidence')
const defaultVisualEvidenceDir = path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'model-builder-evaluation-20260902', 'screenshots')
const campaignEvidenceDir = path.resolve(process.env.NW_VALIDATION_EVIDENCE_DIR ?? defaultCampaignEvidenceDir)
const visualEvidenceDir = process.env.NW_VALIDATION_EVIDENCE_DIR ? campaignEvidenceDir : defaultVisualEvidenceDir
const kerasImportFixture = Buffer.from('UEsDBBQAAAAIALVcKl3fELsi1QAAAL8BAAALAAAAY29uZmlnLmpzb26dj8FqwzAMhl/F+JxLu512K/RSGIOxYylBjZVUzLFTS14pIe9eqxAoaU+7iV+f9EmjbTww1wF6tB/G/uA5YxACbytjmxha6ko+2hn43phdP8Qk6MwXyiWmXyU9XDFxAfbjYuMuDFk+tb3YeARpTjWfYFBuH7L3lXk/FMjJ9Z7Z1keQt7UO8gCJNWzBM5YgQdehewhm4RlqUqedpsosr9liKFtev1YG3dzOgUTfWZcaGqE/EIpBqYQ+3wnG+kigkKSM/5DFLHrmg231ZGPq+kjupfAwTTdQSwMEFAAAAAgAtVwqXafnk7AYAAAAFgAAAA0AAABtZXRhZGF0YS5qc29uq1bKTi1KLI4vSy0qzszPU7JSUDJWqgUAUEsBAhQAFAAAAAgAtVwqXd8QuyLVAAAAvwEAAAsAAAAAAAAAAAAAAIABAAAAAGNvbmZpZy5qc29uUEsBAhQAFAAAAAgAtVwqXafnk7AYAAAAFgAAAA0AAAAAAAAAAAAAAIAB/gAAAG1ldGFkYXRhLmpzb25QSwUGAAAAAAIAAgB0AAAAQQEAAAAA', 'base64')

type CreatedProject = { id: string; current_revision: number; name: string }
type WorkflowFixtures = { page: Page; request: APIRequestContext }

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

function sequenceInspectionGraph() {
  const base = graph()
  return {
    ...base,
    nodes: [
      { ...base.nodes[0], id: 'inspection-sequence-input', definition_id: 'input.sequence', label: 'Sequence input', position: { x: 48, y: 144 }, parameters: { shape: [3, 4], dtype: 'float32' } },
      { ...base.nodes[1], id: 'inspection-rnn', definition_id: 'recurrent.simple_rnn', label: 'Sequence encoder', position: { x: 400, y: 144 }, parameters: { units: 4, return_sequences: true, return_state: true } },
      { ...base.nodes[3], id: 'inspection-sequence-output', definition_id: 'output.sequence', label: 'Sequence output', position: { x: 752, y: 144 }, parameters: {} },
    ],
    edges: [
      { id: 'inspection-sequence-edge-1', source_node_id: 'inspection-sequence-input', source_port_id: 'output', target_node_id: 'inspection-rnn', target_port_id: 'input' },
      { id: 'inspection-sequence-edge-2', source_node_id: 'inspection-rnn', source_port_id: 'output', target_node_id: 'inspection-sequence-output', target_port_id: 'input' },
    ],
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
      { ...base.nodes[1], id: 'dense-downstream-e2e', label: 'Downstream', position: { x: 960, y: 80 }, parameters: denseParameters(1, 'sigmoid') },
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
      { ...base.nodes[1], id: 'simple-rnn-compat-e2e', definition_id: 'recurrent.simple_rnn', label: 'Simple RNN', position: { x: 400, y: 160 }, parameters: { units: 8, return_sequences: true, return_state: true } },
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
  test.setTimeout(90_000)
  let projectId: string | undefined
  try {
    await page.goto('/environment')
    const pytorchCard = page.locator('.backend-status-card-container').filter({ hasText: 'PyTorch' })
    const kerasRuntimeReady = async () => {
      const metadataResponse = await request.get('/api/metadata')
      const metadata = await metadataResponse.json() as {
        execution_backends: Array<{
          id: string
          runtime_apis?: Array<{ id: string; available: boolean; state?: string }>
        }>
      }
      const keras = metadata.execution_backends.find((backend) => backend.id === 'pytorch')?.runtime_apis?.find((api) => api.id === 'keras')
      return Boolean(keras?.available && keras.state === 'verified')
    }
    if (!(await kerasRuntimeReady())) {
      const kerasCheck = pytorchCard.getByRole('button', { name: 'Check Keras', exact: true })
      await expect(kerasCheck).toBeVisible({ timeout: 15_000 })
      await kerasCheck.click()
    }
    await expect.poll(kerasRuntimeReady, { timeout: 90_000 }).toBeTruthy()
    await expect(pytorchCard).toContainText('Usable')

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

test('NW-T1-03 exports and reimports a project bundle through the browser', async ({ page, request }) => {
  const source = await createProject(request, `NW-T1-03 bundle ${Date.now()}`)
  let importedId: string | undefined
  try {
    const sourceResponse = await request.get(`/api/projects/${source.id}`)
    expect(sourceResponse.ok()).toBeTruthy()
    const original = await sourceResponse.json() as {
      name: string
      execution_backend: string
      keras_enabled: boolean
      graph: unknown
    }

    await page.goto(`/models/${source.id}/export`)
    await expect(page.getByRole('heading', { name: `Export ${source.name}` })).toBeVisible()
    const bundleOption = page.getByRole('button', { name: /Project bundle/ })
    await expect(bundleOption).toBeEnabled()
    await bundleOption.click()

    const downloadPromise = page.waitForEvent('download')
    await page.getByRole('button', { name: 'Export artifact', exact: true }).click()
    const download = await downloadPromise
    expect(download.suggestedFilename()).toMatch(/\.json$/i)
    const downloadPath = await download.path()
    expect(downloadPath).not.toBeNull()
    const bundleBytes = await readFile(downloadPath!)
    const bundle = JSON.parse(bundleBytes.toString('utf8')) as {
      format: string
      format_version: number
      project: {
        name: string
        execution_backend: string
        keras_enabled: boolean
        graph: unknown
      }
    }
    expect(bundle.format).toBe('neuro-weave-project')
    expect(bundle.format_version).toBe(7)
    expect(bundle.project).toMatchObject({
      name: original.name,
      execution_backend: original.execution_backend,
      keras_enabled: original.keras_enabled,
      graph: original.graph,
    })

    await page.goto('/models')
    const importResponsePromise = page.waitForResponse((response) =>
      response.url().includes('/api/projects/import') && response.request().method() === 'POST',
    )
    await page.locator('input[type="file"][accept*="application/json"]').setInputFiles({
      name: download.suggestedFilename(),
      mimeType: 'application/json',
      buffer: bundleBytes,
    })
    const importResponse = await importResponsePromise
    expect(importResponse.status()).toBe(201)
    const imported = await importResponse.json() as {
      id: string
      current_revision: number
      name: string
      execution_backend: string
      keras_enabled: boolean
      graph: unknown
    }
    importedId = imported.id

    expect(imported.id).not.toBe(source.id)
    expect(imported.current_revision).toBe(1)
    expect(imported).toMatchObject({
      name: original.name,
      execution_backend: original.execution_backend,
      keras_enabled: original.keras_enabled,
      graph: original.graph,
    })
    await expect(page).toHaveURL(new RegExp(`/models/${imported.id}$`))
    await expect(page.getByRole('heading', { name: original.name })).toBeVisible()

    const persistedResponse = await request.get(`/api/projects/${imported.id}`)
    expect(persistedResponse.ok()).toBeTruthy()
    expect(await persistedResponse.json()).toMatchObject({
      id: imported.id,
      current_revision: 1,
      name: original.name,
      execution_backend: original.execution_backend,
      keras_enabled: original.keras_enabled,
      graph: original.graph,
    })

    await page.goto(`/builder/${imported.id}`)
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 75_000 })
    await expect(page.locator('.nw-flow-node')).toHaveCount(4)
    await mkdir(campaignEvidenceDir, { recursive: true })
    await page.screenshot({ path: path.join(campaignEvidenceDir, 'nw-t1-03-project-bundle-builder.png'), fullPage: true })
  } finally {
    if (importedId) await deleteProject(request, importedId)
    await deleteProject(request, source.id)
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
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 75_000 })
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
    expect(saved.graph).toMatchObject({ schema_version: 6 })
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

test('NW-T2-02: validates connection previews, cardinality, inferred shapes, and rejected graph edits', async ({ page, request }) => {
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

    const rejectedShapeResponse = page.waitForResponse((response) => {
      if (!response.url().endsWith('/api/validation/connection') || response.request().method() !== 'POST') return false
      const body = response.request().postDataJSON() as { connection?: { target_node_id?: string } }
      return body.connection?.target_node_id === 'merge-add-e2e'
    })
    await dragConnection(page, 'dense-fanout-e2e|output|output', 'merge-add-e2e|input|inputs')
    await expect(page.locator('[data-f-connector-id="merge-add-e2e|input|inputs"]')).toHaveClass(/f-connector-connectable/)
    await expect(page.locator('[data-f-connector-id="dense-primary-e2e|input|input"]')).not.toHaveClass(/f-connector-connectable/)
    await page.mouse.up()
    const rejectedShape = await (await rejectedShapeResponse).json() as { valid: boolean; diagnostics: Array<{ code: string }> }
    expect(rejectedShape.valid).toBe(false)
    expect(rejectedShape.diagnostics.some((diagnostic) => diagnostic.code === 'incompatible_connection_shape')).toBe(true)
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(5)

    await page.locator('[data-node-id="dense-fanout-e2e"] .nw-flow-node').click()
    await page.getByLabel('Units').fill('16')
    await expect(page.getByLabel('Units')).toHaveValue('16')
    const acceptedShapeResponse = page.waitForResponse((response) => {
      if (!response.url().endsWith('/api/validation/connection') || response.request().method() !== 'POST') return false
      const body = response.request().postDataJSON() as { connection?: { target_node_id?: string } }
      return body.connection?.target_node_id === 'merge-add-e2e'
    })
    await dragConnection(page, 'dense-fanout-e2e|output|output', 'merge-add-e2e|input|inputs')
    await expect(page.locator('[data-f-connector-id="merge-add-e2e|input|inputs"]')).toHaveClass(/f-connector-connectable/)
    await page.mouse.up()
    const acceptedShape = await (await acceptedShapeResponse).json() as { valid: boolean }
    expect(acceptedShape.valid).toBe(true)
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(6)

    const saveResponse = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT')
    await page.getByRole('button', { name: 'Save model', exact: true }).click()
    const saved = await (await saveResponse).json() as { graph: { edges: unknown[] } }
    expect(saved.graph.edges).toHaveLength(6)

    await dragConnection(page, 'dense-downstream-e2e|output|output', 'merge-add-e2e|input|inputs')
    await expect(page.locator('[data-f-connector-id="merge-add-e2e|input|inputs"]')).toHaveClass(/f-connector-connectable/)
    await page.mouse.up()
    await expect(page.getByRole('status')).toContainText('cycle')
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(6)
    await page.screenshot({ path: path.join(visualEvidenceDir, 'nw-t2-02-cycle-rejection.png'), fullPage: true })

    await dragConnection(page, 'input-branch-e2e|output|output', 'mask-target-e2e|input|input')
    await expect(page.locator('[data-f-connector-id="mask-target-e2e|input|input"]')).not.toHaveClass(/f-connector-connectable/)
    await page.mouse.up()
    await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(6)

    await page.locator('[data-node-id="mask-target-e2e"] .nw-flow-node').click({ button: 'right' })
    await page.getByRole('menuitem', { name: 'Delete node', exact: true }).click()
    await expect(page.locator('.nw-flow-node')).toHaveCount(6)

    const validationResponse = page.waitForResponse((response) => {
      if (!response.url().endsWith('/api/validation') || response.request().method() !== 'POST') return false
      const body = response.request().postDataJSON() as { graph?: { nodes?: unknown[]; edges?: unknown[] } }
      return body.graph?.nodes?.length === 6 && body.graph.edges?.length === 6
    })
    await page.getByRole('button', { name: 'Validate', exact: true }).click()
    const validation = await (await validationResponse).json() as {
      valid: boolean
      shape_summaries: Array<{ node_id: string; outputs: Record<string, { shape: Array<number | null> }> }>
    }
    expect(validation.valid).toBe(true)
    const mergeShape = validation.shape_summaries.find((summary) => summary.node_id === 'merge-add-e2e')
    expect(mergeShape?.outputs.output.shape).toEqual([null, 16])
    await page.screenshot({ path: path.join(visualEvidenceDir, 'nw-t2-02-valid-merge-shape.png'), fullPage: true })
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('retains parameter-incompatible drafts while disabling compilation and export actions', async ({ page, request }) => {
  test.setTimeout(90_000)
  const fixture = await createProject(request, `Angular compatibility ${Date.now()}`, backendCompatibilityGraph())
  const evidenceDir = process.env.NW_VALIDATION_EVIDENCE_DIR
    ? campaignEvidenceDir
    : path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'backend-compatibility-consolidation')
  await mkdir(evidenceDir, { recursive: true })
  try {
    await page.goto(`/builder/${fixture.id}`)
    const kerasToggle = page.getByLabel('Enable Keras', { exact: true })
    await expect(kerasToggle).toBeChecked()
    await kerasToggle.uncheck()

    const incompatibleNode = page.locator('[data-node-id="simple-rnn-compat-e2e"]')
    await expect(incompatibleNode).toHaveClass(/has-validation-error/)
    await expect(incompatibleNode).toHaveAttribute('aria-label', 'Node Simple RNN')

    await page.getByRole('textbox', { name: 'Search nodes' }).fill('Simple RNN')
    const catalogItem = page.locator('.catalog-item').filter({ hasText: 'Simple RNN' }).first()
    await expect(catalogItem).toBeEnabled()

    await incompatibleNode.click()
    const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
    await expect(inspector.locator('label.inspector-field').filter({ hasText: 'Units' }).locator('input')).toBeEnabled()
    await expect(inspector.locator('label.inspector-field').filter({ hasText: 'Return Sequences' }).locator('input')).toBeEnabled()
    await expect(inspector.locator('label.inspector-field').filter({ hasText: 'Return state' }).locator('input')).toBeDisabled()
    await page.screenshot({ path: path.join(evidenceDir, 'profile-parameter-incompatible-builder.png'), fullPage: true })

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
    await expect(page.getByText(/Parameter 'Return state' is not supported for Native PyTorch/i)).toBeVisible()
    await page.screenshot({ path: path.join(evidenceDir, 'invalid-compilation.png'), fullPage: true })

    await page.goto(`/models/${fixture.id}/export`)
    await expect(page.getByRole('button', { name: /Project bundle/ })).toBeDisabled()
    await expect(page.getByRole('button', { name: /Project bundle/ })).toContainText('Fix the validation diagnostics')
    const pytorchExport = page.getByRole('button', { name: /PyTorch ExportedProgram/ })
    await expect(pytorchExport).toBeDisabled()
    await expect(pytorchExport).toHaveAttribute('title', /(?:Unavailable|Native runtime|profile|probed)/i)
    await expect(pytorchExport).toContainText(/(?:Unavailable for this profile|Native runtime is unavailable|capability has not been probed)/i)
    await page.screenshot({ path: path.join(evidenceDir, 'invalid-export-capabilities.png'), fullPage: true })

    await page.goto(`/builder/${fixture.id}`)
    await expect(page.getByLabel('Enable Keras', { exact: true })).not.toBeChecked()
    await page.getByLabel('Enable Keras', { exact: true }).check()
    await expect(page.locator('[data-node-id="simple-rnn-compat-e2e"]')).not.toHaveClass(/has-validation-error/)
  } finally {
    await deleteProject(request, fixture.id)
  }
})

async function pollsCompatibleFunctionalValidation({ page, request }: WorkflowFixtures): Promise<void> {
  const fixture = await createProject(request, `Angular compatibility smoke ${Date.now()}`)
  const evidenceDir = process.env.NW_VALIDATION_EVIDENCE_DIR
    ? campaignEvidenceDir
    : path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'backend-compatibility-consolidation')
  await mkdir(evidenceDir, { recursive: true })
  try {
    await page.goto(`/models/${fixture.id}/validation`)
    await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
    await expect(page.getByText(/keras · pytorch ·/i)).toBeVisible({ timeout: 75_000 })
    const submitResponse = page.waitForResponse((response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
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
}

async function restoresFunctionalValidationAfterReload({ page, request }: WorkflowFixtures): Promise<void> {
  const fixture = await createProject(request, `NW-T2-04 validation reload ${Date.now()}`)
  const evidenceDir = process.env.NW_VALIDATION_EVIDENCE_DIR
    ? campaignEvidenceDir
    : path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260923', 'evidence')
  await mkdir(evidenceDir, { recursive: true })
  try {
    await page.goto(`/models/${fixture.id}/validation`)
    await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
    await expect(page.getByText(/temporary optimizer step in worker memory/i)).toBeVisible()
    await expect(page.getByText(/does not train or change the saved model/i)).toBeVisible()
    await expect(page.getByText(/keras · pytorch ·/i)).toBeVisible({ timeout: 75_000 })

    const firstSubmit = page.waitForResponse((response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
    const firstJob = await (await firstSubmit).json() as { id: string }
    await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 75_000 })
    const firstResult = await (await request.get(`/api/execution/jobs/${firstJob.id}`)).json() as {
      state: string
      worker: { api: string; backend: string }
      result: { status: string; inputs: unknown[]; outputs: unknown[]; parameter_count: number }
    }
    expect(firstResult).toMatchObject({
      state: 'succeeded',
      worker: { api: 'keras', backend: 'pytorch' },
      result: { status: 'succeeded', inputs: expect.any(Array), outputs: expect.any(Array), parameter_count: expect.any(Number) },
    })

    await page.reload()
    await expect(page.getByText('succeeded', { exact: true })).toBeVisible({ timeout: 30_000 })
    await expect(page.getByText(/Completed on/)).toBeVisible()

    const secondSubmit = page.waitForResponse((response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
    const secondJob = await (await secondSubmit).json() as { id: string }
    await page.reload()

    await expect.poll(async () => {
      const response = await request.get(`/api/execution/jobs/${secondJob.id}`)
      return response.ok() ? String((await response.json() as { state: string }).state) : 'missing'
    }, { timeout: 30_000 }).not.toBe('missing')
    let secondResult = await (await request.get(`/api/execution/jobs/${secondJob.id}`)).json() as {
      state: string
      worker: { api: string; backend: string }
      cancellation_requested: boolean
    }
    expect(secondResult.worker).toMatchObject({ api: 'keras', backend: 'pytorch' })
    if (!['succeeded', 'terminated', 'failed'].includes(secondResult.state)) {
      await expect(page.getByRole('button', { name: 'Stop', exact: true })).toBeVisible({ timeout: 30_000 })
      const stopResponse = page.waitForResponse((response) => response.url().includes(`/api/execution/jobs/${secondJob.id}/stop`) && response.request().method() === 'POST')
      await page.getByRole('button', { name: 'Stop', exact: true }).click()
      expect((await stopResponse).ok()).toBeTruthy()
      await expect(page.getByText('terminated', { exact: true })).toBeVisible({ timeout: 30_000 })
      secondResult = await (await request.get(`/api/execution/jobs/${secondJob.id}`)).json() as typeof secondResult
      expect(secondResult.state).toBe('terminated')
      expect(secondResult.cancellation_requested).toBeTruthy()
    } else {
      expect(secondResult.state).toBe('succeeded')
      await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 30_000 })
    }
    await page.screenshot({ path: path.join(evidenceDir, 'nw-t2-04-validation-restored-after-reload.png'), fullPage: true })
  } finally {
    await deleteProject(request, fixture.id)
  }
}

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
    await waitForCanvasFit(page)
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
    await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 75_000 })
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

test('NW-T2-01 verifies Builder selection, editing, history, save/reopen, and invalid-draft recovery', async ({ page, request }) => {
  test.setTimeout(120_000)
  const diagnostics = {
    pageErrors: [] as string[],
    consoleErrors: [] as string[],
    failedRequests: [] as string[],
    httpErrors: [] as Array<{ method: string; path: string; status: number }>,
  }
  const projectWrites: string[] = []
  page.on('pageerror', (error) => diagnostics.pageErrors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') diagnostics.consoleErrors.push(message.text())
  })
  page.on('requestfailed', (request) => {
    diagnostics.failedRequests.push(`${request.method()} ${request.url()}: ${request.failure()?.errorText ?? 'failed'}`)
  })
  page.on('request', (request) => {
    const pathname = new URL(request.url()).pathname
    if (request.method() === 'PUT' && pathname.startsWith('/api/projects/')) projectWrites.push(pathname)
  })
  page.on('response', (response) => {
    if (response.status() >= 400) {
      diagnostics.httpErrors.push({
        method: response.request().method(),
        path: new URL(response.url()).pathname,
        status: response.status(),
      })
    }
  })

  await mkdir(campaignEvidenceDir, { recursive: true })
  const fixture = await createProject(request, `NW T2-01 Builder ${Date.now()}`)
  const projectUrl = `/api/projects/${fixture.id}`
  const hiddenLayer = page.locator('.nw-flow-node').filter({ hasText: 'Hidden layer' })
  const inspector = page.getByRole('complementary', { name: 'Properties inspector' })

  try {
    await page.goto(`/builder/${fixture.id}`)
    await expect(page.locator('.nw-flow-node')).toHaveCount(4)
    await expect(page.locator('.builder-badge')).toContainText('Saved')

    await hiddenLayer.click()
    await expect(inspector.getByRole('heading', { name: 'Hidden layer', exact: true })).toBeVisible()
    const units = page.getByLabel('Units')
    await expect(units).toHaveValue('16')

    await page.locator('.nw-flow-node').filter({ hasText: 'Classifier' }).click()
    await expect(inspector.getByRole('heading', { name: 'Classifier', exact: true })).toBeVisible()
    await hiddenLayer.click()
    await page.getByLabel('Units').fill('32')
    await expect(page.getByLabel('Units')).toHaveValue('32')
    await expect(page.locator('.builder-badge')).toContainText('Drafting')

    await page.getByRole('button', { name: 'Undo', exact: true }).click()
    await expect(page.getByLabel('Units')).toHaveValue('16')
    await expect(page.locator('.builder-badge')).toContainText('Saved')
    await page.getByRole('button', { name: 'Redo', exact: true }).click()
    await expect(page.getByLabel('Units')).toHaveValue('32')
    await expect(page.locator('.builder-badge')).toContainText('Drafting')

    await page.getByLabel('Units').fill('24')
    await expect(page.locator('.builder-badge')).toContainText('Drafting')
    await hiddenLayer.click({ button: 'right' })
    const resetAction = page.getByRole('menuitem', { name: 'Reset parameters', exact: true })
    await expect(resetAction).toBeEnabled()
    await resetAction.click()
    await expect(page.getByLabel('Units')).toHaveValue('32')
    await expect(page.locator('.builder-badge')).toContainText('Drafting')

    await page.getByRole('button', { name: 'Undo', exact: true }).click()
    await expect(page.getByLabel('Units')).toHaveValue('24')
    await page.getByRole('button', { name: 'Redo', exact: true }).click()
    await expect(page.getByLabel('Units')).toHaveValue('32')
    await page.screenshot({ path: path.join(campaignEvidenceDir, 'nw-t2-01-reset-and-history.png'), fullPage: true })

    const firstSave = page.waitForResponse((response) => {
      const url = new URL(response.url())
      return url.pathname === projectUrl && response.request().method() === 'PUT'
    })
    await page.getByRole('button', { name: 'Save model', exact: true }).click()
    expect((await firstSave).ok()).toBeTruthy()
    await expect(page.locator('.builder-badge')).toContainText('Saved')

    const saved = await (await request.get(projectUrl)).json() as {
      current_revision: number
      graph: { nodes: Array<{ id: string; parameters: Record<string, unknown> }> }
    }
    expect(saved.current_revision).toBe(fixture.current_revision + 1)
    expect(saved.graph.nodes.find((node) => node.id === 'layer-dense-e2e-1')?.parameters['units']).toBe(32)

    await page.reload()
    await hiddenLayer.click()
    await expect(page.getByLabel('Units')).toHaveValue('32')
    await expect(page.locator('.builder-badge')).toContainText('Saved')

    await page.getByLabel('Units').fill('1.5')
    await expect(page.getByLabel('Units')).toHaveAttribute('aria-invalid', 'true')
    await expect(page.getByText('Enter a valid units.', { exact: true })).toBeVisible()
    await expect(page.getByLabel('Units')).toBeEnabled()
    await expect(page.getByLabel('Units')).toHaveValue('1.5')
    await expect(page.getByRole('button', { name: 'Save model', exact: true })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Validate', exact: true })).toBeDisabled()
    const writesBeforeRepair = projectWrites.length
    const invalidDraftSnapshot = await (await request.get(projectUrl)).json() as {
      current_revision: number
      graph: { nodes: Array<{ id: string; parameters: Record<string, unknown> }> }
    }
    expect(invalidDraftSnapshot.current_revision).toBe(saved.current_revision)
    expect(invalidDraftSnapshot.graph.nodes.find((node) => node.id === 'layer-dense-e2e-1')?.parameters['units']).toBe(32)
    expect(projectWrites).toHaveLength(writesBeforeRepair)
    await page.screenshot({ path: path.join(campaignEvidenceDir, 'nw-t2-01-invalid-draft-repairable.png'), fullPage: true })

    await page.getByLabel('Units').fill('24')
    await expect(page.getByLabel('Units')).not.toHaveAttribute('aria-invalid', 'true')
    await expect(page.locator('.builder-badge')).toContainText('Drafting')
    const repairSave = page.waitForResponse((response) => {
      const url = new URL(response.url())
      return url.pathname === projectUrl && response.request().method() === 'PUT'
    })
    await page.getByRole('button', { name: 'Save model', exact: true }).click()
    expect((await repairSave).ok()).toBeTruthy()
    await expect(page.locator('.builder-badge')).toContainText('Saved')

    const repaired = await (await request.get(projectUrl)).json() as {
      current_revision: number
      graph: { nodes: Array<{ id: string; parameters: Record<string, unknown> }> }
    }
    expect(repaired.current_revision).toBe(saved.current_revision + 1)
    expect(repaired.graph.nodes.find((node) => node.id === 'layer-dense-e2e-1')?.parameters['units']).toBe(24)
    await page.screenshot({ path: path.join(campaignEvidenceDir, 'nw-t2-01-repaired-saved.png'), fullPage: true })

    await page.reload()
    await hiddenLayer.click()
    await expect(page.getByLabel('Units')).toHaveValue('24')
    await expect(page.locator('.builder-badge')).toContainText('Saved')
    expect(diagnostics).toEqual({ pageErrors: [], consoleErrors: [], failedRequests: [], httpErrors: [] })
    await writeFile(path.join(campaignEvidenceDir, 'nw-t2-01-browser-diagnostics.json'), `${JSON.stringify({
      recorded_at: new Date().toISOString(),
      page_url: page.url(),
      browser_version: page.context().browser()?.version() ?? 'unknown',
      project_write_count: projectWrites.length,
      ...diagnostics,
    }, null, 2)}\n`)
  } finally {
    await deleteProject(request, fixture.id)
    expect((await request.get(projectUrl)).status()).toBe(404)
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
    const configure = page.getByRole('button', { name: 'Configure', exact: true })
    await expect(configure).toBeEnabled({ timeout: 75_000 })
    await configure.click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible({ timeout: 75_000 })
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

    await page.goto(`/models/${fixture.id}/validation`)
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

test('NW-T2-03: rechecks compiler capability and invalidates source when the saved profile changes', async ({ page, request }) => {
  test.setTimeout(120_000)
  const fixture = await createProject(request, `JAX compilation ${Date.now()}`)
  const evidenceDir = campaignEvidenceDir
  await mkdir(evidenceDir, { recursive: true })
  try {
    await page.goto(`/models/${fixture.id}/compilation`)
    await page.getByRole('button', { name: 'Validate profile', exact: true }).click()
    await expect(page.locator('.validation-panel.valid')).toBeVisible()
    const kerasSourceResponse = page.waitForResponse((response) => response.url().includes(`/api/exports/${fixture.id}/source`) && response.request().method() === 'GET')
    await page.getByRole('button', { name: 'Compile source', exact: true }).click()
    expect((await kerasSourceResponse).ok()).toBeTruthy()
    await expect(page.locator('.generated-code code')).toContainText('keras')

    await page.goto(`/builder/${fixture.id}`)
    const backend = page.getByRole('combobox', { name: 'Execution backend' })
    await backend.click()
    await page.getByRole('option', { name: 'JAX / Flax', exact: true }).click()
    await page.getByLabel('Enable Keras', { exact: true }).uncheck()
    const profileResponse = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT')
    await page.getByRole('button', { name: 'Save model', exact: true }).click()
    const savedProfile = await (await profileResponse).json() as { current_revision: number; execution_backend: string; keras_enabled: boolean; generated_source?: string | null }
    expect(savedProfile).toMatchObject({ execution_backend: 'jax', keras_enabled: false, generated_source: null })

    await page.goto(`/models/${fixture.id}/compilation`)
    await expect(page.locator('.generated-code-placeholder')).toContainText('No generated source')
    await page.getByRole('button', { name: 'Validate profile', exact: true }).click()
    await expect(page.locator('.validation-panel.valid')).toBeVisible()
    await page.getByRole('button', { name: 'Configure', exact: true }).click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    await dialog.getByRole('tab', { name: /Compiler/ }).click()
    const jitToggle = dialog.locator('.compilation-parameter-grid input[type="checkbox"]')
    await expect(jitToggle).toHaveCount(1)
    const metadataResponse = await request.get('/api/metadata')
    expect(metadataResponse.ok()).toBeTruthy()
    const metadata = await metadataResponse.json() as { configuration: { compilation_capabilities: { native: { jax: { available: boolean } } } } }
    if (!metadata.configuration.compilation_capabilities.native.jax.available) {
      await expect(jitToggle).toBeDisabled()
      await expect(dialog.locator('.compilation-unavailable')).toContainText(/not operational|not report the compiler/i)
      await page.screenshot({ path: path.join(evidenceDir, 'nw-t2-03-jax-compiler-unavailable.png'), fullPage: true })
      return
    }

    await expect(jitToggle).toBeEnabled()
    await jitToggle.check()
    await dialog.getByRole('button', { name: 'Apply configuration', exact: true }).click()
    await expect(page.getByRole('button', { name: /Saved validated compilation settings/ })).toBeVisible()
    const jaxSourceResponse = page.waitForResponse((response) => response.url().includes(`/api/exports/${fixture.id}/source`) && response.request().method() === 'GET')
    await page.getByRole('button', { name: 'Compile source', exact: true }).click()
    expect((await jaxSourceResponse).ok()).toBeTruthy()
    await expect(page.getByRole('heading', { name: `Python · revision ${savedProfile.current_revision + 1}` })).toBeVisible()
    await expect(page.locator('.generated-code code')).toContainText('return jax.jit(model.apply)')
    await page.locator('.generated-code').screenshot({ path: path.join(evidenceDir, 'nw-t2-03-jax-source-revision.png') })
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('NW-T4-02: keeps native compiler controls aligned with live capability gates', async ({ page, request }) => {
  test.setTimeout(120_000)
  const fixture = await createProject(request, `Native compiler gates ${Date.now()}`)
  const evidenceDir = campaignEvidenceDir
  await mkdir(evidenceDir, { recursive: true })
  try {
    await page.goto(`/builder/${fixture.id}`)
    await page.getByLabel('Enable Keras', { exact: true }).uncheck()
    const saveNativeProfile = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT')
    await page.getByRole('button', { name: 'Save model', exact: true }).click()
    expect((await saveNativeProfile).ok()).toBeTruthy()

    const verifyCompilerControl = async (backend: string, parameter: string, evidenceName: string): Promise<void> => {
      await page.goto(`/models/${fixture.id}/compilation`)
      const configure = page.getByRole('button', { name: 'Configure', exact: true })
      await expect(configure).toBeEnabled({ timeout: 75_000 })
      await configure.click()
      const dialog = page.getByRole('dialog')
      await expect(dialog).toBeVisible({ timeout: 75_000 })
      await dialog.getByRole('tab', { name: /Compiler/ }).click()
      const toggle = dialog.locator('.compilation-parameter-grid label').filter({ hasText: parameter }).locator('input[type="checkbox"]')
      await expect(toggle).toHaveCount(1)
      const response = await request.get('/api/metadata')
      expect(response.ok()).toBeTruthy()
      const metadata = await response.json() as { configuration: { compilation_capabilities: { native: Record<string, { available: boolean }> } } }
      const capability = metadata.configuration.compilation_capabilities.native[backend]
      if (capability.available) await expect(toggle).toBeEnabled()
      else {
        await expect(toggle).toBeDisabled()
        await expect(dialog.locator('.compilation-unavailable')).toBeVisible()
      }
      await page.screenshot({ path: path.join(evidenceDir, evidenceName), fullPage: true })
      await dialog.getByRole('button', { name: 'Cancel', exact: true }).click()
    }

    await verifyCompilerControl('pytorch', 'Enable torch.compile', 'nw-t4-02-pytorch-compiler-gate.png')

    await page.goto(`/builder/${fixture.id}`)
    const backend = page.getByRole('combobox', { name: 'Execution backend' })
    await backend.click()
    await page.getByRole('option', { name: 'TensorFlow', exact: true }).click()
    const saveTensorFlowProfile = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT')
    await page.getByRole('button', { name: 'Save model', exact: true }).click()
    expect((await saveTensorFlowProfile).ok()).toBeTruthy()
    await verifyCompilerControl('tensorflow', 'XLA JIT compilation', 'nw-t4-02-tensorflow-xla-gate.png')
  } finally {
    await deleteProject(request, fixture.id)
  }
})

async function inspectsSequenceTensorsAndDeletesSessions({ page, request }: WorkflowFixtures): Promise<void> {
  const fixture = await createProject(request, `Tensor inspection ${Date.now()}`, sequenceInspectionGraph())
  const initialProject = await (await request.get(`/api/projects/${fixture.id}`)).json() as { current_revision: number; graph: unknown }
  const evidenceDir = campaignEvidenceDir
  const sessionIds: string[] = []
  await mkdir(evidenceDir, { recursive: true })

  const prepareRoute = async (route: string): Promise<void> => {
    await page.goto(route)
    if (!route.startsWith('/builder/')) return
    const validationResponse = page.waitForResponse((response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Validate', exact: true }).click()
    expect((await validationResponse).ok()).toBeTruthy()
    await expect(page.locator('.validation-panel')).toHaveAttribute('data-validation-state', 'ready')
    await expect(page.locator('.validation-panel .status-indicator.valid')).toBeVisible()
  }

  const inspectFrom = async (route: string, saveScreenshot: boolean): Promise<unknown> => {
    await prepareRoute(route)
    const node = page.locator('.nw-flow-node').filter({ hasText: 'Sequence encoder' })
    await expect(node).toBeVisible()
    await node.click({ button: 'right' })
    const inspectAction = page.getByRole('menuitem', { name: 'Inspect tensor', exact: true })
    await expect(inspectAction).toBeEnabled()

    const jobRequestPromise = page.waitForRequest((candidate) => candidate.url().includes('/api/execution/jobs') && candidate.method() === 'POST')
    const sessionResponsePromise = page.waitForResponse((response) => {
      const pathname = new URL(response.url()).pathname
      return /^\/api\/tensor-inspections\/[a-f0-9]{32}$/.test(pathname) && response.request().method() === 'GET'
    })
    await inspectAction.click()
    const jobRequest = await jobRequestPromise
    const jobPayload = jobRequest.postDataJSON() as { operation: { type: string; project_revision: number; seed: number }; worker: { api: string; backend: string } }
    expect(jobPayload.operation).toMatchObject({ type: 'tensor_inspection', project_revision: fixture.current_revision, seed: 42 })
    expect(jobPayload.worker).toMatchObject({ api: 'keras', backend: 'pytorch' })
    const staleRevisionResponse = await request.post('/api/execution/jobs', {
      data: { ...jobPayload, operation: { ...jobPayload.operation, project_revision: fixture.current_revision + 1 } },
    })
    expect(staleRevisionResponse.status()).toBe(422)
    expect(await staleRevisionResponse.json()).toMatchObject({ detail: { code: 'revision_conflict' } })

    const sessionResponse = await sessionResponsePromise
    expect(sessionResponse.ok()).toBeTruthy()
    const session = await sessionResponse.json() as {
      id: string
      project_revision: number
      runtime_api: string
      execution_backend: string
      seed: number
      weight_source: string
      compiler_mode: string
      nodes: Array<{ node_id: string; inspectable: boolean; predecessors: string[]; successors: string[]; outputs: Array<{ id: string; port_id: string; shape: number[] }> }>
    }
    sessionIds.push(session.id)
    expect(session).toMatchObject({
      project_revision: fixture.current_revision,
      runtime_api: 'keras',
      execution_backend: 'pytorch',
      seed: 42,
      weight_source: 'initialized_untrained',
      compiler_mode: 'normal',
    })

    const modal = page.getByRole('dialog')
    await expect(modal).toBeVisible()
    await expect(modal.locator('.tensor-inspector-toolbar select')).toHaveValue('inspection-rnn')
    await expect(modal.getByRole('heading', { name: 'Sequence encoder', exact: true })).toBeVisible()
    await expect(modal.getByRole('note')).toContainText('Synthetic input · untrained initialized weights · not trained activations')
    const recurrent = session.nodes.find((item) => item.node_id === 'inspection-rnn')
    expect(recurrent).toBeDefined()
    expect(recurrent).toMatchObject({ inspectable: true, predecessors: ['inspection-sequence-input'], successors: ['inspection-sequence-output'] })
    expect(recurrent!.outputs.map((tensor) => tensor.port_id).sort()).toEqual(['hidden_state', 'output'])

    const sequenceOutput = recurrent!.outputs.find((tensor) => tensor.port_id === 'output')!
    expect(sequenceOutput.shape).toEqual([1, 3, 4])
    const tensorSelect = modal.locator('.tensor-inspection-ports select')
    await tensorSelect.selectOption(sequenceOutput.id)
    await expect(modal.locator('.tensor-detail-list')).toContainText('[1, 3, 4]')
    const displayAxes = modal.locator('.tensor-axis-controls label select')
    await expect(displayAxes.nth(0)).toHaveValue('1')
    await expect(displayAxes.nth(1)).toHaveValue('2')
    const viewResponsePromise = page.waitForResponse((response) =>
      new URL(response.url()).pathname === `/api/tensor-inspections/${session.id}/view`
      && response.request().method() === 'POST'
      && (response.request().postDataJSON() as { tensor_id?: string; limit?: number } | null)?.tensor_id === sequenceOutput.id
      && (response.request().postDataJSON() as { tensor_id?: string; limit?: number } | null)?.limit === 4096,
      { timeout: 15_000 },
    )
    await modal.getByRole('button', { name: 'Heatmap', exact: true }).click()
    const viewResponse = await viewResponsePromise
    expect(viewResponse.ok()).toBeTruthy()
    const viewRequest = viewResponse.request().postDataJSON() as { tensor_id: string; display_axes: number[]; fixed_indices: Record<string, number>; limit: number }
    expect(viewRequest).toMatchObject({ tensor_id: sequenceOutput.id, display_axes: [1, 2], fixed_indices: {}, limit: 4096 })
    const sequenceView = await viewResponse.json() as { shape: number[]; display_axes: number[]; fixed_indices: Record<string, number>; view_shape: number[]; mode: string; values: unknown }
    expect(sequenceView).toMatchObject({ shape: [1, 3, 4], display_axes: [1, 2], fixed_indices: {}, view_shape: [3, 4], mode: 'exact' })
    await expect(modal.locator('.tensor-heatmap span')).toHaveCount(12)
    const heatmapGeometry = await modal.locator('.tensor-heatmap span').evaluateAll((cells) => {
      const positions = cells.map((cell) => {
        const rect = cell.getBoundingClientRect()
        return { top: Math.round(rect.top), left: Math.round(rect.left) }
      })
      return {
        rows: new Set(positions.map((position) => position.top)).size,
        columns: new Set(positions.map((position) => position.left)).size,
      }
    })
    expect(heatmapGeometry).toEqual({ rows: 3, columns: 4 })
    if (saveScreenshot) await page.screenshot({ path: path.join(evidenceDir, 'nw-t2-05-builder-sequence-heatmap.png'), fullPage: false })

    await modal.getByRole('button', { name: 'Statistics', exact: true }).click()
    await expect(modal.locator('.tensor-stat-grid')).toBeVisible()
    await modal.getByRole('button', { name: 'Histogram', exact: true }).click()
    await expect(modal.locator('.tensor-histogram span')).toHaveCount(64)
    const hiddenState = recurrent!.outputs.find((tensor) => tensor.port_id === 'hidden_state')!
    await tensorSelect.selectOption(hiddenState.id)
    await expect(modal.locator('.tensor-detail-list')).toContainText('[1, 4]')

    const deleteResponsePromise = page.waitForResponse((response) =>
      new URL(response.url()).pathname === `/api/tensor-inspections/${session.id}` && response.request().method() === 'DELETE',
    )
    await modal.getByRole('button', { name: 'Close tensor inspection' }).click()
    const deleteResponse = await deleteResponsePromise
    expect(deleteResponse.status()).toBe(204)
    await expect(modal).toBeHidden()
    expect((await request.get(`/api/tensor-inspections/${session.id}`)).status()).toBe(404)
    return sequenceView.values
  }

  try {
    const builderValues = await inspectFrom(`/builder/${fixture.id}`, true)
    const overviewValues = await inspectFrom(`/models/${fixture.id}`, false)
    expect(overviewValues).toEqual(builderValues)

    await prepareRoute(`/builder/${fixture.id}`)
    const cancelNode = page.locator('.nw-flow-node').filter({ hasText: 'Sequence encoder' })
    await cancelNode.click({ button: 'right' })
    const cancelInspect = page.getByRole('menuitem', { name: 'Inspect tensor', exact: true })
    const submittedJobResponse = page.waitForResponse((response) =>
      new URL(response.url()).pathname === '/api/execution/jobs' && response.request().method() === 'POST',
    )
    await cancelInspect.click()
    const submittedJob = await (await submittedJobResponse).json() as { id: string; operation: string; state: string }
    expect(submittedJob.operation).toBe('tensor_inspection')
    const stopResponsePromise = page.waitForResponse((response) =>
      new URL(response.url()).pathname === `/api/execution/jobs/${submittedJob.id}/stop` && response.request().method() === 'POST',
    )
    await page.getByRole('dialog').getByRole('button', { name: 'Close tensor inspection' }).click()
    const stopResponse = await stopResponsePromise
    expect(stopResponse.ok()).toBeTruthy()
    let finalJob: { state: string; cancellation_requested: boolean; result?: Record<string, unknown> | null } | null = null
    await expect.poll(async () => {
      const current = await request.get(`/api/execution/jobs/${submittedJob.id}`)
      finalJob = await current.json() as typeof finalJob
      return finalJob?.state
    }, { timeout: 45_000 }).toBe('terminated')
    expect(finalJob?.cancellation_requested).toBeTruthy()
    expect(finalJob?.result?.['session_id']).toBeUndefined()

    const after = await (await request.get(`/api/projects/${fixture.id}`)).json() as {
      current_revision: number
      graph: unknown
    }
    expect(after.current_revision).toBe(initialProject.current_revision)
    expect(after.graph).toEqual(initialProject.graph)
  } finally {
    for (const sessionId of sessionIds) {
      await request.delete(`/api/tensor-inspections/${sessionId}`).catch(() => undefined)
    }
    await deleteProject(request, fixture.id)
  }
}

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

test.describe('workflow runtime jobs', () => {
  // RuntimeController intentionally leases one CPU device per backend process.
  // Keep job-bearing workflows serial while leaving fixture-only workflows parallel.
  test.describe.configure({ mode: 'serial' })

  test('polls a compatible functional-validation job to a verified result', async ({ page, request }) => {
    test.setTimeout(120_000)
    await pollsCompatibleFunctionalValidation({ page, request })
  })

  test('NW-T2-04 restores functional-validation results and active jobs after reload', async ({ page, request }) => {
    test.setTimeout(120_000)
    await restoresFunctionalValidationAfterReload({ page, request })
  })

  test('NW-T2-05: inspects sequence tensors from Builder and overview, then deletes each session', async ({ page, request }) => {
    test.setTimeout(180_000)
    await inspectsSequenceTensorsAndDeletesSessions({ page, request })
  })
})
