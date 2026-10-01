// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import fs from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

const evidenceDir = path.resolve(
  process.env.NW_VALIDATION_EVIDENCE_DIR
    ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'ui-refinement-20260909'),
)

type CreatedProject = { id: string; name: string }

function graph() {
  return {
    schema_version: 5,
    nodes: [
      {
        id: 'input-tabular-refinement',
        definition_id: 'input.tabular',
        definition_version: 1,
        label: 'Features',
        position: { x: 48, y: 144 },
        parameters: { shape: [8], dtype: 'float32' },
      },
      {
        id: 'layer-dense-refinement-1',
        definition_id: 'layer.dense',
        definition_version: 1,
        label: 'Hidden layer',
        position: { x: 352, y: 144 },
        parameters: { units: 16, activation: 'relu', use_bias: true },
      },
      {
        id: 'layer-dense-refinement-2',
        definition_id: 'layer.dense',
        definition_version: 1,
        label: 'Classifier',
        position: { x: 656, y: 144 },
        parameters: { units: 1, activation: 'sigmoid', use_bias: true },
      },
      {
        id: 'output-classification-refinement',
        definition_id: 'output.classification',
        definition_version: 1,
        label: 'Prediction',
        position: { x: 960, y: 144 },
        parameters: {},
      },
    ],
    edges: [
      { id: 'edge-refinement-1', source_node_id: 'input-tabular-refinement', source_port_id: 'output', target_node_id: 'layer-dense-refinement-1', target_port_id: 'input' },
      { id: 'edge-refinement-2', source_node_id: 'layer-dense-refinement-1', source_port_id: 'output', target_node_id: 'layer-dense-refinement-2', target_port_id: 'input' },
      { id: 'edge-refinement-3', source_node_id: 'layer-dense-refinement-2', source_port_id: 'output', target_node_id: 'output-classification-refinement', target_port_id: 'input' },
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
        pytorch: { optimizer: 'adamw', optimizer_parameters: { learning_rate: 0.0005, weight_decay: 0.004, beta_1: 0.9, beta_2: 0.999, epsilon: 1e-7 }, losses: { __default__: 'binary_crossentropy' }, metrics: { __default__: ['binary_accuracy', 'auc'] }, compile_enabled: false, mode: 'default', fullgraph: false, dynamic: false },
        jax: { jit_enabled: false },
      },
    },
    export_settings: { include_comments: true, include_shape_assertions: true, preferred_binary_format: 'keras' },
    future_training_configuration: {},
  }
}

async function createProject(request: APIRequestContext, name: string): Promise<CreatedProject> {
  const response = await request.post('/api/projects', { data: { name, description: 'UI refinement browser fixture', execution_backend: 'pytorch', keras_enabled: true, graph: graph() } })
  expect(response.ok()).toBeTruthy()
  return response.json() as Promise<CreatedProject>
}

async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
  const response = await request.delete(`/api/projects/${id}`)
  expect([204, 404]).toContain(response.status())
}

async function openNodeMenu(page: Page, label: string): Promise<ReturnType<Page['locator']>> {
  await page.locator('.nw-flow-node').filter({ hasText: label }).first().click({ button: 'right', force: true })
  const menu = page.locator('.node-context-menu')
  await expect(menu).toBeVisible()
  return menu
}

test.describe.configure({ mode: 'serial' })

test('keeps the Nodes workspace contained and the catalog controls coherent', async ({ page }) => {
  test.setTimeout(90_000)
  await fs.mkdir(evidenceDir, { recursive: true })
  const viewports = [
    { width: 1440, height: 900 },
    { width: 1280, height: 720 },
    { width: 1024, height: 768 },
    { width: 900, height: 700 },
    { width: 560, height: 700 },
  ]
  const viewportEvidence: Array<{ viewport: { width: number; height: number }; geometry: unknown; scroll: unknown }> = []

  for (const viewport of viewports) {
    await page.setViewportSize(viewport)
    await page.goto('/nodes')
    await expect(page.getByRole('heading', { name: 'Nodes', exact: true })).toBeVisible()
    const geometry = await page.evaluate(() => {
      const content = document.querySelector<HTMLElement>('.app-content')
      const workspace = document.querySelector<HTMLElement>('.library-workspace')
      const list = document.querySelector<HTMLElement>('.library-list')
      const details = document.querySelector<HTMLElement>('.detail-body')
      const controls = document.querySelector<HTMLElement>('.library-toolbar')
      const cards = document.querySelector<HTMLElement>('.model-card-grid')
      const bounds = (element: HTMLElement | null) => element?.getBoundingClientRect()
      const workspaceBounds = bounds(workspace)
      const contentBounds = bounds(content)
      return {
        documentStable: document.documentElement.scrollHeight <= document.documentElement.clientHeight + 1,
        contentStable: Boolean(content && content.scrollHeight <= content.clientHeight + 1),
        workspaceContained: Boolean(workspaceBounds && contentBounds && workspaceBounds.bottom <= contentBounds.bottom + 1),
        listScrollable: Boolean(list && list.scrollHeight >= list.clientHeight),
        detailsScrollable: Boolean(details && details.scrollHeight >= details.clientHeight),
        toolbarHeight: bounds(controls)?.height ?? 0,
        cards: Boolean(cards),
        columns: getComputedStyle(workspace!).gridTemplateColumns,
      }
    })
    expect(geometry.documentStable).toBeTruthy()
    expect(geometry.contentStable).toBeTruthy()
    expect(geometry.workspaceContained).toBeTruthy()
    expect(geometry.toolbarHeight).toBeLessThan(viewport.width <= 760 ? 130 : 70)
    expect(geometry.cards).toBeFalsy()
    await page.screenshot({ path: path.join(evidenceDir, `nodes-${viewport.width}x${viewport.height}.png`), fullPage: false })

    const search = page.getByRole('searchbox', { name: 'Search node library' })
    const filters = page.locator('.origin-filter')
    await filters.getByRole('button', { name: 'Built-in', exact: true }).click()
    await expect(filters.getByRole('button', { name: 'Built-in', exact: true })).toHaveClass(/active/)
    await search.fill('pool')
    await expect(page.locator('.library-row')).not.toHaveCount(0)
    expect((await page.locator('.library-row .origin-badge').allTextContents()).every((text) => text.trim() === 'Built-in')).toBeTruthy()
    await filters.getByRole('button', { name: 'Custom', exact: true }).click()
    await expect(filters.getByRole('button', { name: 'Custom', exact: true })).toHaveClass(/active/)
    await expect(page.locator('.library-row')).toHaveCount(0)
    await filters.getByRole('button', { name: 'All', exact: true }).click()
    await search.fill('Average Pooling 3D')
    const poolingRow = page.locator('.library-row').filter({ hasText: 'Average Pooling 3D' }).first()
    await expect(poolingRow).toBeVisible()
    await poolingRow.click()
    await expect(page.locator('.parameter-line')).not.toHaveCount(0)
    const fieldGeometry = await page.locator('.parameter-line').first().evaluate((row) => {
      const children = [...row.children].map((child) => child.getBoundingClientRect())
      return { overlap: children.some((left, index) => children.slice(index + 1).some((right) => left.right > right.left + 0.5 && right.right > left.left + 0.5 && left.bottom > right.top + 0.5 && right.bottom > left.top + 0.5)), allInside: children.every((child) => child.right <= row.getBoundingClientRect().right + 1) }
    })
    expect(fieldGeometry.overlap).toBeFalsy()
    expect(fieldGeometry.allInside).toBeTruthy()
    await search.fill('')
    await page.waitForFunction(() => document.querySelectorAll('.library-row').length > 20)
    const scrollMetrics = await page.evaluate(() => {
      const content = document.querySelector<HTMLElement>('.app-content')!
      const list = document.querySelector<HTMLElement>('.library-list')!
      const details = document.querySelector<HTMLElement>('.detail-body')!
      const initialOuter = content.scrollTop
      list.scrollTop = list.scrollHeight
      const listMoved = list.scrollTop > 0
      const listPosition = list.scrollTop
      details.scrollTop = details.scrollHeight
      return { listMoved, listPosition, detailsMoved: details.scrollTop > 0, outerStable: content.scrollTop === initialOuter }
    })
    expect(scrollMetrics.listMoved).toBeTruthy()
    expect(scrollMetrics.detailsMoved).toBeTruthy()
    expect(scrollMetrics.outerStable).toBeTruthy()
    viewportEvidence.push({ viewport, geometry, scroll: scrollMetrics })
  }
  await fs.writeFile(path.join(evidenceDir, 'nodes-geometry.json'), JSON.stringify(viewportEvidence, null, 2), 'utf8')
})

test('keeps the New Node wizard structured within the viewport', async ({ page }) => {
  await fs.mkdir(evidenceDir, { recursive: true })
  for (const viewport of [{ width: 900, height: 700 }, { width: 560, height: 700 }]) {
    await page.setViewportSize(viewport)
    await page.goto('/nodes')
    await page.getByRole('button', { name: 'Add custom node', exact: true }).click()
    const dialog = page.locator('.node-wizard')
    await expect(dialog).toBeVisible()
    const bounds = await dialog.evaluate((element) => {
      const rect = element.getBoundingClientRect()
      const footer = element.querySelector<HTMLElement>('.dialog-actions')?.getBoundingClientRect()
      return { right: rect.right, bottom: rect.bottom, width: rect.width, footerBottom: footer?.bottom ?? 0, viewportWidth: innerWidth, viewportHeight: innerHeight, footerSeparated: (footer?.top ?? 0) > rect.top }
    })
    expect(bounds.right).toBeLessThanOrEqual(bounds.viewportWidth + 1)
    expect(bounds.bottom).toBeLessThanOrEqual(bounds.viewportHeight + 1)
    expect(bounds.footerBottom).toBeLessThanOrEqual(bounds.bottom + 1)
    expect(bounds.footerSeparated).toBeTruthy()
    await page.screenshot({ path: path.join(evidenceDir, `new-node-${viewport.width}x${viewport.height}.png`), fullPage: false })
    await dialog.getByRole('button', { name: /Mathematical operation/ }).click()
    await dialog.getByRole('button', { name: 'Continue', exact: true }).click()
    await dialog.getByLabel('Name').fill('Viewport-safe custom node')
    await dialog.getByLabel('Category').fill('custom')
    await dialog.getByLabel('Description').fill('A long enough description to exercise the structured wizard body without making the modal overflow the viewport.')
    await dialog.getByRole('button', { name: 'Continue', exact: true }).click()
    await expect(dialog.locator('.wizard-body')).toBeVisible()
    await dialog.getByRole('button', { name: 'Close custom node wizard' }).click()
  }
})

test('adds the intended small Models toolbar gap without changing card structure', async ({ page, request }) => {
  const fixture = await createProject(request, `UI spacing ${Date.now()}`)
  try {
    await page.goto('/models')
    const card = page.locator('.model-card').filter({ hasText: fixture.name }).first()
    await expect(card).toBeVisible()
    const gap = await page.evaluate(() => {
      const controls = document.querySelector<HTMLElement>('.catalog-controls')?.getBoundingClientRect()
      const cards = document.querySelector<HTMLElement>('.model-card-grid')?.getBoundingClientRect()
      return controls && cards ? cards.top - controls.bottom : 0
    })
    expect(gap).toBeGreaterThanOrEqual(16)
    expect(await page.locator('.model-card').count()).toBeGreaterThan(0)
  } finally {
    await deleteProject(request, fixture.id)
  }
})

test('state-gates canvas and node context menus and transitions explicit outputs', async ({ page, request }) => {
  await page.goto('/builder')
  await page.locator('.canvas-shell').click({ button: 'right', position: { x: 42, y: 42 } })
  const emptyMenu = page.locator('.canvas-context-menu:not(.node-context-menu)')
  await expect(emptyMenu.getByRole('menuitem', { name: 'Fit workflow to view' })).toBeDisabled()
  await expect(emptyMenu.getByRole('menuitem', { name: 'Auto layout' })).toBeDisabled()
  await expect(emptyMenu.getByRole('menuitem', { name: 'Reset canvas' })).toBeDisabled()
  await page.keyboard.press('Escape')

  const fixture = await createProject(request, `UI context menu ${Date.now()}`)
  try {
    await page.goto(`/builder/${fixture.id}`)
    await expect(page.locator('.nw-flow-node')).toHaveCount(4)
    await page.locator('.canvas-shell').click({ button: 'right', position: { x: 42, y: 42 } })
    const populatedMenu = page.locator('.canvas-context-menu:not(.node-context-menu)')
    await expect(populatedMenu.getByRole('menuitem', { name: 'Fit workflow to view' })).toBeEnabled()
    await populatedMenu.getByRole('menuitem', { name: 'Fit workflow to view' }).click()
    await page.locator('.canvas-shell').click({ button: 'right', position: { x: 42, y: 42 } })
    await page.locator('.canvas-context-menu:not(.node-context-menu)').getByRole('menuitem', { name: 'Auto layout' }).click()
    await page.locator('.canvas-shell').click({ button: 'right', position: { x: 42, y: 42 } })
    await page.locator('.canvas-context-menu:not(.node-context-menu)').getByRole('menuitem', { name: 'Reset canvas' }).click()
    const resetDialog = page.getByRole('dialog', { name: 'Reset this canvas?' })
    await expect(resetDialog).toBeVisible()
    await resetDialog.getByRole('button', { name: 'Cancel', exact: true }).click()

    const terminalMenu = await openNodeMenu(page, 'Prediction')
    const outputAction = terminalMenu.getByRole('menuitem', { name: 'Use as output', exact: true })
    await expect(outputAction).toBeDisabled()
    await expect(outputAction).toHaveAttribute('title', /inferred workflow output/)
    await page.keyboard.press('Escape')

    const inputMenu = await openNodeMenu(page, 'Features')
    await expect(inputMenu.getByRole('menuitem', { name: 'Use as output', exact: true })).toBeDisabled()
    await page.keyboard.press('Escape')

    const nodeMenu = await openNodeMenu(page, 'Hidden layer')
    await expect(nodeMenu.getByRole('menuitem', { name: 'Duplicate with defaults', exact: true })).toBeEnabled()
    await expect(nodeMenu.getByRole('menuitem', { name: 'Duplicate exact', exact: true })).toBeEnabled()
    await expect(nodeMenu.getByRole('menuitem', { name: 'Copy node', exact: true })).toBeEnabled()
    await expect(nodeMenu.getByRole('menuitem', { name: 'Reset parameters', exact: true })).toBeEnabled()
    await expect(nodeMenu.getByRole('menuitem', { name: 'Disconnect all', exact: true })).toBeEnabled()
    await expect(nodeMenu.getByRole('menuitem', { name: 'Use as output', exact: true })).toBeEnabled()
    await nodeMenu.getByRole('menuitem', { name: 'Reset parameters', exact: true }).click()

    const disconnectedMenu = await openNodeMenu(page, 'Hidden layer')
    await disconnectedMenu.getByRole('menuitem', { name: 'Disconnect all', exact: true }).click()
    const noConnectionsMenu = await openNodeMenu(page, 'Hidden layer')
    await expect(noConnectionsMenu.getByRole('menuitem', { name: 'Disconnect all', exact: true })).toBeDisabled()
    await noConnectionsMenu.getByRole('menuitem', { name: 'Lock node', exact: true }).click()
    const lockedMenu = await openNodeMenu(page, 'Hidden layer')
    await expect(lockedMenu.getByRole('menuitem', { name: 'Unlock node', exact: true })).toBeEnabled()
    await lockedMenu.getByRole('menuitem', { name: 'Unlock node', exact: true }).click()

    const duplicateBaseline = await page.locator('.nw-flow-node').count()
    const defaultsMenu = await openNodeMenu(page, 'Hidden layer')
    await defaultsMenu.getByRole('menuitem', { name: 'Duplicate with defaults', exact: true }).click()
    await expect(page.locator('.nw-flow-node')).toHaveCount(duplicateBaseline + 1)
    const exactMenu = await openNodeMenu(page, 'Hidden layer')
    await exactMenu.getByRole('menuitem', { name: 'Duplicate exact', exact: true }).click()
    await expect(page.locator('.nw-flow-node')).toHaveCount(duplicateBaseline + 2)
    await openNodeMenu(page, 'Hidden layer').then((menu) => menu.getByRole('menuitem', { name: 'Copy node', exact: true }).click())
    await page.keyboard.press('Control+V')
    await expect(page.locator('.nw-flow-node')).toHaveCount(duplicateBaseline + 3)
    const beforeDelete = await page.locator('.nw-flow-node').count()
    const deleteMenu = await openNodeMenu(page, 'Hidden layer')
    await deleteMenu.getByRole('menuitem', { name: 'Delete node', exact: true }).click()
    await expect(page.locator('.nw-flow-node')).toHaveCount(beforeDelete - 1)

    const upstreamMenu = await openNodeMenu(page, 'Classifier')
    await expect(upstreamMenu.getByRole('menuitem', { name: 'Use as output', exact: true })).toBeEnabled()
    await upstreamMenu.getByRole('menuitem', { name: 'Use as output', exact: true }).click()
    const terminalAfterExplicit = await openNodeMenu(page, 'Prediction')
    await expect(terminalAfterExplicit.getByRole('menuitem', { name: 'Use as output', exact: true })).toBeEnabled()
    await terminalAfterExplicit.getByRole('menuitem', { name: 'Use as output', exact: true }).click()
    const terminalExplicit = await openNodeMenu(page, 'Prediction')
    await expect(terminalExplicit.getByRole('menuitem', { name: 'Remove from outputs', exact: true })).toBeEnabled()
    await terminalExplicit.getByRole('menuitem', { name: 'Remove from outputs', exact: true }).click()
    const upstreamExplicit = await openNodeMenu(page, 'Classifier')
    await upstreamExplicit.getByRole('menuitem', { name: 'Remove from outputs', exact: true }).click()
    const terminalInferredAgain = await openNodeMenu(page, 'Prediction')
    await expect(terminalInferredAgain.getByRole('menuitem', { name: 'Use as output', exact: true })).toBeDisabled()
  } finally {
    await deleteProject(request, fixture.id)
  }
})
