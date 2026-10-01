// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { mkdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

const repoRoot = path.resolve(process.cwd(), '..', '..')
const evidenceRoot = path.resolve(process.env.NW_UI_EVIDENCE_DIR ?? path.join(repoRoot, 'assets', 'QA', 'ui-refinement-20260924'))
const fixtureRoot = path.resolve(process.env.NW_UI_FIXTURE_DIR ?? path.join(repoRoot, 'runtimes', 'cache', 'ui-refinement-20260924', 'e2e-fixtures'))
const captureStage = (process.env.NW_UI_CAPTURE_STAGE ?? 'refined').replace(/[^a-z0-9-]/gi, '-').toLowerCase()

type Route = { name: string; path: (ids: { modelId: string; datasetId: string }) => string; ready: string }
type RouteGeometry = {
  route: string
  viewport: { width: number; height: number }
  header: { x: number; y: number; width: number; height: number }
  content: { x: number; y: number; width: number; height: number; scrollHeight: number; clientHeight: number }
  pageOrigin: { x: number; y: number; width: number } | null
  activeNavigationVisible: boolean
  documentOverflowsHorizontally: boolean
}

async function createProject(request: APIRequestContext): Promise<string> {
  const response = await request.post('/api/projects', {
    data: { name: 'UI consistency route fixture', description: 'Disposable UI route fixture', execution_backend: 'pytorch', keras_enabled: true, graph: {} },
  })
  expect(response.status()).toBe(201)
  return ((await response.json()) as { id: string }).id
}

async function createDataset(request: APIRequestContext, sourcePath: string): Promise<string> {
  const response = await request.post('/api/datasets', {
    data: {
      name: 'UI consistency dataset fixture',
      dataset_type: 'tabular',
      sources: [{ type: 'table', id: 'table', path: sourcePath, format: 'csv', has_header: true, delimiter: ',' }],
      input_definition: { bindings: [{ type: 'columns', source_id: 'table', columns: ['age', 'weight'] }] },
      output_definition: { bindings: [{ type: 'columns', source_id: 'table', columns: ['diagnosis'] }] },
      metadata_definition: { bindings: [] },
      relationships: [],
      sequence_definition: null,
      subset_definition: { type: 'all' },
      split_definition: null,
    },
  })
  expect(response.status()).toBe(201)
  return ((await response.json()) as { id: string }).id
}

async function removeFixture(request: APIRequestContext, endpoint: string, id: string): Promise<void> {
  const response = await request.delete(`${endpoint}/${id}`)
  expect([204, 404]).toContain(response.status())
}

async function ensurePytorchKerasProfile(request: APIRequestContext): Promise<void> {
  const started = await request.post('/api/execution/discovery', {
    data: { backend: 'pytorch', api: 'keras', force_refresh: true },
  })
  expect(started.ok(), 'start exact PyTorch/Keras runtime discovery').toBeTruthy()

  let status: { backend: string; api: string; state: string; available: boolean } | undefined
  const deadline = Date.now() + 90_000
  while (Date.now() < deadline) {
    const response = await request.get('/api/execution/discovery?backend=pytorch&api=keras')
    expect(response.ok(), 'read exact PyTorch/Keras runtime discovery').toBeTruthy()
    status = await response.json() as typeof status
    if (status?.state !== 'checking') break
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  expect(status, 'PyTorch/Keras discovery status').toMatchObject({ backend: 'pytorch', api: 'keras', state: 'verified', available: true })
}

async function measureRoute(page: Page, route: string, viewport: { width: number; height: number }): Promise<RouteGeometry> {
  return page.evaluate(({ route, viewport }) => {
    const bounds = (element: Element | null) => {
      const rect = element?.getBoundingClientRect()
      return rect ? { x: rect.x, y: rect.y, width: rect.width, height: rect.height } : null
    }
    const header = bounds(document.querySelector('.app-header'))!
    const contentElement = document.querySelector<HTMLElement>('.app-content')!
    const contentBounds = bounds(contentElement)!
    const primaryNav = document.querySelector<HTMLElement>('.primary-nav')!
    const activeLink = primaryNav.querySelector('.active')
    const navBounds = primaryNav.getBoundingClientRect()
    const activeBounds = activeLink?.getBoundingClientRect()
    const pageHost = [...contentElement.children].find((child) => child.tagName.toLowerCase().startsWith('nw-')) ?? null
    const pageBounds = bounds(pageHost)
    return {
      route,
      viewport,
      header,
      content: { ...contentBounds, scrollHeight: contentElement.scrollHeight, clientHeight: contentElement.clientHeight },
      pageOrigin: pageBounds ? { x: pageBounds.x, y: pageBounds.y, width: pageBounds.width } : null,
      activeNavigationVisible: Boolean(activeBounds && activeBounds.left >= navBounds.left - 1 && activeBounds.right <= navBounds.right + 1),
      documentOverflowsHorizontally: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    }
  }, { route, viewport })
}

test.describe.configure({ mode: 'serial' })

test('keeps the application shell stable and captures every primary workspace at shared viewports', async ({ page, request }) => {
  test.setTimeout(240_000)
  await mkdir(evidenceRoot, { recursive: true })
  await mkdir(fixtureRoot, { recursive: true })
  const sourcePath = path.join(fixtureRoot, `route-fixture-${process.pid}.csv`)
  await writeFile(sourcePath, 'age,weight,diagnosis\n42,71,positive\n50,80,negative\n61,90,positive\n', 'utf8')

  let modelId: string | undefined
  let datasetId: string | undefined
  const geometry: RouteGeometry[] = []
  const pageErrors: string[] = []
  const failedRequests: string[] = []
  page.on('pageerror', (error) => pageErrors.push(error.message))
  page.on('requestfailed', (request) => failedRequests.push(`${request.method()} ${request.url()}`))

  try {
    await ensurePytorchKerasProfile(request)
    modelId = await createProject(request)
    datasetId = await createDataset(request, sourcePath)

    const routes: Route[] = [
      { name: 'models', path: () => '/models', ready: 'nw-model-catalog-page' },
      { name: 'model-overview', path: ({ modelId }) => `/models/${modelId}`, ready: 'nw-model-overview-page' },
      { name: 'model-compilation', path: ({ modelId }) => `/models/${modelId}/compilation`, ready: 'nw-model-compilation-page' },
      { name: 'model-validation', path: ({ modelId }) => `/models/${modelId}/validation`, ready: 'nw-model-validation-page' },
      { name: 'model-export', path: ({ modelId }) => `/models/${modelId}/export`, ready: 'nw-model-export-page' },
      { name: 'builder-new', path: () => '/builder', ready: 'nw-app-shell .app-shell.builder-shell' },
      { name: 'builder-saved-model', path: ({ modelId }) => `/builder/${modelId}`, ready: 'nw-app-shell .app-shell.builder-shell' },
      { name: 'nodes', path: () => '/nodes', ready: 'nw-nodes-page' },
      { name: 'datasets', path: () => '/datasets', ready: 'nw-datasets-page' },
      { name: 'dataset-overview', path: ({ datasetId }) => `/datasets/${datasetId}`, ready: 'nw-dataset-overview-page' },
      { name: 'dataset-validation', path: ({ datasetId }) => `/datasets/${datasetId}/validation`, ready: 'nw-dataset-validation-page' },
      { name: 'dataset-io', path: ({ datasetId }) => `/datasets/${datasetId}/io`, ready: 'nw-dataset-io-page' },
      { name: 'dataset-processing', path: ({ datasetId }) => `/datasets/${datasetId}/processing`, ready: 'nw-dataset-processing-page' },
      { name: 'dataset-split', path: ({ datasetId }) => `/datasets/${datasetId}/split`, ready: 'nw-dataset-split-page' },
      { name: 'training', path: () => '/training', ready: 'nw-training-page' },
      { name: 'environment', path: () => '/environment', ready: 'nw-environment-page' },
      { name: 'metrics', path: () => '/metrics', ready: 'nw-metrics-page' },
      { name: 'settings', path: () => '/settings', ready: 'nw-settings-page' },
    ]
    const viewports = [
      { width: 1440, height: 900 },
      { width: 1024, height: 768 },
      { width: 560, height: 700 },
    ]
    const ids = { modelId, datasetId }

    for (const viewport of viewports) {
      await page.setViewportSize(viewport)
      let referenceHeader: RouteGeometry['header'] | undefined
      let referenceContentOrigin: { x: number; y: number } | undefined

      for (const route of routes) {
        await page.goto(route.path(ids))
        await expect(page.locator(route.ready)).toBeVisible({ timeout: 30_000 })
        await expect(page.locator('.primary-nav a.active')).toBeVisible()
        await expect(page.locator('nw-app-shell')).toBeVisible()
        if (route.name === 'models') await expect(page.getByText('UI consistency route fixture', { exact: true })).toBeVisible()
        if (route.name === 'model-overview') await expect(page.getByRole('heading', { name: 'UI consistency route fixture' })).toBeVisible()
        if (route.name === 'datasets') await expect(page.getByText('UI consistency dataset fixture', { exact: true })).toBeVisible()
        if (route.name === 'dataset-overview') await expect(page.getByRole('heading', { name: 'UI consistency dataset fixture' })).toBeVisible()
        if (route.name === 'training') await expect(page.getByRole('button', { name: 'New training' })).toBeEnabled()

        const routeGeometry = await measureRoute(page, route.name, viewport)
        referenceHeader ??= routeGeometry.header
        referenceContentOrigin ??= { x: routeGeometry.pageOrigin?.x ?? Number.NaN, y: routeGeometry.pageOrigin?.y ?? Number.NaN }
        expect(routeGeometry.header.x).toBeCloseTo(referenceHeader.x, 0)
        expect(routeGeometry.header.y).toBeCloseTo(referenceHeader.y, 0)
        expect(routeGeometry.header.width).toBeCloseTo(referenceHeader.width, 0)
        expect(routeGeometry.header.height).toBeCloseTo(referenceHeader.height, 0)
        if (captureStage !== 'baseline') {
          expect(routeGeometry.pageOrigin?.x).toBeCloseTo(referenceContentOrigin.x, 0)
          expect(routeGeometry.pageOrigin?.y).toBeCloseTo(referenceContentOrigin.y, 0)
          expect(routeGeometry.activeNavigationVisible).toBeTruthy()
        }
        expect(routeGeometry.documentOverflowsHorizontally).toBeFalsy()
        geometry.push(routeGeometry)

        await page.screenshot({
          path: path.join(evidenceRoot, `${captureStage}-${route.name}-${viewport.width}x${viewport.height}.png`),
          fullPage: false,
        })
      }
    }

    expect(pageErrors, 'browser page errors').toEqual([])
    expect(failedRequests, 'failed browser requests').toEqual([])
    await writeFile(path.join(evidenceRoot, `${captureStage}-route-geometry.json`), JSON.stringify({ geometry, pageErrors, failedRequests }, null, 2), 'utf8')
  } finally {
    if (datasetId) await removeFixture(request, '/api/datasets', datasetId)
    if (modelId) await removeFixture(request, '/api/projects', modelId)
    await rm(sourcePath, { force: true })
  }
})
