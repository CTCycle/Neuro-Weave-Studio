// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { mkdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page, type TestInfo } from '@playwright/test'

type HttpFailure = { method: string; path: string; status: number }
type BrowserDiagnostics = {
  routes: Array<{ url: string; primaryNavigation: string | null; workspaceTab: string | null; identity: string }>
  pageErrors: string[]
  consoleErrors: string[]
  failedRequests: string[]
  httpErrors: HttpFailure[]
}

function collectBrowserDiagnostics(page: Page): BrowserDiagnostics {
  const diagnostics: BrowserDiagnostics = { routes: [], pageErrors: [], consoleErrors: [], failedRequests: [], httpErrors: [] }
  page.on('pageerror', (error) => diagnostics.pageErrors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') diagnostics.consoleErrors.push(message.text())
  })
  page.on('requestfailed', (request) => diagnostics.failedRequests.push(`${request.method()} ${request.url()}: ${request.failure()?.errorText ?? 'failed'}`))
  page.on('response', (response) => {
    if (response.status() >= 400) {
      diagnostics.httpErrors.push({ method: response.request().method(), path: new URL(response.url()).pathname, status: response.status() })
    }
  })
  return diagnostics
}

async function recordRoute(page: Page, diagnostics: BrowserDiagnostics, identity: string): Promise<void> {
  const primary = page.locator('.primary-nav a.active')
  const primaryNavigation = await primary.count() ? (await primary.first().innerText()).trim() : null
  const workspace = page.locator('nav[aria-label="Model workspace sections"] a.active, nav[aria-label="Dataset workspace sections"] a.active')
  const workspaceTab = await workspace.count() ? (await workspace.first().innerText()).trim() : null
  diagnostics.routes.push({ url: page.url(), primaryNavigation, workspaceTab, identity })
}

function expectCleanBrowserDiagnostics(
  diagnostics: BrowserDiagnostics,
  allowedHttpErrors: HttpFailure[] = [],
  allowedConsoleErrors: string[] = [],
): void {
  expect(diagnostics.pageErrors, 'browser page errors').toEqual([])
  expect(diagnostics.consoleErrors.sort(), 'unexpected browser console errors').toEqual(allowedConsoleErrors.sort())
  expect(diagnostics.failedRequests, 'failed browser requests').toEqual([])
  const describe = (failure: HttpFailure) => `${failure.method} ${failure.path} ${failure.status}`
  expect(diagnostics.httpErrors.map(describe).sort(), 'unexpected HTTP error responses').toEqual(allowedHttpErrors.map(describe).sort())
}

async function saveBrowserEvidence(page: Page, testInfo: TestInfo, diagnostics: BrowserDiagnostics): Promise<void> {
  const payload = JSON.stringify({ test: testInfo.title, repeatEachIndex: testInfo.repeatEachIndex, ...diagnostics }, null, 2)
  const evidenceRoot = process.env.NEURO_WEAVE_T1_EVIDENCE_DIR
  const runLabel = (process.env.NW_T1_RUN_LABEL ?? 'local').replace(/[^a-z0-9-]/gi, '-').toLowerCase()
  const slug = testInfo.title.replace(/[^a-z0-9-]/gi, '-').replace(/-+/g, '-').toLowerCase()
  const repeat = testInfo.repeatEachIndex + 1
  if (evidenceRoot) {
    await mkdir(evidenceRoot, { recursive: true })
    await writeFile(path.join(evidenceRoot, `${slug}-${runLabel}-${repeat}.json`), payload, 'utf8')
    console.log(`[NW-T1-01] ${testInfo.title}: ${diagnostics.routes.length} routes; ${diagnostics.pageErrors.length} page errors; ${diagnostics.consoleErrors.length} console errors; ${diagnostics.failedRequests.length} failed requests; ${diagnostics.httpErrors.length} HTTP errors`)
  } else {
    console.log(`[NW-T1-01] ${testInfo.title}: ${payload}`)
    await testInfo.attach(`${slug}-browser-diagnostics.json`, { body: payload, contentType: 'application/json' })
  }
  if (evidenceRoot && process.env.NEURO_WEAVE_T1_CAPTURE_SCREENSHOTS === '1') {
    const screenshotName = testInfo.title.includes('dataset workspace')
      ? `dataset-workspace-final-${runLabel}-${repeat}.png`
      : testInfo.title.includes('model workspace')
        ? `model-workspace-validation-${runLabel}-${repeat}.png`
        : null
    if (screenshotName) await page.screenshot({ path: path.join(evidenceRoot, screenshotName), fullPage: true })
  }
}

async function expectPrimaryRoute(page: Page, route: { navigation: string; path: string; heading?: string; selector?: string }): Promise<void> {
  await expect(page).toHaveURL(new RegExp(`${route.path}$`))
  await expect(page.locator('nw-app-shell')).toBeVisible()
  const active = page.locator('.primary-nav a.active')
  await expect(active).toHaveCount(1)
  await expect(active).toHaveText(route.navigation)
  if (route.heading) await expect(page.getByRole('heading', { name: route.heading, exact: true }).first()).toBeVisible()
  if (route.selector) await expect(page.locator(route.selector)).toBeVisible()
}

async function createRouteModel(request: APIRequestContext, name: string): Promise<string> {
  const response = await request.post('/api/projects', {
    data: { name, description: 'Disposable NW-T1-01 route fixture', execution_backend: 'pytorch', keras_enabled: true, graph: {} },
  })
  expect(response.status()).toBe(201)
  return ((await response.json()) as { id: string }).id
}

async function deleteRouteModel(request: APIRequestContext, id: string): Promise<void> {
  const response = await request.delete(`/api/projects/${id}`)
  expect([204, 404]).toContain(response.status())
}

async function selectFileFromBrowser(page: Page, sourcePath: string): Promise<void> {
  const browser = page.getByRole('dialog').filter({ hasText: 'Select CSV table' })
  const filesystemPath = browser.getByLabel('Filesystem path')
  await filesystemPath.fill(path.dirname(sourcePath))
  await expect(browser.locator('.browser-body')).toHaveAttribute('aria-busy', 'false', { timeout: 30_000 })
  const browseResponse = page.waitForResponse((response) => response.url().includes('/api/filesystem/browse') && response.request().method() === 'GET')
  await browser.getByRole('button', { name: 'Browse' }).click()
  expect((await browseResponse).ok()).toBeTruthy()
  await expect(browser.locator('.browser-body')).toHaveAttribute('aria-busy', 'false', { timeout: 30_000 })
  const fixtureEntry = browser.locator('.browser-item').filter({ hasText: path.basename(sourcePath) })
  await expect(fixtureEntry).toBeVisible({ timeout: 30_000 })
  await fixtureEntry.click()
  await browser.getByRole('button', { name: 'Use selected file' }).click()
}

async function prepareTabularWizard(page: Page, sourcePath: string, name: string): Promise<void> {
  await page.goto('/datasets')
  await page.locator('header').getByRole('button', { name: 'Register dataset' }).click()
  await page.getByPlaceholder('e.g. Patient outcomes').fill(name)
  await page.getByRole('button', { name: 'Browse…' }).click()
  await selectFileFromBrowser(page, sourcePath)
  await page.getByRole('button', { name: 'Inspect sources' }).click()
  await expect(page.getByRole('heading', { name: 'Inspect and confirm' })).toBeVisible()
  await page.getByRole('button', { name: 'Inspect & confirm' }).click()
  await expect(page.getByRole('heading', { name: 'Register', exact: true })).toBeVisible()
}

test('starts the Angular shell at the model catalog', async ({ page }) => {
  await page.goto('/models')
  await expect(page.locator('nw-app-shell')).toBeVisible()
  await expect(page.getByRole('link', { name: 'Models', exact: true })).toHaveClass(/active/)
  await expect(page.locator('app-root')).toContainText('Saved models')
})

test('serves Angular deep links and preserves the shell on refresh', async ({ page }) => {
  test.setTimeout(60_000)
  for (const route of ['/datasets', '/training', '/environment', '/metrics', '/nodes', '/settings', '/builder', '/models/missing/compilation']) {
    await page.goto(route)
    await expect(page.locator('nw-app-shell')).toBeVisible()
    await page.reload()
    await expect(page.locator('nw-app-shell')).toBeVisible()
  }
})

test('exposes dataset and training as primary navigation tabs', async ({ page }) => {
  await page.goto('/datasets')
  await expect(page.getByRole('link', { name: 'Datasets', exact: true })).toHaveClass(/active/)
  await expect(page.getByRole('heading', { name: 'Datasets', exact: true })).toBeVisible()
  await page.getByRole('link', { name: 'Training', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Training', exact: true })).toBeVisible()
})

test('exposes node library and global settings navigation', async ({ page }) => {
  await page.goto('/nodes')
  await expect(page.getByRole('link', { name: 'Nodes', exact: true })).toHaveClass(/active/)
  await expect(page.getByRole('heading', { name: 'Nodes', exact: true })).toBeVisible()
  await expect(page.getByText('Built-in', { exact: true }).first()).toBeVisible()

  await page.getByRole('link', { name: 'Settings', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible()
  await expect(page.getByRole('navigation', { name: 'Settings sections' })).toBeVisible()
  await expect(page.getByText('There are no editable application-level settings defined yet.', { exact: true })).toBeVisible()
  await expect(page.locator('.settings-content input, .settings-content select')).toHaveCount(0)
  await expect(page.getByRole('link', { name: /Node Library/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /Runtime Environment/ })).toBeVisible()

  const settingsSections = page.getByRole('navigation', { name: 'Settings sections' })
  await settingsSections.getByRole('link', { name: /Application/ }).click()
  await expect(page).toHaveURL(/\/settings#application-general$/)
  await expect(page.locator('#application-general')).toBeVisible()
  await settingsSections.getByRole('link', { name: /Related management/ }).click()
  await expect(page).toHaveURL(/\/settings#related-management$/)
  await expect(page.locator('#related-management')).toBeVisible()
})

test('dismisses the dataset registration dialog with Escape', async ({ page }) => {
  await page.goto('/datasets')
  await page.getByRole('button', { name: 'Register dataset' }).click()
  await expect(page.getByRole('dialog', { name: /Dataset and sources/ })).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog', { name: /Dataset and sources/ })).toHaveCount(0)
})

test('NW-T1-01 navigates every primary section and resolves unknown routes to Models', async ({ page }, testInfo) => {
  const diagnostics = collectBrowserDiagnostics(page)
  const routes = [
    { navigation: 'Models', path: '/models', heading: 'Saved models' },
    { navigation: 'Model Builder', path: '/builder', selector: 'nw-model-builder-page' },
    { navigation: 'Nodes', path: '/nodes', heading: 'Nodes' },
    { navigation: 'Datasets', path: '/datasets', heading: 'Datasets' },
    { navigation: 'Training', path: '/training', heading: 'Training' },
    { navigation: 'Metrics', path: '/metrics', heading: 'Metrics' },
    { navigation: 'Environment', path: '/environment', heading: 'Execution backends' },
    { navigation: 'Settings', path: '/settings', heading: 'Settings' },
  ]
  try {
    await page.goto('/models')
    for (const [index, route] of routes.entries()) {
      if (index > 0) await page.getByRole('link', { name: route.navigation, exact: true }).click()
      await expectPrimaryRoute(page, route)
      await recordRoute(page, diagnostics, route.heading ?? route.selector ?? route.navigation)
    }

    await page.goto('/nw-t1-01-unknown-application-route')
    await expectPrimaryRoute(page, routes[0])
    await recordRoute(page, diagnostics, 'wildcard redirected to Saved models')

    const datasets = routes.find((route) => route.navigation === 'Datasets')!
    const settings = routes.find((route) => route.navigation === 'Settings')!
    await page.getByRole('link', { name: datasets.navigation, exact: true }).click()
    await expectPrimaryRoute(page, datasets)
    await recordRoute(page, diagnostics, 'Datasets')
    await page.getByRole('link', { name: settings.navigation, exact: true }).click()
    await expectPrimaryRoute(page, settings)
    await recordRoute(page, diagnostics, 'Settings')

    await page.goBack()
    await expectPrimaryRoute(page, datasets)
    await recordRoute(page, diagnostics, 'Back to Datasets')
    await page.goBack()
    await expectPrimaryRoute(page, routes[0])
    await recordRoute(page, diagnostics, 'Back to Models')
    await page.goForward()
    await expectPrimaryRoute(page, datasets)
    await recordRoute(page, diagnostics, 'Forward to Datasets')
    await page.goForward()
    await expectPrimaryRoute(page, settings)
    await recordRoute(page, diagnostics, 'Forward to Settings')

    expectCleanBrowserDiagnostics(diagnostics)
  } finally {
    await saveBrowserEvidence(page, testInfo, diagnostics)
  }
})

test('NW-T1-01 keeps the shell and Back to models boundary for missing workspaces', async ({ page }, testInfo) => {
  const diagnostics = collectBrowserDiagnostics(page)
  const missingModelId = '00000000-0000-4000-8000-000000000001'
  const missingPath = `/api/projects/${missingModelId}`
  try {
    for (const route of [`/models/${missingModelId}`, `/models/${missingModelId}/compilation`]) {
      await page.goto(route)
      await expect(page.locator('nw-app-shell')).toBeVisible()
      await expect(page.getByRole('heading', { name: 'Model unavailable', exact: true })).toBeVisible()
      await expect(page.getByRole('link', { name: 'Back to models', exact: true })).toHaveAttribute('href', '/models')
      await expect(page.locator('.workspace-body')).toHaveCount(0)
      await expect(page.locator('.primary-nav a.active')).toHaveText('Models')
      diagnostics.routes.push({ url: page.url(), primaryNavigation: 'Models', workspaceTab: null, identity: 'Model unavailable; no nested outlet' })
    }

    await page.getByRole('link', { name: 'Back to models', exact: true }).click()
    await expectPrimaryRoute(page, { navigation: 'Models', path: '/models', heading: 'Saved models' })
    await recordRoute(page, diagnostics, 'Back to models')

    const expectedNotFound: HttpFailure[] = [
      { method: 'GET', path: missingPath, status: 404 },
      { method: 'GET', path: missingPath, status: 404 },
    ]
    const expectedNotFoundConsoleError = 'Failed to load resource: the server responded with a status of 404 (Not Found)'
    expectCleanBrowserDiagnostics(diagnostics, expectedNotFound, [expectedNotFoundConsoleError, expectedNotFoundConsoleError])
  } finally {
    await saveBrowserEvidence(page, testInfo, diagnostics)
  }
})

test('NW-T1-01 preserves Settings anchors and browser history', async ({ page }, testInfo) => {
  const diagnostics = collectBrowserDiagnostics(page)
  try {
    await page.goto('/settings#application-general')
    await expect(page).toHaveURL(/\/settings#application-general$/)
    await expect(page.locator('nw-app-shell')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible()
    await expect(page.locator('#application-general')).toBeVisible()
    await expect(page.locator('.primary-nav a.active')).toHaveText('Settings')
    await recordRoute(page, diagnostics, 'Settings: application-general')

    const sections = page.getByRole('navigation', { name: 'Settings sections' })
    await sections.getByRole('link', { name: /Related management/ }).click()
    await expect(page).toHaveURL(/\/settings#related-management$/)
    await expect(page.locator('#related-management')).toBeVisible()
    await recordRoute(page, diagnostics, 'Settings: related-management')

    await page.goBack()
    await expect(page).toHaveURL(/\/settings#application-general$/)
    await expect(page.locator('#application-general')).toBeVisible()
    await recordRoute(page, diagnostics, 'Back to application-general')
    await page.goForward()
    await expect(page).toHaveURL(/\/settings#related-management$/)
    await expect(page.locator('#related-management')).toBeVisible()
    await recordRoute(page, diagnostics, 'Forward to related-management')

    expectCleanBrowserDiagnostics(diagnostics)
  } finally {
    await saveBrowserEvidence(page, testInfo, diagnostics)
  }
})

test('NW-T1-01 retains a saved model across workspace tabs and browser history', async ({ page, request }, testInfo) => {
  const diagnostics = collectBrowserDiagnostics(page)
  let modelId: string | undefined
  const modelName = `NW T1-01 route model ${Date.now()}`
  const tabs = [
    { name: 'Overview', suffix: '', selector: 'nw-model-overview-page' },
    { name: 'Compilation', suffix: '/compilation', selector: 'nw-model-compilation-page' },
    { name: 'Validation', suffix: '/validation', selector: 'nw-model-validation-page' },
    { name: 'Export', suffix: '/export', selector: 'nw-model-export-page' },
  ]
  const expectModelTab = async (tab: (typeof tabs)[number]) => {
    const route = `/models/${modelId}${tab.suffix}`
    await expect(page).toHaveURL(new RegExp(`${route}$`))
    await expect(page.locator('nw-app-shell')).toBeVisible()
    await expect(page.getByRole('heading', { name: modelName, exact: true })).toBeVisible()
    await expect(page.locator('.primary-nav a.active')).toHaveText('Models')
    const navigation = page.getByRole('navigation', { name: 'Model workspace sections' })
    await expect(navigation.getByRole('link', { name: tab.name, exact: true })).toHaveClass(/active/)
    await expect(page.locator(tab.selector)).toBeVisible()
    await recordRoute(page, diagnostics, `Model ${modelName}: ${tab.name}`)
  }

  try {
    modelId = await createRouteModel(request, modelName)
    await page.goto(`/models/${modelId}`)
    for (const [index, tab] of tabs.entries()) {
      if (index > 0) await page.getByRole('navigation', { name: 'Model workspace sections' }).getByRole('link', { name: tab.name, exact: true }).click()
      await expectModelTab(tab)
    }

    await page.goBack()
    await expectModelTab(tabs[2])
    await page.goBack()
    await expectModelTab(tabs[1])
    await page.goForward()
    await expectModelTab(tabs[2])

    expectCleanBrowserDiagnostics(diagnostics)
  } finally {
    try {
      if (modelId) await deleteRouteModel(request, modelId)
    } finally {
      await saveBrowserEvidence(page, testInfo, diagnostics)
    }
  }
})

test('NW-T1-01 exercises dataset workspace tabs and structure redirect', async ({ page, request }, testInfo) => {
  test.setTimeout(120_000)
  const diagnostics = collectBrowserDiagnostics(page)
  const fixtureBase = path.resolve(process.env.NEURO_WEAVE_E2E_FIXTURE_DIR ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA'))
  const fixtureRoot = path.join(fixtureBase, `nw-t1-01-${process.pid}-${Date.now()}`)
  const fixturePath = path.join(fixtureRoot, 'route-dataset.csv')
  const datasetName = `NW T1-01 route dataset ${Date.now()}`
  const sourceContents = 'age,weight,diagnosis\n42,71,positive\n50,80,negative\n61,90,positive\n'
  let datasetId: string | undefined
  const tabs = [
    { name: 'Overview', suffix: '', selector: 'nw-dataset-overview-page' },
    { name: 'Validation', suffix: '/validation', selector: 'nw-dataset-validation-page' },
    { name: 'I/O', suffix: '/io', selector: 'nw-dataset-io-page' },
    { name: 'Processing', suffix: '/processing', selector: 'nw-dataset-processing-page' },
    { name: 'Split', suffix: '/split', selector: 'nw-dataset-split-page' },
  ]
  const expectDatasetTab = async (tab: (typeof tabs)[number]) => {
    const route = `/datasets/${datasetId}${tab.suffix}`
    await expect(page).toHaveURL(new RegExp(`${route}$`))
    await expect(page.locator('nw-app-shell')).toBeVisible()
    await expect(page.getByRole('heading', { name: datasetName, exact: true })).toBeVisible()
    await expect(page.locator('.primary-nav a.active')).toHaveText('Datasets')
    const navigation = page.getByRole('navigation', { name: 'Dataset workspace sections' })
    await expect(navigation.getByRole('link', { name: tab.name, exact: true })).toHaveClass(/active/)
    await expect(page.locator(tab.selector)).toBeVisible()
    await recordRoute(page, diagnostics, `Dataset ${datasetName}: ${tab.name}`)
  }

  try {
    await mkdir(fixtureRoot, { recursive: true })
    await writeFile(fixturePath, sourceContents, 'utf8')
    await prepareTabularWizard(page, fixturePath, datasetName)
    const createResponse = page.waitForResponse((response) => new URL(response.url()).pathname === '/api/datasets' && response.request().method() === 'POST')
    await page.locator('nw-dataset-wizard').getByRole('button', { name: 'Register dataset', exact: true }).click()
    const created = await createResponse
    expect(created.status()).toBe(201)
    datasetId = ((await created.json()) as { id: string }).id

    const card = page.locator('.dataset-card').filter({ hasText: datasetName })
    await expect(card).toBeVisible()
    await card.getByRole('link').click()
    await expect(page).toHaveURL(new RegExp(`/datasets/${datasetId}$`))
    for (const [index, tab] of tabs.entries()) {
      if (index > 0) await page.getByRole('navigation', { name: 'Dataset workspace sections' }).getByRole('link', { name: tab.name, exact: true }).click()
      await expectDatasetTab(tab)
    }

    await page.goBack()
    await expectDatasetTab(tabs[3])
    await page.goForward()
    await expectDatasetTab(tabs[4])
    await page.goto(`/datasets/${datasetId}/structure`)
    await expect(page).toHaveURL(new RegExp(`/datasets/${datasetId}/io$`))
    await expectDatasetTab(tabs[2])

    expectCleanBrowserDiagnostics(diagnostics)
  } finally {
    try {
      if (datasetId) {
        const response = await request.delete(`/api/datasets/${datasetId}`)
        expect([204, 404]).toContain(response.status())
      }
    } finally {
      await rm(fixtureRoot, { recursive: true, force: true })
      await saveBrowserEvidence(page, testInfo, diagnostics)
    }
  }
})
