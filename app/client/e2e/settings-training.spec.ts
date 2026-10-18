// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { mkdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

type BrowserDiagnostics = {
  pageErrors: string[]
  consoleErrors: string[]
  failedRequests: string[]
  httpErrors: Array<{ method: string; path: string; status: number }>
  trainingWrites: string[]
}

const campaignRoot = path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260923')
const fixtureRoot = path.join(campaignRoot, 'fixtures')
const evidenceRoot = process.env.NW_T1_05_EVIDENCE_DIR ?? path.join(campaignRoot, 'evidence')
const fixturePath = path.join(fixtureRoot, `training-boundary-${process.pid}.csv`)
const projectName = `NW T1-05 model ${process.pid}`
const datasetName = `NW T1-05 dataset ${process.pid}`
const trainingName = `NW T1-05 draft ${process.pid}`

function collectBrowserDiagnostics(page: Page): BrowserDiagnostics {
  const diagnostics: BrowserDiagnostics = {
    pageErrors: [],
    consoleErrors: [],
    failedRequests: [],
    httpErrors: [],
    trainingWrites: [],
  }

  page.on('pageerror', (error) => diagnostics.pageErrors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') diagnostics.consoleErrors.push(message.text())
  })
  page.on('requestfailed', (request) => {
    diagnostics.failedRequests.push(`${request.method()} ${request.url()}: ${request.failure()?.errorText ?? 'failed'}`)
  })
  page.on('request', (request) => {
    const pathname = new URL(request.url()).pathname
    const hasTrainingOrJobSegment = pathname.split('/').some((segment) => segment === 'training' || segment === 'jobs')
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method()) && pathname.startsWith('/api/') && hasTrainingOrJobSegment) {
      diagnostics.trainingWrites.push(`${request.method()} ${pathname}`)
    }
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

  return diagnostics
}

async function createProject(request: APIRequestContext): Promise<string> {
  const response = await request.post('/api/projects', {
    data: {
      name: projectName,
      description: 'Disposable NW-T1-05 training boundary fixture',
      execution_backend: 'pytorch',
      keras_enabled: true,
      graph: {},
    },
  })
  expect(response.status()).toBe(201)
  return ((await response.json()) as { id: string }).id
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

async function registerDataset(page: Page): Promise<string> {
  await page.goto('/datasets')
  await page.locator('header').getByRole('button', { name: 'Register dataset' }).click()
  await page.getByPlaceholder('e.g. Patient outcomes').fill(datasetName)
  await page.getByRole('button', { name: 'Browse…' }).click()
  await selectFileFromBrowser(page, fixturePath)
  await page.getByRole('button', { name: 'Inspect sources' }).click()
  await expect(page.getByRole('heading', { name: 'Inspect and confirm' })).toBeVisible()
  await page.getByRole('button', { name: 'Inspect & confirm' }).click()
  await expect(page.getByRole('heading', { name: 'Register', exact: true })).toBeVisible()

  const createResponse = page.waitForResponse((response) => new URL(response.url()).pathname === '/api/datasets' && response.request().method() === 'POST')
  await page.locator('nw-dataset-wizard').getByRole('button', { name: 'Register dataset' }).click()
  const created = await createResponse
  expect(created.status()).toBe(201)
  return ((await created.json()) as { id: string }).id
}

test('NW-T1-05 keeps Settings informational and Training session-only', async ({ page, request }) => {
  test.setTimeout(120_000)
  const diagnostics = collectBrowserDiagnostics(page)
  let projectId: string | undefined
  let datasetId: string | undefined

  await mkdir(fixtureRoot, { recursive: true })
  await mkdir(evidenceRoot, { recursive: true })
  await writeFile(fixturePath, 'feature_a,label\n1.0,positive\n2.0,negative\n', 'utf8')

  try {
    await page.goto('/settings')
    await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible()
    await expect(page.getByText('There are no editable application-level settings defined yet.', { exact: true })).toBeVisible()
    await expect(page.locator('.settings-content button, .settings-content input, .settings-content select, .settings-content textarea, .settings-content [contenteditable]:not([contenteditable="false"]), .settings-content [role="switch"], .settings-content [role="checkbox"]')).toHaveCount(0)
    await expect(page.locator('.settings-content form')).toHaveCount(0)
    await expect(page.locator('.settings-boundary')).toContainText('.env')
    await expect(page.locator('.settings-boundary')).toContainText('database paths')
    await page.screenshot({ path: path.join(evidenceRoot, 'nw-t1-05-settings-boundary.png'), fullPage: true })

    await page.getByRole('link', { name: /Node Library/ }).click()
    await expect(page).toHaveURL(/\/nodes$/)
    await expect(page.getByRole('heading', { name: 'Nodes', exact: true })).toBeVisible()
    await page.goto('/settings')
    await page.getByRole('link', { name: /Runtime Environment/ }).click()
    await expect(page).toHaveURL(/\/environment$/)
    await expect(page.getByRole('heading', { name: 'Execution backends', exact: true })).toBeVisible()

    const emptyProjects = await request.get('/api/projects')
    const emptyDatasets = await request.get('/api/datasets')
    expect(emptyProjects.ok()).toBeTruthy()
    expect(emptyDatasets.ok()).toBeTruthy()
    expect((await emptyProjects.json() as { projects: unknown[] }).projects).toEqual([])
    expect((await emptyDatasets.json() as { datasets: unknown[] }).datasets).toEqual([])

    await page.goto('/training')
    await expect(page.getByRole('heading', { name: 'Training', exact: true })).toBeVisible()
    await expect(page.getByText('A registered dataset is required.', { exact: true })).toBeVisible()
    await expect(page.getByText('A saved model is required.', { exact: true })).toBeVisible()
    await expect(page.locator('header').getByRole('button', { name: 'New training' })).toBeDisabled()

    projectId = await createProject(request)
    datasetId = await registerDataset(page)

    // Reload the browser after fixture setup so AppResourcesService sees the persisted prerequisites.
    await page.goto('/training')
    await page.reload()
    await expect(page.getByRole('button', { name: 'New training' })).toBeEnabled()
    await expect(page.locator('.training-prerequisite')).toHaveCount(0)
    await page.getByRole('button', { name: 'New training' }).click()

    const dialog = page.getByRole('dialog', { name: 'Training setup' })
    await expect(dialog).toBeVisible()
    await dialog.getByLabel('Name').fill(trainingName)
    await dialog.getByLabel('Epochs').fill('7')
    await dialog.getByLabel('Batch size').fill('16')
    await dialog.getByRole('button', { name: 'Create draft' }).click()

    const draft = page.locator('.training-card').filter({ hasText: trainingName })
    await expect(draft).toBeVisible()
    await expect(draft).toContainText(projectName)
    await expect(draft).toContainText(datasetName)
    await expect(draft.locator('.training-stats strong').nth(0)).toHaveText('7')
    await expect(draft.locator('.training-stats strong').nth(1)).toHaveText('16')
    await expect(draft).toContainText('DRAFT · not started')
    await expect(draft.getByRole('button', { name: 'Start training' })).toBeDisabled()
    expect(await page.evaluate(() => [...Object.keys(localStorage), ...Object.keys(sessionStorage)].filter((key) => /training/i.test(key)))).toEqual([])
    await page.screenshot({ path: path.join(evidenceRoot, 'nw-t1-05-training-session-draft.png'), fullPage: true })

    await page.getByRole('link', { name: 'Settings', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible()
    await page.getByRole('link', { name: 'Training', exact: true }).click()
    await expect(page.locator('.training-card').filter({ hasText: trainingName })).toBeVisible()

    await page.reload()
    await expect(page.getByRole('heading', { name: 'No training sessions', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'New training' })).toBeEnabled()
    await page.screenshot({ path: path.join(evidenceRoot, 'nw-t1-05-training-after-reload.png'), fullPage: true })
    expect(diagnostics.trainingWrites).toEqual([])
    expect(diagnostics.pageErrors).toEqual([])
    expect(diagnostics.consoleErrors).toEqual([])
    expect(diagnostics.failedRequests).toEqual([])
    expect(diagnostics.httpErrors).toEqual([])
  } finally {
    try {
      if (datasetId) {
        const response = await request.delete(`/api/datasets/${datasetId}`)
        expect([204, 404]).toContain(response.status())
      }
      if (projectId) {
        const response = await request.delete(`/api/projects/${projectId}`)
        expect([204, 404]).toContain(response.status())
      }
    } finally {
      await rm(fixturePath, { force: true })
      await writeFile(path.join(evidenceRoot, 'nw-t1-05-browser-diagnostics.json'), JSON.stringify(diagnostics, null, 2), 'utf8')
    }
  }
})
