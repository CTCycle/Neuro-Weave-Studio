// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import fs, { mkdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page, type TestInfo } from '@playwright/test'

  test.describe('accessibility.spec', () => {
    const coreRoutes = ['/models', '/nodes', '/datasets', '/training', '/metrics', '/environment', '/settings', '/builder']

    type AccessibilityAudit = {
      language: string
      title: string
      landmarkCount: { main: number; navigation: number }
      missingNames: Array<{ tag: string; role: string | null; html: string }>
    }

    test('runs the in-browser accessibility baseline across core routes', async ({ page }) => {
      test.setTimeout(120_000)
      for (const route of coreRoutes) {
        await page.goto(route)
        await expect(page.locator('nw-app-shell')).toBeVisible()
        const audit = await page.evaluate((): AccessibilityAudit => {
          const visible = (element: Element): boolean => {
            if (!(element instanceof HTMLElement)) return false
            const style = getComputedStyle(element)
            const bounds = element.getBoundingClientRect()
            return style.display !== 'none'
              && style.visibility !== 'hidden'
              && style.opacity !== '0'
              && bounds.width > 0
              && bounds.height > 0
          }
          const referencedName = (element: Element): string => {
            const ids = element.getAttribute('aria-labelledby')?.split(/\s+/).filter(Boolean) ?? []
            return ids.map((id) => document.getElementById(id)?.textContent ?? '').join(' ')
          }
          const controlName = (element: Element): string => {
            const ariaLabel = element.getAttribute('aria-label')?.trim()
            if (ariaLabel) return ariaLabel
            const labelledBy = referencedName(element).trim()
            if (labelledBy) return labelledBy
            if (element instanceof HTMLInputElement || element instanceof HTMLSelectElement || element instanceof HTMLTextAreaElement) {
              const label = element.labels?.[0]?.textContent?.trim()
              if (label) return label
            }
            const title = element.getAttribute('title')?.trim()
            if (title) return title
            return (element.textContent ?? '').replace(/\s+/g, ' ').trim()
          }
          const candidates = Array.from(document.querySelectorAll(
            'a,button,input,select,textarea,summary,[contenteditable="true"],[role="button"],[role="link"],[role="tab"],[role="checkbox"],[role="combobox"],[role="menuitem"]',
          )).filter(visible)
          return {
            language: document.documentElement.lang,
            title: document.title,
            landmarkCount: { main: document.querySelectorAll('main').length, navigation: document.querySelectorAll('nav').length },
            missingNames: candidates
              .filter((element) => !controlName(element))
              .map((element) => ({ tag: element.tagName.toLowerCase(), role: element.getAttribute('role'), html: element.outerHTML.slice(0, 180) })),
          }
        })
        expect(audit.language, route).toBe('en')
        expect(audit.title, route).toBe('NeuroWeave Model Studio')
        expect(audit.landmarkCount.main, route).toBe(1)
        expect(audit.landmarkCount.navigation, route).toBeGreaterThan(0)
        expect(audit.missingNames, route).toEqual([])
      }
    })

    test('keeps keyboard focus visible on core controls and graph landmarks', async ({ page }) => {
      await page.goto('/builder')
      await expect(page.locator('nw-app-shell')).toBeVisible()
      const focusSamples: Array<{ tag: string; outlineWidth: string; outlineStyle: string; boxShadow: string }> = []
      for (let index = 0; index < 12; index += 1) {
        await page.keyboard.press('Tab')
        const sample = await page.evaluate(() => {
          const element = document.activeElement
          if (!(element instanceof HTMLElement) || element === document.body) return null
          const style = getComputedStyle(element)
          const bounds = element.getBoundingClientRect()
          return bounds.width > 0 && bounds.height > 0
            ? { tag: element.tagName.toLowerCase(), outlineWidth: style.outlineWidth, outlineStyle: style.outlineStyle, boxShadow: style.boxShadow }
            : null
        })
        if (sample) focusSamples.push(sample)
      }
      expect(focusSamples.length).toBeGreaterThan(0)
      expect(focusSamples.some((sample) => (
        (sample.outlineStyle !== 'none' && sample.outlineWidth !== '0px') || sample.boxShadow !== 'none'
      ))).toBeTruthy()
    })
  })
  test.describe('metrics.spec', () => {
    const evidenceDir = path.resolve(
      process.cwd(),
      process.env.NW_VALIDATION_EVIDENCE_DIR ?? '../../assets/QA/comprehensive-validation-20260926/evidence/metrics',
    )

    test('persists, edits, validates, reloads, and deletes custom metrics and losses', async ({ page, request }) => {
      test.setTimeout(90_000)
      await fs.mkdir(evidenceDir, { recursive: true })
      const suffix = Date.now()
      const metricLabel = `Browser validation metric ${suffix}`
      const lossLabel = `Browser validation loss ${suffix}`
      let metricId: string | undefined
      let lossId: string | undefined
      const evidence = { metricLabel, lossLabel, events: [] as Array<{ action: string; status?: number }> }

      try {
        await page.goto('/metrics')
        await expect(page.getByRole('heading', { name: 'Metrics', exact: true })).toBeVisible()
        await page.screenshot({ path: path.join(evidenceDir, 'metrics-initial.png'), fullPage: false })

        await page.getByRole('button', { name: 'Add loss / metric' }).click()
        const createMetricDialog = page.getByRole('dialog')
        await expect(createMetricDialog.getByRole('heading', { name: 'Add loss or metric' })).toBeVisible()
        await createMetricDialog.getByLabel('Name').fill(metricLabel)
        await createMetricDialog.getByLabel('Description').fill('Browser lifecycle metric.')
        await createMetricDialog.getByRole('textbox', { name: 'Formula', exact: true }).fill('mean(abs(y-yhat))')
        const createMetricResponse = page.waitForResponse(
          (response) => response.url().endsWith('/api/metrics') && response.request().method() === 'POST',
        )
        await createMetricDialog.getByRole('button', { name: 'Add definition' }).click()
        const metricResponse = await createMetricResponse
        evidence.events.push({ action: 'create-metric', status: metricResponse.status() })
        expect(metricResponse.status()).toBe(201)
        metricId = (await metricResponse.json() as { id: string }).id

        const metricCard = page.locator('.metrics-card').filter({ hasText: metricLabel })
        await expect(metricCard).toBeVisible()
        await expect(metricCard.getByRole('button', { name: `View ${metricLabel}` })).toBeVisible()
        await page.screenshot({ path: path.join(evidenceDir, 'metrics-after-create.png'), fullPage: true })

        await metricCard.getByRole('button', { name: `View ${metricLabel}` }).click()
        const viewDialog = page.getByRole('dialog')
        await expect(viewDialog.getByRole('heading', { name: metricLabel })).toBeVisible()
        await expect(viewDialog).toContainText('Browser lifecycle metric.')
        await viewDialog.getByRole('button', { name: 'Close definition' }).click()

        await metricCard.getByRole('button', { name: `Edit ${metricLabel}` }).click()
        const editDialog = page.getByRole('dialog')
        await expect(editDialog.getByRole('heading', { name: 'Edit loss or metric' })).toBeVisible()
        await editDialog.getByLabel('Description').fill('Edited browser lifecycle metric.')
        await editDialog.getByRole('textbox', { name: 'Formula', exact: true }).fill('mean((y-yhat)^2)')
        const updateMetricResponse = page.waitForResponse(
          (response) => response.url().includes(`/api/metrics/${metricId}`) && response.request().method() === 'PUT',
        )
        await editDialog.getByRole('button', { name: 'Save changes' }).click()
        const updatedMetricResponse = await updateMetricResponse
        evidence.events.push({ action: 'update-metric', status: updatedMetricResponse.status() })
        expect(updatedMetricResponse.status()).toBe(200)
        await expect(page.locator('.metrics-card').filter({ hasText: 'Edited browser lifecycle metric.' })).toBeVisible()

        await page.getByRole('button', { name: 'Add loss / metric' }).click()
        const createLossDialog = page.getByRole('dialog')
        await createLossDialog.getByLabel('Kind').selectOption('loss')
        await createLossDialog.getByLabel('Name').fill(lossLabel)
        await createLossDialog.getByLabel('Description').fill('Browser lifecycle loss.')
        await createLossDialog.getByRole('tab', { name: 'Equation builder' }).click()
        await expect(createLossDialog.getByRole('button', { name: 'mean(' })).toBeVisible()
        await createLossDialog.getByRole('textbox', { name: 'Formula', exact: true }).fill('mean((y-yhat)^2)')
        const createLossResponse = page.waitForResponse(
          (response) => response.url().endsWith('/api/metrics') && response.request().method() === 'POST',
        )
        await createLossDialog.getByRole('button', { name: 'Add definition' }).click()
        const lossResponse = await createLossResponse
        evidence.events.push({ action: 'create-loss', status: lossResponse.status() })
        expect(lossResponse.status()).toBe(201)
        lossId = (await lossResponse.json() as { id: string }).id
        await expect(page.locator('.metrics-card').filter({ hasText: lossLabel })).toBeVisible()

        await page.reload()
        await expect(page.getByRole('heading', { name: 'Metrics', exact: true })).toBeVisible()
        const reloadedMetricCard = page.locator('.metrics-card').filter({ hasText: metricLabel })
        const reloadedLossCard = page.locator('.metrics-card').filter({ hasText: lossLabel })
        await expect(reloadedMetricCard).toBeVisible()
        await expect(reloadedLossCard).toBeVisible()
        await reloadedMetricCard.screenshot({ path: path.join(evidenceDir, 'metrics-after-reload-metric-card.png') })
        await reloadedLossCard.screenshot({ path: path.join(evidenceDir, 'metrics-after-reload-loss-card.png') })
        await page.screenshot({ path: path.join(evidenceDir, 'metrics-after-reload.png'), fullPage: true })

        await page.getByRole('button', { name: 'Add loss / metric' }).click()
        const invalidDialog = page.getByRole('dialog')
        await invalidDialog.getByLabel('Name').fill(`Invalid formula ${suffix}`)
        await invalidDialog.getByLabel('Description').fill('Rejected browser validation case.')
        await invalidDialog.getByRole('textbox', { name: 'Formula', exact: true }).fill("__import__('os').system('whoami')")
        const invalidResponsePromise = page.waitForResponse(
          (response) => response.url().endsWith('/api/metrics') && response.request().method() === 'POST',
        )
        await invalidDialog.getByRole('button', { name: 'Add definition' }).click()
        const invalidResponse = await invalidResponsePromise
        evidence.events.push({ action: 'reject-invalid-formula', status: invalidResponse.status() })
        expect(invalidResponse.status()).toBe(422)
        await expect(invalidDialog.getByRole('alert')).toBeVisible()
        await page.screenshot({ path: path.join(evidenceDir, 'metrics-invalid-formula.png'), fullPage: false })
        await invalidDialog.getByRole('button', { name: 'Cancel' }).click()

        const deleteMetricCard = page.locator('.metrics-card').filter({ hasText: metricLabel })
        await deleteMetricCard.getByRole('button', { name: `Delete ${metricLabel}` }).click()
        const deleteDialog = page.getByRole('alertdialog')
        await expect(deleteDialog).toContainText('installation-wide compilation catalog')
        const deleteMetricResponse = page.waitForResponse(
          (response) => response.url().includes(`/api/metrics/${metricId}`) && response.request().method() === 'DELETE',
        )
        await deleteDialog.getByRole('button', { name: 'Delete definition' }).click()
        const deletedMetricResponse = await deleteMetricResponse
        evidence.events.push({ action: 'delete-metric', status: deletedMetricResponse.status() })
        expect(deletedMetricResponse.status()).toBe(204)
        metricId = undefined
        await expect(page.locator('.metrics-card').filter({ hasText: metricLabel })).toHaveCount(0)

        const deleteLossCard = page.locator('.metrics-card').filter({ hasText: lossLabel })
        await deleteLossCard.getByRole('button', { name: `Delete ${lossLabel}` }).click()
        const deleteLossDialog = page.getByRole('alertdialog')
        const deleteLossResponse = page.waitForResponse(
          (response) => response.url().includes(`/api/metrics/${lossId}`) && response.request().method() === 'DELETE',
        )
        await deleteLossDialog.getByRole('button', { name: 'Delete definition' }).click()
        const deletedLossResponse = await deleteLossResponse
        evidence.events.push({ action: 'delete-loss', status: deletedLossResponse.status() })
        expect(deletedLossResponse.status()).toBe(204)
        lossId = undefined
        await expect(page.locator('.metrics-card').filter({ hasText: lossLabel })).toHaveCount(0)
        await expect(page.getByRole('alertdialog')).toHaveCount(0)
        await page.screenshot({ path: path.join(evidenceDir, 'metrics-after-delete.png'), fullPage: false })
      } finally {
        if (metricId) {
          const response = await request.delete(`/api/metrics/${metricId}`)
          expect([204, 404]).toContain(response.status())
        }
        if (lossId) {
          const response = await request.delete(`/api/metrics/${lossId}`)
          expect([204, 404]).toContain(response.status())
        }
        await fs.writeFile(path.join(evidenceDir, 'metrics-lifecycle.json'), JSON.stringify(evidence, null, 2), 'utf8')
      }
    })
  })
  test.describe('shell.spec', () => {
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
      await expect(page.getByLabel('Log level')).toBeVisible()
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
  })
  test.describe('ui consistency.spec', () => {
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
  })
  test.describe('ui refinement.spec', () => {
    const evidenceDir = path.resolve(
      process.env.NW_VALIDATION_EVIDENCE_DIR
        ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'ui-refinement-20260909'),
    )

    type CreatedProject = { id: string; name: string }

    function graph() {
      return {
        schema_version: 6,
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
        model_configuration: {},
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
      await page.keyboard.press('Escape')
      await expect(page.locator('.canvas-context-menu')).toHaveCount(0)
      const node = page.locator('.nw-flow-node').filter({ hasText: label }).first()
      await expect(node).toBeVisible()
      await node.scrollIntoViewIfNeeded()
      await node.click({ button: 'right', force: true })
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
  })
