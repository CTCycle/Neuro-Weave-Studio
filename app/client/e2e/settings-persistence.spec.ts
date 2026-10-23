// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

  test.describe('database settings.spec', () => {
    const repositoryRoot = path.resolve(process.cwd(), '..', '..')
    const evidenceRoot = path.resolve(
      process.env.NW_DATABASE_SETTINGS_EVIDENCE_DIR
        ?? path.join(repositoryRoot, 'assets', 'QA', 'database-settings-ui-e2e-20261021'),
    )
    test.describe.configure({ mode: 'serial' })

    test('tests, switches, reloads, and restores independent SQLite settings through Settings UI', async ({ page, request }) => {
      test.setTimeout(120_000)
      await mkdir(evidenceRoot, { recursive: true })

      const initialStatusResponse = await request.get('/api/settings/database')
      expect(initialStatusResponse.ok()).toBeTruthy()
      const initialStatus = await initialStatusResponse.json() as { sqlite_path: string }
      const initialSettingsResponse = await request.get('/api/settings')
      expect(initialSettingsResponse.ok()).toBeTruthy()
      const initialSettings = await initialSettingsResponse.json() as { log_level: string }
      const secondaryPath = path.join(evidenceRoot, `switch-candidate-${process.pid}.sqlite`)
      const invalidPath = path.join(evidenceRoot, `invalid-target-${process.pid}`)
      const profileRoot = process.env['NW_SETTINGS_E2E_PROFILE']
      const configPath = profileRoot ? path.join(profileRoot, 'database-config.json') : null
      const originalConfig = configPath ? await readFile(configPath, 'utf8') : null
      await mkdir(invalidPath, { recursive: true })

      try {
        await page.goto('/settings')
        await expect(page.getByRole('heading', { name: 'Database', exact: true })).toBeVisible()
        await expect(page.getByText('SQLITE · Connected', { exact: true })).toBeVisible()
        await expect(page.getByLabel('Log level')).toHaveValue(initialSettings.log_level)

        const savePrimary = page.waitForResponse((response) => response.url().endsWith('/api/settings') && response.request().method() === 'PATCH')
        await page.getByLabel('Log level').selectOption('WARNING')
        expect((await savePrimary).status()).toBe(200)
        await expect(page.getByRole('status').filter({ hasText: 'Changes saved automatically.' })).toBeVisible()

        await page.getByLabel('Database file path').fill(invalidPath)
        const failedTest = page.waitForResponse((response) => response.url().endsWith('/api/settings/database/test') && response.request().method() === 'POST')
        await page.getByRole('button', { name: 'Test connection', exact: true }).click()
        expect((await failedTest).status()).toBe(422)
        await expect(page.getByRole('alert')).toContainText('Connection test failed')

        const failedSwitch = page.waitForResponse((response) => response.url().endsWith('/api/settings/database') && response.request().method() === 'PUT')
        await page.getByRole('button', { name: 'Confirm and switch', exact: true }).click()
        expect((await failedSwitch).status()).toBe(422)
        const statusAfterFailedSwitch = await request.get('/api/settings/database')
        expect((await statusAfterFailedSwitch.json()).sqlite_path).toBe(initialStatus.sqlite_path)
        if (configPath && originalConfig !== null) expect(await readFile(configPath, 'utf8')).toBe(originalConfig)

        await page.getByLabel('Database file path').fill(secondaryPath)
        const successfulTest = page.waitForResponse((response) => response.url().endsWith('/api/settings/database/test') && response.request().method() === 'POST')
        await page.getByRole('button', { name: 'Test connection', exact: true }).click()
        expect((await successfulTest).status()).toBe(200)
        const statusBeforeConfirmation = await request.get('/api/settings/database')
        expect((await statusBeforeConfirmation.json()).sqlite_path).toBe(initialStatus.sqlite_path)
        if (configPath && originalConfig !== null) expect(await readFile(configPath, 'utf8')).toBe(originalConfig)
        await expect(page.getByText('SQLite connection and required schema are ready.', { exact: true })).toBeVisible()
        await page.screenshot({ path: path.join(evidenceRoot, 'settings-database-tested.png'), fullPage: true })

        const switchToSecondary = page.waitForResponse((response) => response.url().endsWith('/api/settings/database') && response.request().method() === 'PUT')
        await page.getByRole('button', { name: 'Confirm and switch', exact: true }).click()
        expect((await switchToSecondary).status()).toBe(200)
        await expect(page.getByText(secondaryPath, { exact: true })).toBeVisible()
        if (configPath) expect(JSON.parse(await readFile(configPath, 'utf8')).sqlite_path).toBe(secondaryPath)
        await expect(page.getByLabel('Log level')).toHaveValue('INFO')
        await expect(page.getByRole('status').filter({ hasText: 'Database switched.' })).toBeVisible()

        await page.getByLabel('Log level').selectOption('DEBUG')
        const saveSecondary = page.waitForResponse((response) => response.url().endsWith('/api/settings') && response.request().method() === 'PATCH')
        expect((await saveSecondary).status()).toBe(200)
        await expect(page.getByLabel('Log level')).toHaveValue('DEBUG')
        await page.screenshot({ path: path.join(evidenceRoot, 'settings-database-switched.png'), fullPage: true })

        await page.reload()
        await expect(page.getByText(secondaryPath, { exact: true })).toBeVisible()
        await expect(page.getByLabel('Log level')).toHaveValue('DEBUG')
        await page.screenshot({ path: path.join(evidenceRoot, 'settings-database-reloaded.png'), fullPage: true })

        await page.getByLabel('Database file path').fill(initialStatus.sqlite_path)
        const switchBack = page.waitForResponse((response) => response.url().endsWith('/api/settings/database') && response.request().method() === 'PUT')
        await page.getByRole('button', { name: 'Confirm and switch', exact: true }).click()
        expect((await switchBack).status()).toBe(200)
        await expect(page.getByText(initialStatus.sqlite_path, { exact: true })).toBeVisible()
        if (configPath) expect(JSON.parse(await readFile(configPath, 'utf8')).sqlite_path).toBe(initialStatus.sqlite_path)
        await expect(page.getByLabel('Log level')).toHaveValue('WARNING')
        await page.getByLabel('Log level').selectOption(initialSettings.log_level)
        const restorePrimary = page.waitForResponse((response) => response.url().endsWith('/api/settings') && response.request().method() === 'PATCH')
        expect((await restorePrimary).status()).toBe(200)
      } finally {
        const status = await request.get('/api/settings/database')
        if (status.ok()) {
          const current = await status.json() as { sqlite_path: string }
          if (current.sqlite_path !== initialStatus.sqlite_path) {
            await request.put('/api/settings/database', {
              data: { backend: 'sqlite', schema_version: 1, sqlite_path: initialStatus.sqlite_path, connect_timeout_seconds: 30 },
            })
          }
        }
        for (const suffix of ['', '-wal', '-shm', '-journal']) {
          await rm(`${secondaryPath}${suffix}`, { force: true })
        }
        await rm(invalidPath, { recursive: true, force: true })
      }
    })

    test('renders the selected database and its settings from the active backend state', async ({ page, request }) => {
      const databaseResponse = await request.get('/api/settings/database')
      const settingsResponse = await request.get('/api/settings')
      expect(databaseResponse.ok()).toBeTruthy()
      expect(settingsResponse.ok()).toBeTruthy()
      const database = await databaseResponse.json() as { backend: string; sqlite_path: string; available: boolean }
      const settings = await settingsResponse.json() as { log_level: string }

      await page.goto('/settings')
      await expect(page.getByText(`${database.backend.toUpperCase()} · Connected`, { exact: true })).toBeVisible()
      await expect(page.getByText(database.sqlite_path, { exact: true })).toBeVisible()
      await expect(page.getByLabel('Log level')).toHaveValue(settings.log_level)
      expect(database.available).toBe(true)
      await page.screenshot({ path: path.join(evidenceRoot, 'settings-database-after-backend-restart.png'), fullPage: true })
    })
  })
  test.describe('settings autosave.spec', () => {
    const repositoryRoot = path.resolve(process.cwd(), '..', '..')
    const evidenceRoot = path.resolve(
      process.env.NW_SETTINGS_AUTOSAVE_EVIDENCE_DIR
        ?? path.join(repositoryRoot, 'assets', 'QA', 'settings-autosave-ui-e2e-20261021'),
    )
    type SettingsRecord = Record<string, unknown>

    async function currentSettings(request: import('@playwright/test').APIRequestContext): Promise<SettingsRecord> {
      const response = await request.get('/api/settings')
      expect(response.ok()).toBeTruthy()
      return response.json() as Promise<SettingsRecord>
    }

    test.describe.configure({ mode: 'serial' })

    let settingsBaseline: SettingsRecord | null = null

    test.beforeEach(async ({ request }) => {
      settingsBaseline = await currentSettings(request)
    })

    test.afterEach(async ({ request }) => {
      if (!settingsBaseline) return
      const response = await request.patch('/api/settings', { data: settingsBaseline })
      expect(response.ok(), 'restore settings after the autosave test').toBeTruthy()
      settingsBaseline = null
    })

    test('automatically saves every ordinary application settings control', async ({ page, request }) => {
      await mkdir(evidenceRoot, { recursive: true })
      await page.goto('/settings')
      await expect(page.getByRole('heading', { name: 'General', exact: true })).toBeVisible()
      await expect(page.getByRole('button', { name: /save application settings|apply|cancel/i })).toHaveCount(0)

      const original = await currentSettings(request)
      expect(original).not.toHaveProperty('cors_origins')
      await expect(page.getByLabel('Allowed browser origins')).toHaveCount(0)
      const nextNumber = (field: string, minimum: number, maximum: number): number => {
        const current = Number(original[field])
        return current < maximum ? Math.min(maximum, current + 1) : Math.max(minimum, current - 1)
      }
      const uniquePath = (name: string): string => path.join(evidenceRoot, `${process.pid}-${name}`)
      const values: Array<{ label: string; field: string; value: string | number; select?: boolean }> = [
        { label: 'Log level', field: 'log_level', value: original['log_level'] === 'DEBUG' ? 'INFO' : 'DEBUG', select: true },
        { label: 'Worker timeout (seconds)', field: 'ml_worker_timeout_seconds', value: nextNumber('ml_worker_timeout_seconds', 1, 86400) },
        { label: 'Runtime discovery timeout (seconds)', field: 'runtime_discovery_timeout_seconds', value: nextNumber('runtime_discovery_timeout_seconds', 1, 600) },
        { label: 'Tensor inspection retention (seconds)', field: 'tensor_inspection_ttl_seconds', value: nextNumber('tensor_inspection_ttl_seconds', 60, 86400) },
        { label: 'Cancel grace period (seconds)', field: 'ml_cancel_grace_seconds', value: nextNumber('ml_cancel_grace_seconds', 1, 300) },
        { label: 'Terminate grace period (seconds)', field: 'ml_terminate_grace_seconds', value: nextNumber('ml_terminate_grace_seconds', 1, 300) },
        { label: 'Job history limit', field: 'ml_job_history_limit', value: nextNumber('ml_job_history_limit', 10, 1000) },
        { label: 'Log file', field: 'log_file', value: uniquePath('backend.log') },
        { label: 'Export folder', field: 'export_root', value: uniquePath('exports') },
        { label: 'ML job folder', field: 'ml_job_root', value: uniquePath('jobs') },
        { label: 'Tensor inspection folder', field: 'tensor_inspection_root', value: uniquePath('inspections') },
        { label: 'Runtime discovery cache', field: 'runtime_discovery_path', value: path.join(repositoryRoot, 'runtimes', 'cache', `settings-autosave-discovery-${process.pid}.json`) },
      ]

      for (const item of values) {
        if (item.field === 'log_file') await page.getByText('Storage and log locations', { exact: true }).click()
        const responsePromise = page.waitForResponse((response) => response.url().endsWith('/api/settings') && response.request().method() === 'PATCH')
        const control = page.getByLabel(item.label)
        if (item.select) await control.selectOption(String(item.value))
        else await control.fill(String(item.value))
        const response = await responsePromise
        expect(response.status(), `${item.label} should save successfully`).toBe(200)
        const saved = await response.json() as SettingsRecord
        const expected = item.field.endsWith('_root') || item.field === 'log_file' || item.field === 'runtime_discovery_path'
          ? path.resolve(String(item.value))
          : item.value
        expect(saved[item.field], `${item.label} should be applied immediately`).toBe(expected)
        expect((await currentSettings(request))[item.field], `${item.label} should persist in the active database`).toBe(expected)
      }

      await expect(page.getByRole('status').filter({ hasText: 'Changes saved automatically.' })).toBeVisible()
      await page.locator('main.app-content').evaluate((element) => element.scrollTo(0, 0))
      await page.screenshot({ path: path.join(evidenceRoot, 'settings-general-autosave.png'), fullPage: true })
      await page.reload()
      await page.getByText('Storage and log locations', { exact: true }).click()
      const restored = await currentSettings(request)
      for (const item of values) {
        const expected = item.field.endsWith('_root') || item.field === 'log_file' || item.field === 'runtime_discovery_path'
          ? path.resolve(String(item.value))
          : item.value
        expect(restored[item.field], `${item.label} should restore after reload`).toBe(expected)
        await expect(page.getByLabel(item.label)).toHaveValue(String(expected))
      }
    })

    test('debounces rapid text edits and never lets a stale value win', async ({ page, request }) => {
      await page.goto('/settings')
      const original = await currentSettings(request)
      await page.getByText('Storage and log locations', { exact: true }).click()
      const input = page.getByLabel('Log file')
      const patchBodies: string[] = []
      page.on('request', (requestEvent) => {
        if (requestEvent.url().endsWith('/api/settings') && requestEvent.method() === 'PATCH') {
          patchBodies.push(requestEvent.postData() ?? '')
        }
      })

      const intermediateValue = path.join(evidenceRoot, `${process.pid}-intermediate.log`)
      const finalValue = path.join(evidenceRoot, `${process.pid}-debounced.log`)
      await input.fill(intermediateValue)
      const finalSave = page.waitForResponse((response) => response.url().endsWith('/api/settings') && response.request().method() === 'PATCH')
      await input.fill(finalValue)
      expect((await finalSave).status()).toBe(200)
      await expect.poll(async () => (await currentSettings(request))['log_file']).toBe(path.resolve(finalValue))
      expect(patchBodies).toHaveLength(1)
      expect(JSON.parse(patchBodies[0])['log_file']).toBe(finalValue)
      expect(original['log_file']).not.toBe(finalValue)
    })

    test('reverts a failed application settings write and explains the failure', async ({ page, request }) => {
      const initial = await currentSettings(request)
      await page.route('**/api/settings', async (route) => {
        if (route.request().method() === 'PATCH') {
          await route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'injected settings write failure' }) })
          return
        }
        await route.continue()
      })
      await page.goto('/settings')
      await page.getByText('Storage and log locations', { exact: true }).click()
      const input = page.getByLabel('Log file')
      await input.fill(path.join(evidenceRoot, `${process.pid}-failed.log`))
      await expect(page.getByRole('alert')).toContainText('injected settings write failure')
      await expect(input).toHaveValue(String(initial['log_file']))
      expect((await currentSettings(request))['log_file']).toBe(initial['log_file'])
    })
  })
  test.describe('settings training.spec', () => {
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

    test('NW-T1-05 supports UI settings and keeps Training session-only', async ({ page, request }) => {
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
        await expect(page.getByRole('heading', { name: 'General', exact: true })).toBeVisible()
        await expect(page.getByRole('heading', { name: 'Database', exact: true })).toBeVisible()
        await expect(page.getByLabel('Log level')).toHaveValue('INFO')
        await expect(page.getByLabel('Database file path')).toBeVisible()
        await expect(page.getByRole('button', { name: 'Test connection' })).toBeVisible()
        await expect(page.getByRole('button', { name: 'Confirm and switch' })).toBeVisible()
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
  })
