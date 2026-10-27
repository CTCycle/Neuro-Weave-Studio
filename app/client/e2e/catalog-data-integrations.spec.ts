// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import fs, { access, mkdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

  test.describe('custom node templates.spec', () => {
    const evidenceDir = path.resolve(
      process.env.NW_CUSTOM_NODE_EVIDENCE_DIR
        ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260926', 'evidence', 'nw-t2-09'),
    )

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

    async function ensurePyTorchKeras(page: Page, request: APIRequestContext): Promise<void> {
      await page.goto('/environment')
      const pytorch = page.locator('.backend-status-card-container').filter({ hasText: 'PyTorch' }).first()
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
        const kerasCheck = pytorch.getByRole('button', { name: 'Check Keras', exact: true })
        await expect(kerasCheck).toBeVisible({ timeout: 15_000 })
        await kerasCheck.click()
      }
      await expect.poll(kerasRuntimeReady, { timeout: 90_000 }).toBeTruthy()
      await expect(pytorch).toContainText('Usable')
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

      await ensurePyTorchKeras(page, request)
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
  })
  test.describe('datasets.spec', () => {
    const fixtureRoot = path.resolve(process.env.NEURO_WEAVE_E2E_FIXTURE_DIR ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA'))
    const evidenceRoot = path.resolve(process.env.NW_DATASET_EVIDENCE_DIR ?? path.join(path.resolve(process.cwd(), '..', '..'), 'assets', 'QA', 'dataset-workspace-e2e'))
    const fixturePath = path.join(fixtureRoot, `dataset-registration-${process.pid}.csv`)
    const datasetName = `E2E patient outcomes ${process.pid}`

    test.describe.configure({ mode: 'serial' })

    function trackDatasetPosts(page: Page): string[] {
      const requests: string[] = []
      page.on('request', (request) => {
        const pathname = new URL(request.url()).pathname
        if (request.method() === 'POST' && ['/api/datasets', '/api/datasets/inspect', '/api/datasets/validate'].includes(pathname)) requests.push(`${request.method()} ${pathname}`)
      })
      return requests
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

    async function getPersistedDataset(request: APIRequestContext, id: string): Promise<Record<string, unknown>> {
      const response = await request.get(`/api/datasets/${id}`)
      expect(response.ok(), 'read persisted dataset definition').toBeTruthy()
      return await response.json() as Record<string, unknown>
    }

    test('registers, configures, reloads, and deletes a tabular dataset without deleting its source', async ({ page, request }) => {
      test.setTimeout(120_000)
      let datasetId: string | undefined
      const datasetPosts = trackDatasetPosts(page)
      const datasetUpdates: string[] = []
      const jobOrMaterializationWrites: string[] = []
      page.on('request', (request) => {
        const pathname = new URL(request.url()).pathname
        if (request.method() === 'PUT' && /^\/api\/datasets\/[^/]+$/.test(pathname)) datasetUpdates.push(pathname)
        if (request.method() === 'POST' && /\/api\/.*(?:job|partition|materializ)/i.test(pathname)) jobOrMaterializationWrites.push(`${request.method()} ${pathname}`)
      })
      await mkdir(fixtureRoot, { recursive: true })
      await mkdir(evidenceRoot, { recursive: true })
      await page.setViewportSize({ width: 1440, height: 900 })
      const sourceContents = 'age,weight,diagnosis\n42,71,positive\n50,80,negative\n61,90,positive\n'
      const replacementPath = path.join(fixtureRoot, `dataset-registration-replacement-${process.pid}.csv`)
      const invalidReplacementPath = path.join(fixtureRoot, `dataset-registration-invalid-replacement-${process.pid}.csv`)
      const replacementContents = 'age;weight;diagnosis\n42;71;positive\n50;80;negative\n61;90;positive\n70;95;negative\n'
      await writeFile(fixturePath, sourceContents, 'utf8')
      await writeFile(replacementPath, replacementContents, 'utf8')
      await writeFile(invalidReplacementPath, 'age;diagnosis\n42;positive\n50;negative\n61;positive\n70;negative\n', 'utf8')

      try {
        await prepareTabularWizard(page, fixturePath, datasetName)
        const createResponse = page.waitForResponse((response) => response.url().endsWith('/api/datasets') && response.request().method() === 'POST')
        await page.locator('nw-dataset-wizard').getByRole('button', { name: 'Register dataset' }).click()
        const created = await createResponse
        expect(created.status()).toBe(201)
        datasetId = (await created.json()).id as string

        const card = page.locator('.dataset-card').filter({ hasText: datasetName })
        await expect(card).toBeVisible()
        await expect(card).toContainText('3')
        await card.getByRole('link').click()
        await expect(page).toHaveURL(new RegExp(`/datasets/${datasetId}$`))
        await expect(page.getByRole('heading', { name: datasetName })).toBeVisible()
        for (const tab of ['Overview', 'Validation', 'I/O', 'Processing', 'Split']) await expect(page.getByRole('link', { name: tab, exact: true })).toBeVisible()

        await page.goto(`/datasets/${datasetId}/io`)
        await expect(page).toHaveURL(new RegExp(`/datasets/${datasetId}/io$`))

        await page.getByRole('link', { name: 'Validation', exact: true }).click()
        await page.getByRole('button', { name: 'Revalidate' }).click()
        await expect(page.locator('.validation-page')).toContainText('Valid')
        await writeFile(fixturePath, 'age,diagnosis\n42,positive\n50,negative\n61,positive\n', 'utf8')
        await page.getByRole('button', { name: 'Revalidate' }).click()
        await expect(page.locator('.validation-page')).toContainText('Invalid')
        await expect(page.locator('.validation-page')).toContainText('missing_columns')
        await writeFile(fixturePath, sourceContents, 'utf8')
        await page.getByRole('button', { name: 'Revalidate' }).click()
        await expect(page.locator('.validation-page')).toContainText('Valid')

        await page.getByRole('link', { name: 'I/O', exact: true }).click()
        await page.locator('.role-row').filter({ hasText: /^diagnosis/ }).getByRole('combobox').selectOption('output')
        await page.getByRole('button', { name: 'Save I/O configuration' }).click()
        await expect(page.getByRole('status')).toContainText('I/O configuration saved')
        await page.reload()
        await expect(page.locator('.role-row').filter({ hasText: /^diagnosis/ }).getByRole('combobox')).toHaveValue('output')

        await page.getByRole('link', { name: 'Processing', exact: true }).click()
        await expect(page.getByRole('heading', { name: 'Processing and interpretation' })).toBeVisible()
        const sourceCard = page.locator('.source-card').first()
        await sourceCard.getByRole('button', { name: 'Browse…' }).click()
        await selectFileFromBrowser(page, replacementPath)
        await sourceCard.getByLabel('Delimiter').selectOption(';')
        const replacementSave = page.waitForResponse((response) => response.url().endsWith(`/api/datasets/${datasetId}`) && response.request().method() === 'PUT')
        await page.getByRole('button', { name: 'Save and revalidate' }).click()
        expect((await replacementSave).ok(), 'save valid source replacement').toBeTruthy()
        await expect(page.getByRole('status')).toContainText('Source processing configuration saved after fresh inspection.')
        await page.reload()
        await expect(page.locator('.source-card').first().locator('input').first()).toHaveValue(replacementPath)
        await expect(page.locator('.source-card').first().getByLabel('Delimiter')).toHaveValue(';')
        await page.screenshot({ path: path.join(evidenceRoot, 'dataset-processing-relink-persisted.png') })
        let persisted = await getPersistedDataset(request, datasetId)
        expect(persisted).toMatchObject({
          sources: [{ path: replacementPath, delimiter: ';' }],
          sample_count: 4,
          output_definition: { bindings: [{ columns: ['diagnosis'] }] },
        })

        await page.locator('.source-card').first().getByRole('button', { name: 'Browse…' }).click()
        await selectFileFromBrowser(page, invalidReplacementPath)
        await page.locator('.source-card').first().getByLabel('Delimiter').selectOption(';')
        const updatesBeforeRejectedReplacement = datasetUpdates.length
        const invalidInspection = page.waitForResponse((response) => response.url().endsWith('/api/datasets/validate') && response.request().method() === 'POST')
        await page.getByRole('button', { name: 'Save and revalidate' }).click()
        expect((await invalidInspection).ok(), 'invalid replacement returns structured validation').toBeTruthy()
        await expect(page.getByRole('alert')).toContainText('The persisted definition was not changed.')
        await expect(page.locator('.inline-validation')).toContainText('missing_columns')
        await page.screenshot({ path: path.join(evidenceRoot, 'dataset-processing-invalid-replacement-rejected.png') })
        expect(datasetUpdates).toHaveLength(updatesBeforeRejectedReplacement)
        persisted = await getPersistedDataset(request, datasetId)
        expect(persisted).toMatchObject({ sources: [{ path: replacementPath, delimiter: ';' }], sample_count: 4 })
        await page.reload()
        await expect(page.locator('.source-card').first().locator('input').first()).toHaveValue(replacementPath)
        await expect(page.locator('.source-card').first().getByLabel('Delimiter')).toHaveValue(';')

        await page.locator('.source-card').first().getByRole('button', { name: 'Browse…' }).click()
        await selectFileFromBrowser(page, replacementPath)
        await page.locator('.source-card').first().getByLabel('Delimiter').selectOption(';')
        const restoredSave = page.waitForResponse((response) => response.url().endsWith(`/api/datasets/${datasetId}`) && response.request().method() === 'PUT')
        await page.getByRole('button', { name: 'Save and revalidate' }).click()
        expect((await restoredSave).ok(), 're-save valid source after rejected replacement').toBeTruthy()
        await expect(page.getByRole('status')).toContainText('Source processing configuration saved after fresh inspection.')
        await page.getByRole('link', { name: 'I/O', exact: true }).click()
        await expect(page.locator('.role-row').filter({ hasText: /^diagnosis/ }).getByRole('combobox')).toHaveValue('output')

        await page.getByRole('link', { name: 'Split', exact: true }).click()
        await page.locator('.population-option').filter({ hasText: 'First N' }).locator('input').check()
        await page.locator('.number-field input').first().fill('2')
        await page.getByLabel('Enable split').check()
        await page.locator('.number-field input').first().fill('5')
        await expect(page.locator('.field-error')).toContainText('First N must be between 1 and 4.')
        await expect(page.getByRole('button', { name: 'Save population and split' })).toBeDisabled()
        await page.locator('.number-field input').first().fill('2')
        const fractionInputs = page.locator('.fraction-grid input')
        await fractionInputs.nth(0).fill('0.8')
        await fractionInputs.nth(1).fill('0.3')
        await fractionInputs.nth(2).fill('0.1')
        await expect(page.locator('.fraction-total')).toContainText('must sum to 1')
        await expect(page.getByRole('button', { name: 'Save population and split' })).toBeDisabled()
        await fractionInputs.nth(0).fill('0.5')
        await fractionInputs.nth(1).fill('0.25')
        await fractionInputs.nth(2).fill('0.25')
        await page.locator('.option-grid input[type="number"]').fill('17')
        await expect(page.getByRole('button', { name: 'Save population and split' })).toBeEnabled()
        await page.getByRole('button', { name: 'Save population and split' }).click()
        await expect(page.getByRole('status')).toContainText('Population and split configuration saved')
        await page.reload()
        await expect(page.locator('.population-panel')).toContainText('2 selected of 4')
        await expect(page.locator('.split-panel')).toContainText('Random')
        await page.screenshot({ path: path.join(evidenceRoot, 'dataset-split-configuration-persisted.png') })
        persisted = await getPersistedDataset(request, datasetId)
        expect(persisted).toMatchObject({
          subset_definition: { type: 'first_n', count: 2 },
          split_definition: { strategy: 'random', train_fraction: 0.5, validation_fraction: 0.25, test_fraction: 0.25, seed: 17 },
        })

        await page.getByRole('link', { name: 'All datasets' }).click()
        const catalogCard = page.locator('.dataset-card').filter({ hasText: datasetName })
        await catalogCard.getByRole('button', { name: `Delete registration for ${datasetName}` }).click()
        await page.getByRole('button', { name: 'Delete registration', exact: true }).click()
        await expect(catalogCard).toHaveCount(0)
        await access(fixturePath)
        await access(replacementPath)
        await access(invalidReplacementPath)
        expect(datasetPosts).toEqual([
          'POST /api/datasets/inspect', 'POST /api/datasets/validate', 'POST /api/datasets',
          'POST /api/datasets/inspect', 'POST /api/datasets/validate',
          'POST /api/datasets/inspect', 'POST /api/datasets/validate',
          'POST /api/datasets/inspect', 'POST /api/datasets/validate',
        ])
        expect(datasetUpdates).toHaveLength(4)
        expect(jobOrMaterializationWrites).toEqual([])
      } finally {
        if (datasetId) {
          const response = await request.delete(`/api/datasets/${datasetId}`)
          expect([204, 404]).toContain(response.status())
        }
        await rm(fixturePath, { force: true })
        await rm(replacementPath, { force: true })
        await rm(invalidReplacementPath, { force: true })
      }
    })

    test('returns authoritative registration findings to Inspect and confirm when the source changes', async ({ page }) => {
      test.setTimeout(120_000)
      const invalidFixturePath = path.join(fixtureRoot, `dataset-registration-invalid-${process.pid}.csv`)
      const invalidDatasetName = `E2E invalid patient outcomes ${process.pid}`
      const datasetPosts = trackDatasetPosts(page)
      await mkdir(fixtureRoot, { recursive: true })
      await writeFile(invalidFixturePath, 'age,weight,diagnosis\n42,71,positive\n', 'utf8')

      try {
        await prepareTabularWizard(page, invalidFixturePath, invalidDatasetName)
        await writeFile(invalidFixturePath, 'age,diagnosis\n42,positive\n', 'utf8')
        const rejectedResponse = page.waitForResponse((response) => response.url().endsWith('/api/datasets') && response.request().method() === 'POST')
        await page.locator('nw-dataset-wizard').getByRole('button', { name: 'Register dataset' }).click()
        const rejected = await rejectedResponse

        expect(rejected.status()).toBe(422)
        await expect(page.getByRole('heading', { name: 'Inspect and confirm' })).toBeVisible()
        await expect(page.locator('.validation-result')).toContainText('Definition needs attention')
        await expect(page.locator('.validation-result')).toContainText('missing_columns')
        await expect(page.locator('.validation-result')).toContainText('Columns do not exist: weight')
        expect(datasetPosts).toEqual(['POST /api/datasets/inspect', 'POST /api/datasets/validate', 'POST /api/datasets'])
      } finally {
        await rm(invalidFixturePath, { force: true })
      }
    })
  })
  test.describe('repository network.spec', () => {
    test.skip(
      process.env.NEURO_WEAVE_RUN_LIVE_REPOSITORY_E2E !== '1',
      'Opt-in: exercises the Nodes UI against the live Hugging Face repository resolver.',
    )

    test('verifies and displays a live public repository node without executing repository code', async ({ page, request }) => {
      const displayName = `Live repository reference ${Date.now()}`
      let definitionId: string | undefined
      try {
        await page.goto('/nodes')
        await page.getByRole('button', { name: 'Add custom node' }).click()
        const dialog = page.getByRole('dialog')
        await dialog.getByRole('button', { name: 'Public repository' }).click()
        await dialog.getByRole('button', { name: 'Continue' }).click()
        await dialog.getByLabel('Name').fill(displayName)
        await dialog.getByLabel('Category').fill('repository')
        await dialog.getByLabel('Description').fill('Live repository reference regression.')
        await dialog.getByRole('button', { name: 'Continue' }).click()
        await dialog.getByLabel('Repository id').fill('openai-community/gpt2')
        await dialog.getByLabel('Revision').fill('main')

        const createResponse = page.waitForResponse((response) => response.url().endsWith('/api/nodes/custom') && response.request().method() === 'POST')
        await dialog.getByRole('button', { name: 'Add custom node' }).click()
        const response = await createResponse
        expect(response.status()).toBe(201)
        const created = await response.json() as { id: string }
        definitionId = created.id

        await expect(page.getByRole('heading', { name: displayName })).toBeVisible()
        const facts = page.locator('.repository-facts')
        await expect(facts).toContainText('openai-community/gpt2')
        await expect(facts.locator('div').filter({ hasText: 'Resolved commit' }).locator('code')).toHaveText(/^[0-9a-f]{40}$/)
        const repositorySection = page.locator('.detail-section').filter({ hasText: 'Verified public repository reference' })
        await expect(repositorySection.getByText('Reference only', { exact: true })).toBeVisible()
        await expect(page.getByText('repository code is never imported or executed', { exact: false })).toBeVisible()
      } finally {
        if (definitionId) {
          const response = await request.delete(`/api/nodes/custom/${definitionId}`)
          expect([204, 404]).toContain(response.status())
        }
      }
    })
  })
