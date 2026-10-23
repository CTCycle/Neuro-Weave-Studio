// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { createHash } from 'node:crypto'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type APIRequestContext, type Page, type TestInfo, type Locator } from '@playwright/test'

  test.describe('advanced convolution slice.spec', () => {
    type CreatedProject = { id: string; current_revision: number; name: string }
    type GraphNode = {
      id: string
      definition_id: string
      definition_version: number
      label: string
      position: { x: number; y: number }
      parameters: Record<string, unknown>
    }
    type AdvancedCase = {
      name: string
      inputDefinition: 'input.sequence' | 'input.image' | 'input.tensor'
      inputShape: number[]
      catalogNode: string
      staticOutputShape: Array<number | null>
      runtimeOutputShape: number[]
      parameterCount: number
      kerasEnabled?: boolean
    }
    type ShapeSummary = {
      node_id: string
      parameter_count: number
      outputs: Record<string, { shape: Array<number | null> }>
    }
    type ValidationPayload = {
      valid: boolean
      diagnostics: unknown[]
      shape_summaries: ShapeSummary[]
      total_parameters: number
    }
    type RuntimeJob = {
      state: string
      worker?: { api: string; backend: string; device_id: string }
      result?: {
        status: string
        inputs: Array<{ shape: number[] }>
        outputs: Array<{ shape: number[] }>
        parameter_count: number
      }
    }

    const evidenceDir = path.resolve(
      process.env.NW_VALIDATION_EVIDENCE_DIR
        ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260928', 'evidence'),
    )

    function graphShell(nodes: GraphNode[]) {
      return {
        schema_version: 6,
        nodes,
        edges: [],
        model_configuration: {},
        export_settings: {
          include_comments: true,
          include_shape_assertions: true,
          preferred_binary_format: 'keras',
        },
        future_training_configuration: {},
      }
    }

    function node(
      id: string,
      definitionId: string,
      label: string,
      position: { x: number; y: number },
      parameters: Record<string, unknown>,
    ): GraphNode {
      return {
        id,
        definition_id: definitionId,
        definition_version: 1,
        label,
        position,
        parameters,
      }
    }

    function inputOutputGraph(inputDefinition: AdvancedCase['inputDefinition'], inputShape: number[]) {
      return graphShell([
        node('matrix-input', inputDefinition, 'Matrix input', { x: 64, y: 180 }, { shape: inputShape, dtype: 'float32' }),
        node('matrix-output', 'output.custom', 'Matrix output', { x: 1200, y: 180 }, {}),
      ])
    }

    async function createProject(
      request: APIRequestContext,
      name: string,
      modelGraph: ReturnType<typeof graphShell>,
      kerasEnabled: boolean,
    ): Promise<CreatedProject> {
      const response = await request.post('/api/projects', {
        data: {
          name,
          description: 'NW-T3-04 advanced and spatial convolution validation fixture',
          execution_backend: 'pytorch',
          keras_enabled: kerasEnabled,
          graph: modelGraph,
        },
      })
      expect(response.ok()).toBeTruthy()
      return response.json() as Promise<CreatedProject>
    }

    async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
      const response = await request.delete(`/api/projects/${id}`)
      expect([204, 404]).toContain(response.status())
    }

    async function addCatalogNode(page: Page, displayName: string): Promise<string> {
      const nodes = page.locator('[data-node-id]')
      const before = new Set(
        await nodes.evaluateAll((elements) => elements
          .map((element) => element.getAttribute('data-node-id'))
          .filter(Boolean) as string[]),
      )
      await page.getByRole('textbox', { name: 'Search nodes' }).fill(displayName)
      const item = page.locator('.catalog-item').filter({ hasText: displayName }).first()
      await expect(item, `Catalog node ${displayName} should be available`).toBeEnabled()
      await item.click()
      await expect(nodes).toHaveCount(before.size + 1)
      const after = await nodes.evaluateAll((elements) => elements
        .map((element) => element.getAttribute('data-node-id'))
        .filter(Boolean) as string[])
      const created = after.find((nodeId) => !before.has(nodeId))
      expect(created, `Builder should materialize ${displayName}`).toBeTruthy()
      await page.getByRole('textbox', { name: 'Search nodes' }).fill('')
      return created!
    }

    async function connectByLabels(
      page: Page,
      sourceLabel: string,
      targetLabel: string,
      expectedConnectionCount: number,
    ): Promise<void> {
      const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
      const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
      const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
      const target = targetNode.getByRole('button', { name: /^Port Input \(input\)/ })
      await expect(source).toBeVisible()
      await expect(target).toBeVisible()
      await source.press('Enter')
      await expect(source).toHaveAttribute('aria-pressed', 'true')
      const responsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation/connection') && response.request().method() === 'POST',
      )
      await target.press('Enter')
      const response = await responsePromise
      expect(response.ok()).toBeTruthy()
      const payload = await response.json() as { valid: boolean }
      expect(payload.valid, `${sourceLabel} → ${targetLabel} should be accepted`).toBe(true)
      await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
    }

    async function validateSaveAndRun(
      page: Page,
      request: APIRequestContext,
      fixture: CreatedProject,
      advancedNodeId: string,
      expected: AdvancedCase,
      testInfo: TestInfo,
      browserErrors: string[],
    ): Promise<void> {
      const validationResponsePromise = page.waitForResponse((response) => {
        if (!response.url().endsWith('/api/validation') || response.request().method() !== 'POST') return false
        const body = response.request().postDataJSON() as {
          graph?: { nodes?: Array<{ id?: string }>; edges?: unknown[] }
        }
        return body.graph?.nodes?.some((node) => node.id === advancedNodeId) === true
          && body.graph?.edges?.length === 2
      })
      await page.getByRole('button', { name: 'Validate', exact: true }).click()
      const validationResponse = await validationResponsePromise
      expect(validationResponse.ok()).toBeTruthy()
      const validationPayload = await validationResponse.json() as ValidationPayload
      expect(validationPayload.valid).toBe(true)
      expect(validationPayload.diagnostics).toEqual([])
      const summary = validationPayload.shape_summaries.find((item) => item.node_id === advancedNodeId)
      expect(summary?.outputs.output?.shape).toEqual(expected.staticOutputShape)
      expect(summary?.parameter_count).toBe(expected.parameterCount)
      expect(validationPayload.total_parameters).toBe(expected.parameterCount)
      await expect(page.locator('.validation-summary')).toContainText('Ready')
      await page.screenshot({ path: testInfo.outputPath(`${expected.name}-builder-ready.png`), fullPage: true })

      const saveResponsePromise = page.waitForResponse(
        (response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT',
      )
      await page.getByRole('button', { name: 'Save model', exact: true }).click()
      const saveResponse = await saveResponsePromise
      expect(saveResponse.ok()).toBeTruthy()
      const saved = await saveResponse.json() as {
        current_revision: number
        validation_status: string
        graph: { nodes: unknown[]; edges: unknown[] }
      }
      expect(saved.validation_status).toBe('valid')
      expect(saved.graph.nodes).toHaveLength(3)
      expect(saved.graph.edges).toHaveLength(2)

      await page.reload()
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
      await expect(page.locator('[data-node-id]')).toHaveCount(3)
      await mkdir(evidenceDir, { recursive: true })
      await page.screenshot({ path: path.join(evidenceDir, `${expected.name}-builder-ready.png`), fullPage: true })

      await page.goto(`/models/${fixture.id}/validation`)
      await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
      const profilePattern = expected.kerasEnabled === false ? /native.*pytorch/i : /keras.*pytorch/i
      await expect(page.getByText(profilePattern).first()).toBeVisible({ timeout: 75_000 })
      const submitResponsePromise = page.waitForResponse(
        (response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
      const submitResponse = await submitResponsePromise
      expect(submitResponse.ok()).toBeTruthy()
      const submitted = await submitResponse.json() as { id: string }
      await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 90_000 })

      let finalJob: RuntimeJob = { state: 'missing' }
      await expect.poll(async () => {
        const response = await request.get(`/api/execution/jobs/${submitted.id}`)
        if (!response.ok()) return 'missing'
        finalJob = await response.json() as RuntimeJob
        return finalJob.state
      }, { timeout: 90_000 }).toBe('succeeded')

      expect(finalJob.worker).toMatchObject({
        api: expected.kerasEnabled === false ? 'native' : 'keras',
        backend: 'pytorch',
        device_id: 'cpu',
      })
      expect(finalJob.result?.status).toBe('succeeded')
      expect(finalJob.result?.parameter_count).toBe(expected.parameterCount)
      expect(finalJob.result?.outputs[0]?.shape).toEqual(expected.runtimeOutputShape)
      expect(browserErrors, `${expected.name} should not emit browser errors`).toEqual([])
      await page.screenshot({ path: testInfo.outputPath(`${expected.name}-runtime-succeeded.png`), fullPage: true })
      await page.screenshot({ path: path.join(evidenceDir, `${expected.name}-runtime-succeeded.png`), fullPage: true })
      await writeFile(path.join(evidenceDir, `${expected.name}.json`), JSON.stringify({
        case: expected.name,
        project_id: fixture.id,
        profile: {
          execution_backend: 'pytorch',
          keras_enabled: expected.kerasEnabled !== false,
          worker: finalJob.worker,
        },
        revision: saved.current_revision,
        expected: {
          static_output_shape: expected.staticOutputShape,
          runtime_output_shape: expected.runtimeOutputShape,
          parameter_count: expected.parameterCount,
        },
        validation: validationPayload,
        submitted_job_id: submitted.id,
        final_job: finalJob,
        browser_errors: browserErrors,
      }, null, 2))
      await writeFile(path.join(evidenceDir, `${expected.name}-browser-errors.json`), JSON.stringify(browserErrors, null, 2))
    }

    const advancedCases: AdvancedCase[] = [
      {
        name: 'nw-t3-04-max-pool-1d-keras-pytorch',
        inputDefinition: 'input.sequence',
        inputShape: [8, 2],
        catalogNode: 'Max Pooling 1D',
        staticOutputShape: [null, 4, 2],
        runtimeOutputShape: [1, 4, 2],
        parameterCount: 0,
      },
      {
        name: 'nw-t3-04-average-pool-2d-keras-pytorch',
        inputDefinition: 'input.image',
        inputShape: [8, 8, 2],
        catalogNode: 'Average Pooling 2D',
        staticOutputShape: [null, 4, 4, 2],
        runtimeOutputShape: [1, 4, 4, 2],
        parameterCount: 0,
      },
      {
        name: 'nw-t3-04-max-pool-3d-keras-pytorch',
        inputDefinition: 'input.tensor',
        inputShape: [4, 4, 4, 2],
        catalogNode: 'Max Pooling 3D',
        staticOutputShape: [null, 2, 2, 2, 2],
        runtimeOutputShape: [1, 2, 2, 2, 2],
        parameterCount: 0,
      },
      {
        name: 'nw-t3-04-depthwise-keras-pytorch',
        inputDefinition: 'input.image',
        inputShape: [4, 4, 2],
        catalogNode: 'Depthwise Convolution',
        staticOutputShape: [null, 2, 2, 2],
        runtimeOutputShape: [1, 2, 2, 2],
        parameterCount: 20,
      },
      {
        name: 'nw-t3-04-depthwise-native-pytorch',
        inputDefinition: 'input.image',
        inputShape: [4, 4, 2],
        catalogNode: 'Depthwise Convolution',
        staticOutputShape: [null, 2, 2, 2],
        runtimeOutputShape: [1, 2, 2, 2],
        parameterCount: 20,
        kerasEnabled: false,
      },
      {
        name: 'nw-t3-04-separable-keras-pytorch',
        inputDefinition: 'input.image',
        inputShape: [4, 4, 2],
        catalogNode: 'Separable Convolution',
        staticOutputShape: [null, 2, 2, 32],
        runtimeOutputShape: [1, 2, 2, 32],
        parameterCount: 114,
      },
      {
        name: 'nw-t3-04-transposed-keras-pytorch',
        inputDefinition: 'input.image',
        inputShape: [4, 4, 1],
        catalogNode: 'Transposed Convolution',
        staticOutputShape: [null, 6, 6, 32],
        runtimeOutputShape: [1, 6, 6, 32],
        parameterCount: 320,
      },
      {
        name: 'nw-t3-04-upsampling-keras-pytorch',
        inputDefinition: 'input.image',
        inputShape: [3, 4, 1],
        catalogNode: 'Upsampling',
        staticOutputShape: [null, 6, 8, 1],
        runtimeOutputShape: [1, 6, 8, 1],
        parameterCount: 0,
      },
      {
        name: 'nw-t3-04-padding-keras-pytorch',
        inputDefinition: 'input.image',
        inputShape: [4, 5, 1],
        catalogNode: 'Padding',
        staticOutputShape: [null, 6, 7, 1],
        runtimeOutputShape: [1, 6, 7, 1],
        parameterCount: 0,
      },
      {
        name: 'nw-t3-04-cropping-keras-pytorch',
        inputDefinition: 'input.image',
        inputShape: [6, 7, 1],
        catalogNode: 'Cropping',
        staticOutputShape: [null, 4, 5, 1],
        runtimeOutputShape: [1, 4, 5, 1],
        parameterCount: 0,
      },
    ]

    test.describe('NW-T3-04 advanced and spatial convolution catalog slice', () => {
      test.describe.configure({ mode: 'serial' })
      test.setTimeout(180_000)

      test('live catalog exposes the selected convolution definitions and profile mappings', async ({ request }) => {
        const response = await request.get('/api/nodes/catalog')
        expect(response.ok()).toBeTruthy()
        const payload = await response.json() as {
          schema_version: number
          nodes: Array<{
            id: string
            display_name: string
            category: string
            implementations: Record<string, { supported?: boolean }>
          }>
        }
        const selected = payload.nodes.filter((item) => item.category === 'convolutional' && advancedCases.some((candidate) => candidate.catalogNode === item.display_name))
        const selectedIds = selected.map((item) => item.id).sort()
        expect(selectedIds).toEqual([
          'conv.average_pool2d',
          'conv.cropping',
          'conv.depthwise',
          'conv.max_pool1d',
          'conv.max_pool3d',
          'conv.padding',
          'conv.separable',
          'conv.transposed',
          'conv.upsampling',
        ])
        expect(selected).toHaveLength(9)
        for (const definition of selected) {
          expect(definition.implementations.keras?.supported).toBe(true)
          expect(definition.implementations.pytorch?.supported).toBe(true)
        }
        await mkdir(evidenceDir, { recursive: true })
        await writeFile(path.join(evidenceDir, 'nw-t3-04-live-catalog.json'), JSON.stringify({
          schema_version: payload.schema_version,
          catalog_count: payload.nodes.length,
          selected,
        }, null, 2))
      })

      for (const candidate of advancedCases) {
        test(`${candidate.name} builds, persists, and completes exact-profile functional validation`, async ({ page, request }, testInfo) => {
          const fixture = await createProject(
            request,
            `NW-T3-04 ${candidate.catalogNode} ${Date.now()}`,
            inputOutputGraph(candidate.inputDefinition, candidate.inputShape),
            candidate.kerasEnabled !== false,
          )
          const browserErrors: string[] = []
          page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
          page.on('console', (message) => {
            if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`)
          })
          try {
            await page.goto(`/builder/${fixture.id}`)
            await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
            await expect(page.locator('[data-node-id]')).toHaveCount(2)
            const advancedNodeId = await addCatalogNode(page, candidate.catalogNode)
            await connectByLabels(page, 'Matrix input', candidate.catalogNode, 1)
            await connectByLabels(page, candidate.catalogNode, 'Matrix output', 2)
            await validateSaveAndRun(page, request, fixture, advancedNodeId, candidate, testInfo, browserErrors)
          } finally {
            await deleteProject(request, fixture.id)
          }
        })
      }
    })
  })
  test.describe('arithmetic merge slice.spec', () => {
    type CreatedProject = { id: string; current_revision: number; name: string }
    type GraphNode = {
      id: string
      definition_id: string
      definition_version: number
      label: string
      position: { x: number; y: number }
      parameters: Record<string, unknown>
    }
    type GraphEdge = {
      id: string
      source_node_id: string
      source_port_id: string
      target_node_id: string
      target_port_id: string
    }
    type ShapeSummary = {
      node_id: string
      parameter_count: number
      outputs: Record<string, { shape: Array<number | null> }>
    }
    type ValidationDiagnostic = { code?: string; message?: string; node_id?: string | null }
    type ValidationPayload = {
      valid: boolean
      diagnostics: ValidationDiagnostic[]
      shape_summaries: ShapeSummary[]
      total_parameters: number
    }
    type RuntimeJob = {
      state: string
      worker?: { api: string; backend: string; device_id: string }
      result?: { status: string; outputs: Array<{ shape: number[] }>; parameter_count: number }
    }
    type SavedProject = {
      current_revision: number
      validation_status: string
      graph: { nodes: GraphNode[]; edges: GraphEdge[] }
    }
    type MergeCase = {
      name: string
      displayName: string
      definitionId: string
      kerasEnabled: boolean
      worker: { api: string; backend: string; device_id: string }
    }

    const evidenceDir = path.resolve(
      process.env.NW_VALIDATION_EVIDENCE_DIR
        ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260929', 'evidence', 'nw-t3-08-merge'),
    )

    function graphShell(nodes: GraphNode[], edges: GraphEdge[] = []) {
      return {
        schema_version: 6,
        nodes,
        edges,
        model_configuration: {},
        export_settings: {
          include_comments: true,
          include_shape_assertions: true,
          preferred_binary_format: 'keras',
        },
        future_training_configuration: {},
      }
    }

    function node(
      id: string,
      definitionId: string,
      label: string,
      position: { x: number; y: number },
      parameters: Record<string, unknown>,
    ): GraphNode {
      return {
        id,
        definition_id: definitionId,
        definition_version: 1,
        label,
        position,
        parameters,
      }
    }

    function fanInGraph(definitionId?: string, mismatch = false) {
      const nodes = [
        node('left-input', 'input.tensor', 'Left input', { x: 48, y: 80 }, { shape: [4], dtype: 'float32' }),
        node('middle-input', 'input.tensor', 'Middle input', { x: 48, y: 240 }, { shape: [mismatch ? 5 : 4], dtype: 'float32' }),
        node('right-input', 'input.tensor', 'Right input', { x: 48, y: 400 }, { shape: [4], dtype: 'float32' }),
        node('merge', definitionId ?? 'merge.add', 'Merge operation', { x: 560, y: 240 }, {}),
        node('merged-output', 'output.custom', 'Merged output', { x: 1040, y: 240 }, {}),
      ]
      if (!definitionId) return graphShell(nodes)
      const edges = [
        ['left', 'left-input'],
        ['middle', 'middle-input'],
        ['right', 'right-input'],
      ].map(([edgeId, sourceNodeId]) => ({
        id: `${edgeId}-merge`,
        source_node_id: sourceNodeId,
        source_port_id: 'output',
        target_node_id: 'merge',
        target_port_id: 'inputs',
      }))
      edges.push({
        id: 'merge-output',
        source_node_id: 'merge',
        source_port_id: 'output',
        target_node_id: 'merged-output',
        target_port_id: 'input',
      })
      return graphShell(nodes, edges)
    }

    async function createProject(
      request: APIRequestContext,
      name: string,
      graph: ReturnType<typeof graphShell>,
      kerasEnabled: boolean,
    ): Promise<CreatedProject> {
      const response = await request.post('/api/projects', {
        data: {
          name,
          description: 'NW-T3-08 arithmetic merge validation fixture',
          execution_backend: 'pytorch',
          keras_enabled: kerasEnabled,
          graph,
        },
      })
      expect(response.ok()).toBeTruthy()
      return response.json() as Promise<CreatedProject>
    }

    async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
      const response = await request.delete(`/api/projects/${id}`)
      expect([204, 404]).toContain(response.status())
    }

    async function addCatalogNode(page: Page, displayName: string, definitionId: string): Promise<string> {
      const nodes = page.locator('[data-node-id]')
      const before = new Set(
        await nodes.evaluateAll((elements) => elements
          .map((element) => element.getAttribute('data-node-id'))
          .filter(Boolean) as string[]),
      )
      await page.getByRole('textbox', { name: 'Search nodes' }).fill(definitionId)
      const item = page.locator('.catalog-item').filter({ hasText: definitionId }).first()
      await expect(item, `${definitionId} should be available in the live catalog`).toBeEnabled()
      await item.click()
      await expect(nodes).toHaveCount(before.size + 1)
      const after = await nodes.evaluateAll((elements) => elements
        .map((element) => element.getAttribute('data-node-id'))
        .filter(Boolean) as string[])
      const created = after.find((nodeId) => !before.has(nodeId))
      expect(created, `Builder should materialize ${definitionId}`).toBeTruthy()
      await page.getByRole('textbox', { name: 'Search nodes' }).fill('')
      return created!
    }

    async function connectByPorts(
      page: Page,
      sourceLabel: string,
      targetLabel: string,
      targetPort: string,
      expectedConnectionCount: number,
    ): Promise<void> {
      const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
      const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
      const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
      const target = targetNode.getByRole('button', { name: new RegExp(`^Port ${targetPort} \\(input\\)`) })
      await expect(source).toBeVisible()
      await expect(target).toBeVisible()
      await source.press('Enter')
      await expect(source).toHaveAttribute('aria-pressed', 'true')
      const responsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation/connection') && response.request().method() === 'POST',
      )
      await target.press('Enter')
      const response = await responsePromise
      expect(response.ok()).toBeTruthy()
      const payload = await response.json() as { valid: boolean }
      expect(payload.valid, `${sourceLabel} → ${targetLabel}.${targetPort} should be accepted`).toBe(true)
      await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
    }

    async function validateSaveAndRun(
      page: Page,
      request: APIRequestContext,
      fixture: CreatedProject,
      candidate: MergeCase,
      mergeNodeId: string,
      browserErrors: string[],
      failedRequests: string[],
      testInfo: TestInfo,
    ): Promise<{ validation: ValidationPayload; saved: SavedProject; finalJob: RuntimeJob; submittedJobId: string }> {
      await connectByPorts(page, 'Left input', candidate.displayName, 'Inputs', 1)
      await connectByPorts(page, 'Middle input', candidate.displayName, 'Inputs', 2)
      await connectByPorts(page, 'Right input', candidate.displayName, 'Inputs', 3)
      await connectByPorts(page, candidate.displayName, 'Merged output', 'Input', 4)

      const validationResponsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Validate', exact: true }).click()
      const validationResponse = await validationResponsePromise
      expect(validationResponse.ok()).toBeTruthy()
      const validation = await validationResponse.json() as ValidationPayload
      expect(validation.valid).toBe(true)
      expect(validation.diagnostics).toEqual([])
      const summary = validation.shape_summaries.find((item) => item.node_id === mergeNodeId)
      expect(summary?.outputs.output?.shape).toEqual([null, 4])
      expect(summary?.parameter_count).toBe(0)
      expect(validation.total_parameters).toBe(0)
      await expect(page.locator('.validation-summary')).toContainText('Ready')
      await page.screenshot({ path: testInfo.outputPath(`${candidate.name}-builder-ready.png`), fullPage: true })

      const saveResponsePromise = page.waitForResponse(
        (response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT',
      )
      await page.getByRole('button', { name: 'Save model', exact: true }).click()
      const saveResponse = await saveResponsePromise
      expect(saveResponse.ok()).toBeTruthy()
      const saved = await saveResponse.json() as SavedProject
      expect(saved.validation_status).toBe('valid')
      expect(saved.graph.nodes).toHaveLength(5)
      expect(saved.graph.edges).toHaveLength(4)
      expect(saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)?.parameters).toEqual({})

      await page.reload()
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
      await expect(page.locator('[data-node-id]')).toHaveCount(5)
      await expect(page.getByRole('group', { name: `Node ${candidate.displayName}`, exact: true })).toBeVisible()
      await mkdir(evidenceDir, { recursive: true })
      await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-builder-ready.png`), fullPage: true })

      await page.goto(`/models/${fixture.id}/validation`)
      await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
      await expect(page.getByText(candidate.kerasEnabled ? /keras.*pytorch/i : /native.*pytorch/i).first()).toBeVisible({ timeout: 75_000 })
      const submitResponsePromise = page.waitForResponse(
        (response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
      const submitResponse = await submitResponsePromise
      expect(submitResponse.ok()).toBeTruthy()
      const submitted = await submitResponse.json() as { id: string }
      await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 90_000 })

      let finalJob: RuntimeJob = { state: 'missing' }
      await expect.poll(async () => {
        const jobResponse = await request.get(`/api/execution/jobs/${submitted.id}`)
        if (!jobResponse.ok()) return 'missing'
        finalJob = await jobResponse.json() as RuntimeJob
        return finalJob.state
      }, { timeout: 90_000 }).toBe('succeeded')
      expect(finalJob.worker).toMatchObject(candidate.worker)
      expect(finalJob.result?.status).toBe('succeeded')
      expect(finalJob.result?.outputs[0]?.shape).toEqual([1, 4])
      expect(finalJob.result?.parameter_count).toBe(0)
      expect(browserErrors, `${candidate.name} should not emit browser errors`).toEqual([])
      expect(failedRequests, `${candidate.name} should not emit failed product requests`).toEqual([])
      await page.screenshot({ path: testInfo.outputPath(`${candidate.name}-runtime-succeeded.png`), fullPage: true })
      await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-runtime-succeeded.png`), fullPage: true })
      await writeFile(path.join(evidenceDir, `${candidate.name}.json`), JSON.stringify({
        case: candidate.name,
        project_id: fixture.id,
        definition_id: candidate.definitionId,
        profile: { execution_backend: 'pytorch', keras_enabled: candidate.kerasEnabled, worker: finalJob.worker },
        saved_revision: saved.current_revision,
        graph: { node_count: saved.graph.nodes.length, edge_count: saved.graph.edges.length },
        expected: {
          static_output_shape: [null, 4],
          runtime_output_shape: [1, 4],
          parameter_count: 0,
        },
        validation,
        submitted_job_id: submitted.id,
        final_job: finalJob,
        browser_errors: browserErrors,
        failed_requests: failedRequests,
        cleanup: 'project deleted in finally block',
      }, null, 2))
      await writeFile(path.join(evidenceDir, `${candidate.name}-browser-errors.json`), JSON.stringify({ browser_errors: browserErrors, failed_requests: failedRequests }, null, 2))
      return { validation, saved, finalJob, submittedJobId: submitted.id }
    }

    const arithmeticCases: MergeCase[] = [
      { name: 'nw-t3-08-add-keras-pytorch', displayName: 'Add', definitionId: 'merge.add', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-08-subtract-keras-pytorch', displayName: 'Subtract', definitionId: 'merge.subtract', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-08-multiply-keras-pytorch', displayName: 'Multiply', definitionId: 'merge.multiply', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-08-divide-keras-pytorch', displayName: 'Divide', definitionId: 'merge.divide', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-08-maximum-keras-pytorch', displayName: 'Maximum', definitionId: 'merge.maximum', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-08-minimum-keras-pytorch', displayName: 'Minimum', definitionId: 'merge.minimum', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-08-average-keras-pytorch', displayName: 'Average', definitionId: 'merge.average', kerasEnabled: true, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-08-add-native-pytorch', displayName: 'Add', definitionId: 'merge.add', kerasEnabled: false, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-08-average-native-pytorch', displayName: 'Average', definitionId: 'merge.average', kerasEnabled: false, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
    ]

    test.describe('NW-T3-08 arithmetic merge and fan-in catalog slice', () => {
      test.describe.configure({ mode: 'serial' })
      test.setTimeout(240_000)

      test('live catalog renders arithmetic merge definitions and verified mappings', async ({ page, request }) => {
        const response = await request.get('/api/nodes/catalog')
        expect(response.ok()).toBeTruthy()
        const catalog = await response.json() as {
          schema_version: number
          nodes: Array<{
            id: string
            display_name: string
            category: string
            operation?: { family?: string; shape_behavior?: string; variadic?: boolean }
            implementations: Record<string, { supported?: boolean; adapter?: string | null }>
          }>
        }
        const merge = catalog.nodes.filter((item) => item.category === 'merge').sort((left, right) => left.id.localeCompare(right.id))
        expect(merge).toHaveLength(14)
        const expectedIds = [
          'merge.add', 'merge.average', 'merge.concatenate', 'merge.divide', 'merge.dot', 'merge.matmul',
          'merge.maximum', 'merge.minimum', 'merge.multiply', 'merge.residual', 'merge.skip', 'merge.split',
          'merge.stack', 'merge.subtract',
        ]
        expect(merge.map((item) => item.id)).toEqual(expectedIds)
        const arithmetic = merge.filter((item) => [
          'merge.add', 'merge.subtract', 'merge.multiply', 'merge.divide', 'merge.maximum', 'merge.minimum',
        ].includes(item.id))
        expect(arithmetic).toHaveLength(6)
        for (const item of arithmetic) {
          expect(item.operation).toMatchObject({ family: 'elementwise', shape_behavior: 'preserve', variadic: true })
          expect(item.implementations.keras?.supported).toBe(true)
          expect(item.implementations.tensorflow?.supported).toBe(true)
          expect(item.implementations.pytorch?.supported).toBe(true)
          expect(item.implementations.jax?.supported).toBe(true)
        }
        const average = merge.find((item) => item.id === 'merge.average')!
        expect(average.implementations.keras).toMatchObject({ supported: true, adapter: 'average' })
        expect(average.implementations.pytorch).toMatchObject({ supported: true, adapter: 'average' })

        await page.goto('/builder')
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        await page.getByRole('button', { name: /Merge 14 nodes/ }).click()
        for (const item of merge) {
          await expect(page.getByRole('button', { name: new RegExp(`${item.display_name} ${item.id} ·`) })).toBeEnabled()
        }
        await mkdir(evidenceDir, { recursive: true })
        await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-08-merge-catalog-rendered.png'), fullPage: true })
        await writeFile(path.join(evidenceDir, 'nw-t3-08-live-catalog.json'), JSON.stringify({
          schema_version: catalog.schema_version,
          catalog_count: catalog.nodes.length,
          catalog_fingerprint_sha256: createHash('sha256').update(JSON.stringify(catalog)).digest('hex'),
          merge,
          arithmetic_ids: arithmetic.map((item) => item.id),
        }, null, 2))
      })

      for (const candidate of arithmeticCases) {
        test(`${candidate.name} builds, persists, and completes exact-profile functional validation`, async ({ page, request }, testInfo) => {
          const fixture = await createProject(
            request,
            `Tier 3 arithmetic merge ${candidate.definitionId} ${candidate.kerasEnabled ? 'keras' : 'native'} ${Date.now()}`,
            graphShell([
              node('left-input', 'input.tensor', 'Left input', { x: 48, y: 80 }, { shape: [4], dtype: 'float32' }),
              node('middle-input', 'input.tensor', 'Middle input', { x: 48, y: 240 }, { shape: [4], dtype: 'float32' }),
              node('right-input', 'input.tensor', 'Right input', { x: 48, y: 400 }, { shape: [4], dtype: 'float32' }),
              node('merged-output', 'output.custom', 'Merged output', { x: 1040, y: 240 }, {}),
            ]),
            candidate.kerasEnabled,
          )
          const browserErrors: string[] = []
          const failedRequests: string[] = []
          page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
          page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
          page.on('requestfailed', (event) => failedRequests.push(`${event.method()} ${event.url()} ${event.failure()?.errorText ?? 'unknown'}`))
          page.on('response', (event) => {
            if (event.status() >= 400 && event.url().includes('/api/')) failedRequests.push(`${event.status()} ${event.request().method()} ${event.url()}`)
          })
          try {
            await page.goto(`/builder/${fixture.id}`)
            await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
            if (!candidate.kerasEnabled) await expect(page.getByLabel('Enable Keras', { exact: true })).not.toBeChecked()
            const mergeNodeId = await addCatalogNode(page, candidate.displayName, candidate.definitionId)
            await validateSaveAndRun(page, request, fixture, candidate, mergeNodeId, browserErrors, failedRequests, testInfo)
          } finally {
            await deleteProject(request, fixture.id)
          }
        })
      }

      test('incompatible fan-in shapes remain blocking and fail closed', async ({ page, request }, testInfo) => {
        const fixture = await createProject(
          request,
          `Tier 3 arithmetic merge mismatch ${Date.now()}`,
          fanInGraph('merge.add', true),
          true,
        )
        const browserErrors: string[] = []
        const failedRequests: string[] = []
        page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
        page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
        page.on('requestfailed', (event) => failedRequests.push(`${event.method()} ${event.url()} ${event.failure()?.errorText ?? 'unknown'}`))
        try {
          await page.goto(`/builder/${fixture.id}`)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
          const validationResponsePromise = page.waitForResponse(
            (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
          )
          await page.getByRole('button', { name: 'Validate', exact: true }).click()
          const validationResponse = await validationResponsePromise
          expect(validationResponse.ok()).toBeTruthy()
          const validation = await validationResponse.json() as ValidationPayload
          expect(validation.valid).toBe(false)
          expect(validation.diagnostics.some((item) => item.message?.includes('compatible shapes'))).toBe(true)
          await expect(page.locator('.validation-summary')).toContainText(/Blocked/i)
          await mkdir(evidenceDir, { recursive: true })
          await page.screenshot({ path: testInfo.outputPath('nw-t3-08-incompatible-fan-in.png'), fullPage: true })
          await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-08-incompatible-fan-in.png'), fullPage: true })
          await writeFile(path.join(evidenceDir, 'nw-t3-08-incompatible-fan-in.json'), JSON.stringify({
            case: 'nw-t3-08-incompatible-fan-in',
            project_id: fixture.id,
            definition_id: 'merge.add',
            authored_shapes: { left: [4], middle: [5], right: [4] },
            validation,
            browser_errors: browserErrors,
            failed_requests: failedRequests,
            execution_job_submissions: 0,
          }, null, 2))
          expect(browserErrors).toEqual([])
          expect(failedRequests).toEqual([])
        } finally {
          await deleteProject(request, fixture.id)
        }
      })
    })
  })
  test.describe('attention primitives slice.spec', () => {
    type CreatedProject = { id: string; current_revision: number; name: string }
    type GraphNode = {
      id: string
      definition_id: string
      definition_version: number
      label: string
      position: { x: number; y: number }
      parameters: Record<string, unknown>
    }
    type ShapeSummary = {
      node_id: string
      parameter_count: number
      outputs: Record<string, { shape: Array<number | null> }>
    }
    type ValidationDiagnostic = {
      code?: string
      message?: string
      node_id?: string | null
      parameter_id?: string | null
    }
    type ValidationPayload = {
      valid: boolean
      diagnostics: ValidationDiagnostic[]
      shape_summaries: ShapeSummary[]
      total_parameters: number
    }
    type RuntimeJob = {
      state: string
      worker?: { api: string; backend: string; device_id: string }
      result?: {
        status: string
        outputs: Array<{ shape: number[] }>
        parameter_count: number
      }
    }
    type SavedProject = {
      current_revision: number
      validation_status: string
      graph: { nodes: GraphNode[]; edges: unknown[] }
    }
    type AttentionCase = {
      name: string
      catalogNode: string
      definitionId: string
      includeContext: boolean
      inputLabels: string[]
      parameters: Record<string, unknown>
      staticOutputShape: Array<number | null>
      runtimeOutputShape: number[]
      parameterCount: number
      kerasEnabled: boolean
      worker: { api: string; backend: string; device_id: string }
    }

    const evidenceDir = path.resolve(
      process.env.NW_VALIDATION_EVIDENCE_DIR
        ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260929', 'evidence', 'nw-t3-06-attention'),
    )

    function graphShell(nodes: GraphNode[]) {
      return {
        schema_version: 6,
        nodes,
        edges: [],
        model_configuration: {},
        export_settings: {
          include_comments: true,
          include_shape_assertions: true,
          preferred_binary_format: 'keras',
        },
        future_training_configuration: {},
      }
    }

    function node(
      id: string,
      definitionId: string,
      label: string,
      position: { x: number; y: number },
      parameters: Record<string, unknown>,
    ): GraphNode {
      return {
        id,
        definition_id: definitionId,
        definition_version: 1,
        label,
        position,
        parameters,
      }
    }

    function initialGraph(includeContext: boolean) {
      const nodes = includeContext
        ? [
            node('query-input', 'input.sequence', 'Query Sequence', { x: 48, y: 120 }, { shape: [5, 8], dtype: 'float32' }),
            node('context-input', 'input.sequence', 'Context Sequence', { x: 48, y: 360 }, { shape: [3, 8], dtype: 'float32' }),
            node('custom-output', 'output.custom', 'Custom Output', { x: 1120, y: 240 }, {}),
          ]
        : [
            node('sequence-input', 'input.sequence', 'Input Sequence', { x: 48, y: 240 }, { shape: [5, 8], dtype: 'float32' }),
            node('custom-output', 'output.custom', 'Custom Output', { x: 1120, y: 240 }, {}),
          ]
      return graphShell(nodes)
    }

    async function createProject(
      request: APIRequestContext,
      name: string,
      includeContext: boolean,
      kerasEnabled: boolean,
    ): Promise<CreatedProject> {
      const response = await request.post('/api/projects', {
        data: {
          name,
          description: 'NW-T3-06 attention primitive validation fixture',
          execution_backend: 'pytorch',
          keras_enabled: kerasEnabled,
          graph: initialGraph(includeContext),
        },
      })
      expect(response.ok()).toBeTruthy()
      return response.json() as Promise<CreatedProject>
    }

    async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
      const response = await request.delete(`/api/projects/${id}`)
      expect([204, 404]).toContain(response.status())
    }

    async function addCatalogNode(page: Page, displayName: string): Promise<string> {
      const nodes = page.locator('[data-node-id]')
      const before = new Set(
        await nodes.evaluateAll((elements) => elements
          .map((element) => element.getAttribute('data-node-id'))
          .filter(Boolean) as string[]),
      )
      await page.getByRole('textbox', { name: 'Search nodes' }).fill(displayName)
      const item = page.locator('.catalog-item').filter({ hasText: displayName }).first()
      await expect(item, `Catalog node ${displayName} should be available`).toBeEnabled()
      await item.click()
      await expect(nodes).toHaveCount(before.size + 1)
      const after = await nodes.evaluateAll((elements) => elements
        .map((element) => element.getAttribute('data-node-id'))
        .filter(Boolean) as string[])
      const created = after.find((nodeId) => !before.has(nodeId))
      expect(created, `Builder should materialize ${displayName}`).toBeTruthy()
      await page.getByRole('textbox', { name: 'Search nodes' }).fill('')
      return created!
    }

    async function connectByPorts(
      page: Page,
      sourceLabel: string,
      targetLabel: string,
      targetPortLabel: string,
      expectedConnectionCount: number,
    ): Promise<void> {
      const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
      const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
      const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
      const target = targetNode.getByRole('button', { name: new RegExp(`^Port ${targetPortLabel} \\(input\\)`) })
      await expect(source).toBeVisible()
      await expect(target).toBeVisible()
      await source.press('Enter')
      await expect(source).toHaveAttribute('aria-pressed', 'true')
      const responsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation/connection') && response.request().method() === 'POST',
      )
      await target.press('Enter')
      const response = await responsePromise
      expect(response.ok()).toBeTruthy()
      const payload = await response.json() as { valid: boolean }
      expect(payload.valid, `${sourceLabel} → ${targetLabel}.${targetPortLabel} should be accepted`).toBe(true)
      await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
    }

    async function parameterField(inspector: Locator, displayName: string): Promise<Locator> {
      const field = inspector.locator('label.inspector-field').filter({ hasText: displayName }).first()
      if (!(await field.isVisible())) {
        const group = inspector.locator('details.parameter-group').filter({ has: field }).first()
        await group.locator('summary').click()
        await expect(group).toHaveAttribute('open', '')
      }
      await expect(field, `${displayName} should be visible in the Inspector`).toBeVisible()
      return field
    }

    async function setAttentionParameters(
      page: Page,
      nodeId: string,
      parameters: Record<string, unknown>,
    ): Promise<void> {
      await page.locator(`[data-node-id="${nodeId}"]`).click()
      const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
      for (const [id, value] of Object.entries(parameters)) {
        const displayName = {
          num_heads: 'Num Heads',
          key_dim: 'Key Dim',
          value_dim: 'Value dim',
          layers: 'Layers',
          dropout: 'Dropout',
          use_bias: 'Use bias',
          output_shape: 'Output shape',
        }[id]
        if (!displayName) continue
        const field = await parameterField(inspector, displayName)
        if (id === 'use_bias') {
          const checkbox = field.locator('input[type="checkbox"]')
          if (value === true) await checkbox.check()
          else await checkbox.uncheck()
        } else if (id === 'output_shape') {
          await field.locator('textarea').fill(JSON.stringify(value))
        } else {
          const input = field.locator('input')
          if (await input.isEnabled()) {
            await input.fill(String(value))
          }
          await expect(input).toHaveValue(String(value))
        }
      }
    }

    async function validateGraph(page: Page): Promise<ValidationPayload> {
      const validationResponsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Validate', exact: true }).click()
      const validationResponse = await validationResponsePromise
      expect(validationResponse.ok()).toBeTruthy()
      return validationResponse.json() as Promise<ValidationPayload>
    }

    async function saveProject(page: Page, fixture: CreatedProject): Promise<SavedProject> {
      const saveResponsePromise = page.waitForResponse(
        (response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT',
      )
      await page.getByRole('button', { name: 'Save model', exact: true }).click()
      const saveResponse = await saveResponsePromise
      expect(saveResponse.ok()).toBeTruthy()
      return saveResponse.json() as Promise<SavedProject>
    }

    async function validateSaveAndRun(
      page: Page,
      request: APIRequestContext,
      fixture: CreatedProject,
      attentionNodeId: string,
      candidate: AttentionCase,
      browserErrors: string[],
      failedRequests: string[],
    ): Promise<{ validation: ValidationPayload; saved: SavedProject; submittedJobId: string; finalJob: RuntimeJob }> {
      const validation = await validateGraph(page)
      expect(validation.valid, JSON.stringify(validation.diagnostics)).toBe(true)
      expect(validation.diagnostics).toEqual([])
      const summary = validation.shape_summaries.find((item) => item.node_id === attentionNodeId)
      expect(summary?.outputs.output?.shape).toEqual(candidate.staticOutputShape)
      expect(summary?.parameter_count).toBe(candidate.parameterCount)
      expect(validation.total_parameters).toBe(candidate.parameterCount)
      await expect(page.locator('.validation-summary')).toContainText('Ready')
      await mkdir(evidenceDir, { recursive: true })
      await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-builder-ready.png`), fullPage: true })

      const saved = await saveProject(page, fixture)
      expect(saved.validation_status).toBe('valid')
      expect(saved.graph.nodes).toHaveLength(candidate.includeContext ? 4 : 3)
      expect(saved.graph.edges).toHaveLength(candidate.includeContext ? 3 : 2)
      const savedAttention = saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)
      expect(savedAttention?.parameters).toMatchObject(candidate.parameters)

      await page.reload()
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
      await expect(page.locator('[data-node-id]')).toHaveCount(candidate.includeContext ? 4 : 3)
      await expect(page.getByRole('group', { name: `Node ${candidate.catalogNode}`, exact: true })).toBeVisible()

      await page.goto(`/models/${fixture.id}/validation`)
      await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
      const profilePattern = candidate.kerasEnabled ? /keras.*pytorch/i : /native.*pytorch/i
      await expect(page.getByText(profilePattern).first()).toBeVisible({ timeout: 75_000 })
      await expect(page.getByRole('button', { name: 'Run functional validation', exact: true })).toBeEnabled()

      const submitResponsePromise = page.waitForResponse(
        (response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
      const submitResponse = await submitResponsePromise
      expect(submitResponse.ok()).toBeTruthy()
      const submitted = await submitResponse.json() as { id: string }
      await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 90_000 })

      let finalJob: RuntimeJob = { state: 'missing' }
      await expect.poll(async () => {
        const jobResponse = await request.get(`/api/execution/jobs/${submitted.id}`)
        if (!jobResponse.ok()) return 'missing'
        finalJob = await jobResponse.json() as RuntimeJob
        return finalJob.state
      }, { timeout: 90_000 }).toBe('succeeded')

      expect(finalJob.worker).toMatchObject(candidate.worker)
      expect(finalJob.result?.status).toBe('succeeded')
      expect(finalJob.result?.outputs[0]?.shape).toEqual(candidate.runtimeOutputShape)
      expect(finalJob.result?.parameter_count).toBe(candidate.parameterCount)
      await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-runtime-succeeded.png`), fullPage: true })
      expect(browserErrors, `${candidate.name} should not emit browser errors`).toEqual([])
      expect(failedRequests, `${candidate.name} should not emit failed product requests`).toEqual([])

      await writeFile(path.join(evidenceDir, `${candidate.name}.json`), JSON.stringify({
        case: candidate.name,
        project_id: fixture.id,
        definition_id: candidate.definitionId,
        authored_parameters: candidate.parameters,
        profile: {
          execution_backend: 'pytorch',
          keras_enabled: candidate.kerasEnabled,
          worker: finalJob.worker,
        },
        saved_revision: saved.current_revision,
        graph: { node_count: saved.graph.nodes.length, edge_count: saved.graph.edges.length },
        validation,
        submitted_job_id: submitted.id,
        final_job: finalJob,
        browser_errors: browserErrors,
        failed_requests: failedRequests,
        cleanup: 'project deleted in finally block',
      }, null, 2))
      await writeFile(path.join(evidenceDir, `${candidate.name}-browser-errors.json`), JSON.stringify({ browser_errors: browserErrors, failed_requests: failedRequests }, null, 2))
      return { validation, saved, submittedJobId: submitted.id, finalJob }
    }

    const attentionCases: AttentionCase[] = [
      {
        name: 'nw-t3-06-self-keras-pytorch',
        catalogNode: 'Self-attention',
        definitionId: 'attention.self',
        includeContext: false,
        inputLabels: ['Input Sequence'],
        parameters: { num_heads: 2, key_dim: 4, value_dim: 4, layers: 1, dropout: 0, use_bias: true },
        staticOutputShape: [null, 5, 8],
        runtimeOutputShape: [1, 5, 8],
        parameterCount: 288,
        kerasEnabled: true,
        worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
      },
      {
        name: 'nw-t3-06-cross-keras-pytorch',
        catalogNode: 'Cross-attention',
        definitionId: 'attention.cross',
        includeContext: true,
        inputLabels: ['Query Sequence', 'Context Sequence'],
        parameters: { num_heads: 2, key_dim: 4, value_dim: 4, layers: 1, dropout: 0, use_bias: true },
        staticOutputShape: [null, 5, 8],
        runtimeOutputShape: [1, 5, 8],
        parameterCount: 288,
        kerasEnabled: true,
        worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
      },
      {
        name: 'nw-t3-06-multi-head-output-shape-keras-pytorch',
        catalogNode: 'Multi-head Attention',
        definitionId: 'attention.multi_head',
        includeContext: false,
        inputLabels: ['Input Sequence'],
        parameters: { num_heads: 2, key_dim: 4, value_dim: 4, output_shape: [2, 4], layers: 1 },
        staticOutputShape: [null, 5, 2, 4],
        runtimeOutputShape: [1, 5, 2, 4],
        parameterCount: 288,
        kerasEnabled: true,
        worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
      },
      {
        name: 'nw-t3-06-self-native-pytorch',
        catalogNode: 'Self-attention',
        definitionId: 'attention.self',
        includeContext: false,
        inputLabels: ['Input Sequence'],
        parameters: { num_heads: 2, key_dim: 4, value_dim: 4, layers: 1, dropout: 0, use_bias: true },
        staticOutputShape: [null, 5, 8],
        runtimeOutputShape: [1, 5, 8],
        parameterCount: 288,
        kerasEnabled: false,
        worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' },
      },
    ]

    test.describe('NW-T3-06 attention primitive catalog slice', () => {
      test.describe.configure({ mode: 'serial' })
      test.setTimeout(240_000)

      test('live catalog renders all attention primitives with Keras and native mappings', async ({ page, request }) => {
        const catalogResponse = await request.get('/api/nodes/catalog')
        expect(catalogResponse.ok()).toBeTruthy()
        const catalog = await catalogResponse.json() as {
          schema_version: number
          nodes: Array<{
            id: string
            display_name: string
            category: string
            implementations: Record<string, { supported?: boolean }>
          }>
        }
        const metadataResponse = await request.get('/api/metadata')
        expect(metadataResponse.ok()).toBeTruthy()
        const metadata = await metadataResponse.json() as Record<string, unknown>
        const attention = catalog.nodes.filter((item) => item.id.startsWith('attention.'))
        expect(attention.map((item) => item.id).sort()).toEqual(
          ['attention.cross', 'attention.multi_head', 'attention.self'].sort(),
        )
        for (const item of attention) {
          expect(item.implementations.keras?.supported).toBe(true)
          expect(item.implementations.pytorch?.supported).toBe(true)
          expect(item.implementations.tensorflow?.supported).toBe(true)
          expect(item.implementations.jax?.supported).toBe(true)
        }

        await page.goto('/builder')
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        await page.getByRole('button', { name: /Transformer 10 nodes/ }).click()
        for (const candidate of attentionCases.slice(0, 3)) {
          await expect(page.getByRole('button', {
            name: new RegExp(`${candidate.catalogNode} ${candidate.definitionId} · operation`),
          })).toBeEnabled()
        }
        await mkdir(evidenceDir, { recursive: true })
        await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-06-attention-catalog-rendered.png'), fullPage: true })
        await writeFile(path.join(evidenceDir, 'nw-t3-06-live-catalog.json'), JSON.stringify({
          schema_version: catalog.schema_version,
          catalog_count: catalog.nodes.length,
          category_counts: catalog.nodes.reduce<Record<string, number>>((counts, item) => {
            counts[item.category] = (counts[item.category] ?? 0) + 1
            return counts
          }, {}),
          catalog_fingerprint_sha256: createHash('sha256').update(JSON.stringify(catalog)).digest('hex'),
          attention,
          metadata,
        }, null, 2))
      })

      for (const candidate of attentionCases) {
        test(`${candidate.name} builds, persists, and completes exact-profile functional validation`, async ({ page, request }) => {
          const fixture = await createProject(
            request,
            `Tier 3 attention ${candidate.catalogNode} ${Date.now()}`,
            candidate.includeContext,
            candidate.kerasEnabled,
          )
          const browserErrors: string[] = []
          const failedRequests: string[] = []
          page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
          page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
          page.on('requestfailed', (requestEvent) => failedRequests.push(`${requestEvent.method()} ${requestEvent.url()} ${requestEvent.failure()?.errorText ?? 'unknown'}`))
          page.on('response', (response) => {
            if (response.status() >= 400 && response.url().includes('/api/')) failedRequests.push(`${response.status()} ${response.request().method()} ${response.url()}`)
          })
          try {
            await page.goto(`/builder/${fixture.id}`)
            await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
            const attentionNodeId = await addCatalogNode(page, candidate.catalogNode)
            await setAttentionParameters(page, attentionNodeId, candidate.parameters)
            if (candidate.includeContext) {
              await connectByPorts(page, 'Query Sequence', candidate.catalogNode, 'Query', 1)
              await connectByPorts(page, 'Context Sequence', candidate.catalogNode, 'Context', 2)
            } else {
              await connectByPorts(page, 'Input Sequence', candidate.catalogNode, 'Query', 1)
            }
            await connectByPorts(page, candidate.catalogNode, 'Custom Output', 'Input', candidate.includeContext ? 3 : 2)
            await validateSaveAndRun(page, request, fixture, attentionNodeId, candidate, browserErrors, failedRequests)
          } finally {
            await deleteProject(request, fixture.id)
          }
        })
      }

      test('invalid layers are blocked, repaired, persisted, and executed', async ({ page, request }) => {
        const candidate = attentionCases[0]!
        const fixture = await createProject(request, `Tier 3 attention repair ${Date.now()}`, false, true)
        const browserErrors: string[] = []
        const failedRequests: string[] = []
        let jobSubmitCount = 0
        page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
        page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
        page.on('requestfailed', (requestEvent) => failedRequests.push(`${requestEvent.method()} ${requestEvent.url()} ${requestEvent.failure()?.errorText ?? 'unknown'}`))
        page.on('request', (requestEvent) => {
          if (requestEvent.method() === 'POST' && requestEvent.url().includes('/api/execution/jobs')) jobSubmitCount += 1
        })
        try {
          await page.goto(`/builder/${fixture.id}`)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
          const attentionNodeId = await addCatalogNode(page, candidate.catalogNode)
          await setAttentionParameters(page, attentionNodeId, candidate.parameters)
          await connectByPorts(page, 'Input Sequence', candidate.catalogNode, 'Query', 1)
          await connectByPorts(page, candidate.catalogNode, 'Custom Output', 'Input', 2)

          const validBeforeRepair = await validateGraph(page)
          expect(validBeforeRepair.valid).toBe(true)
          const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
          const layers = (await parameterField(inspector, 'Layers')).locator('input')
          await layers.fill('2')
          await expect(layers).toHaveValue('2')
          const invalid = await validateGraph(page)
          expect(invalid.valid).toBe(false)
          expect(invalid.diagnostics.some((item) => item.message?.includes('accepts layers=1'))).toBe(true)
          await expect(page.locator('.validation-summary')).toContainText('Blocked')
          await mkdir(evidenceDir, { recursive: true })
          await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-06-invalid-layers.png'), fullPage: true })
          await writeFile(path.join(evidenceDir, 'nw-t3-06-invalid-layers.json'), JSON.stringify({
            case: 'nw-t3-06-invalid-layers',
            project_id: fixture.id,
            authored_parameters: { ...candidate.parameters, layers: 2 },
            validation: invalid,
            execution_job_submissions: jobSubmitCount,
            browser_errors: browserErrors,
            failed_requests: failedRequests,
          }, null, 2))

          await layers.fill('1')
          await expect(layers).toHaveValue('1')
          const repaired = await validateSaveAndRun(page, request, fixture, attentionNodeId, candidate, browserErrors, failedRequests)
          expect(repaired.saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)?.parameters.layers).toBe(1)
          await writeFile(path.join(evidenceDir, 'nw-t3-06-repaired-layers.json'), JSON.stringify({
            case: 'nw-t3-06-repaired-layers',
            project_id: fixture.id,
            invalid_diagnostic: invalid.diagnostics.find((item) => item.parameter_id === 'layers'),
            repaired_saved_revision: repaired.saved.current_revision,
            repaired_graph: {
              node_count: repaired.saved.graph.nodes.length,
              edge_count: repaired.saved.graph.edges.length,
              layers: repaired.saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)?.parameters.layers,
            },
            submitted_job_id: repaired.submittedJobId,
            final_job: repaired.finalJob,
            execution_job_submissions: jobSubmitCount,
            browser_errors: browserErrors,
            failed_requests: failedRequests,
          }, null, 2))
          await page.goto(`/builder/${fixture.id}`)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
          const persistedNode = page.getByRole('group', { name: `Node ${candidate.catalogNode}`, exact: true })
          await persistedNode.click()
          const persistedInspector = page.getByRole('complementary', { name: 'Properties inspector' })
          await expect(persistedInspector.getByRole('heading', { name: candidate.catalogNode, exact: true })).toBeVisible()
          await expect((await parameterField(persistedInspector, 'Layers')).locator('input')).toHaveValue('1')
          expect(jobSubmitCount).toBe(1)
          expect(browserErrors).toEqual([])
          expect(failedRequests).toEqual([])
        } finally {
          await deleteProject(request, fixture.id)
        }
      })

      test('native attention visibly fixes unsupported dropout without submitting a job', async ({ page, request }) => {
        const candidate = attentionCases[3]!
        const fixture = await createProject(request, `Tier 3 attention native restriction ${Date.now()}`, false, false)
        const browserErrors: string[] = []
        const failedRequests: string[] = []
        let jobSubmitCount = 0
        page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
        page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
        page.on('requestfailed', (requestEvent) => failedRequests.push(`${requestEvent.method()} ${requestEvent.url()} ${requestEvent.failure()?.errorText ?? 'unknown'}`))
        page.on('request', (requestEvent) => {
          if (requestEvent.method() === 'POST' && requestEvent.url().includes('/api/execution/jobs')) jobSubmitCount += 1
        })
        try {
          await page.goto(`/builder/${fixture.id}`)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
          const attentionNodeId = await addCatalogNode(page, candidate.catalogNode)
          await setAttentionParameters(page, attentionNodeId, candidate.parameters)
          const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
          const dropout = await parameterField(inspector, 'Dropout')
          await expect(dropout).toContainText('Fixed to 0 by this backend.')
          await expect(dropout.locator('input')).toBeDisabled()
          await expect(dropout.locator('input')).toHaveValue('0')
          await connectByPorts(page, 'Input Sequence', candidate.catalogNode, 'Query', 1)
          await connectByPorts(page, candidate.catalogNode, 'Custom Output', 'Input', 2)
          const validation = await validateGraph(page)
          expect(validation.valid).toBe(true)
          expect(validation.diagnostics).toEqual([])
          await mkdir(evidenceDir, { recursive: true })
          await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-06-native-dropout-fixed.png'), fullPage: true })
          await writeFile(path.join(evidenceDir, 'nw-t3-06-native-dropout-fixed.json'), JSON.stringify({
            case: 'nw-t3-06-native-dropout-fixed',
            project_id: fixture.id,
            profile: candidate.worker,
            authored_parameters: candidate.parameters,
            capability_boundary: {
              parameter_id: 'dropout',
              rendered_value: 0,
              editable: false,
              message: 'Fixed to 0 by this backend.',
              authored_nonzero_value_is_rejected_by_static_contract: true,
            },
            validation,
            execution_job_submissions: jobSubmitCount,
            browser_errors: browserErrors,
            failed_requests: failedRequests,
          }, null, 2))
          expect(jobSubmitCount).toBe(0)
          expect(browserErrors).toEqual([])
          expect(failedRequests).toEqual([])
        } finally {
          await deleteProject(request, fixture.id)
        }
      })
    })
  })
  test.describe('catalog risk slice.spec', () => {
    type CreatedProject = { id: string; current_revision: number; name: string }
    type GraphNode = {
      id: string
      definition_id: string
      definition_version: number
      label: string
      position: { x: number; y: number }
      parameters: Record<string, unknown>
    }
    type GraphEdge = {
      id: string
      source_node_id: string
      source_port_id: string
      target_node_id: string
      target_port_id: string
    }

    const evidenceDir = path.resolve(
      process.env.NW_VALIDATION_EVIDENCE_DIR
        ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260927', 'evidence'),
    )

    function graphShell(nodes: GraphNode[], edges: GraphEdge[] = []) {
      return {
        schema_version: 6,
        nodes,
        edges,
        model_configuration: {},
        export_settings: {
          include_comments: true,
          include_shape_assertions: true,
          preferred_binary_format: 'keras',
        },
        future_training_configuration: {},
      }
    }

    function node(
      id: string,
      definitionId: string,
      label: string,
      position: { x: number; y: number },
      parameters: Record<string, unknown>,
    ): GraphNode {
      return {
        id,
        definition_id: definitionId,
        definition_version: 1,
        label,
        position,
        parameters,
      }
    }

    async function createProject(
      request: APIRequestContext,
      name: string,
      modelGraph: ReturnType<typeof graphShell>,
    ): Promise<CreatedProject> {
      const response = await request.post('/api/projects', {
        data: {
          name,
          description: 'Risk-based Tier 3 catalog validation fixture',
          execution_backend: 'pytorch',
          keras_enabled: true,
          graph: modelGraph,
        },
      })
      expect(response.ok()).toBeTruthy()
      return response.json() as Promise<CreatedProject>
    }

    async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
      const response = await request.delete(`/api/projects/${id}`)
      expect([204, 404]).toContain(response.status())
    }

    async function addCatalogNode(page: Page, displayName: string): Promise<string> {
      const nodes = page.locator('[data-node-id]')
      const before = new Set(
        await nodes.evaluateAll((elements) => elements
          .map((element) => element.getAttribute('data-node-id'))
          .filter(Boolean) as string[]),
      )
      await page.getByRole('textbox', { name: 'Search nodes' }).fill(displayName)
      const item = page.locator('.catalog-item').filter({ hasText: displayName }).first()
      await expect(item, `Catalog node ${displayName} should be available`).toBeEnabled()
      await item.click()
      await expect(nodes).toHaveCount(before.size + 1)
      const after = await nodes.evaluateAll((elements) => elements
        .map((element) => element.getAttribute('data-node-id'))
        .filter(Boolean) as string[])
      const created = after.find((nodeId) => !before.has(nodeId))
      expect(created, `Builder should materialize ${displayName}`).toBeTruthy()
      await page.getByRole('textbox', { name: 'Search nodes' }).fill('')
      return created!
    }

    async function connectByPorts(
      page: Page,
      sourceLabel: string,
      sourcePort: string,
      targetLabel: string,
      targetPort: string,
      expectedConnectionCount: number,
    ): Promise<Record<string, unknown>> {
      const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
      const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
      const source = sourceNode.getByRole('button', { name: new RegExp(`^Port ${sourcePort} \\(output\\)`) })
      const target = targetNode.getByRole('button', { name: new RegExp(`^Port ${targetPort} \\(input\\)`) })
      await expect(source).toBeVisible()
      await expect(target).toBeVisible()
      await source.press('Enter')
      await expect(source).toHaveAttribute('aria-pressed', 'true')
      const responsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation/connection') && response.request().method() === 'POST',
      )
      await target.press('Enter')
      const response = await responsePromise
      expect(response.ok()).toBeTruthy()
      const payload = await response.json() as Record<string, unknown>
      expect(payload['valid'], `${sourceLabel} → ${targetLabel} should be accepted`).toBe(true)
      await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
      return payload
    }

    async function saveValidateAndRun(
      page: Page,
      request: APIRequestContext,
      fixture: CreatedProject,
      expectedNodes: number,
      expectedEdges: number,
      caseName: string,
      testInfo: TestInfo,
    ): Promise<Record<string, unknown>> {
      const validationResponsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Validate', exact: true }).click()
      const validationResponse = await validationResponsePromise
      expect(validationResponse.ok()).toBeTruthy()
      const validationPayload = await validationResponse.json() as Record<string, unknown>
      expect(validationPayload['valid']).toBe(true)
      await expect(page.locator('.validation-summary')).toContainText('Ready')
      await page.screenshot({ path: testInfo.outputPath(`${caseName}-builder-ready.png`), fullPage: true })

      const saveResponsePromise = page.waitForResponse(
        (response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT',
      )
      await page.getByRole('button', { name: 'Save model', exact: true }).click()
      const saveResponse = await saveResponsePromise
      expect(saveResponse.ok()).toBeTruthy()
      const saved = await saveResponse.json() as {
        current_revision: number
        validation_status: string
        graph: { nodes: unknown[]; edges: unknown[] }
      }
      expect(saved.validation_status).toBe('valid')
      expect(saved.graph.nodes).toHaveLength(expectedNodes)
      expect(saved.graph.edges).toHaveLength(expectedEdges)

      await page.reload()
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
      await expect(page.locator('[data-node-id]')).toHaveCount(expectedNodes)
      await mkdir(evidenceDir, { recursive: true })
      await page.screenshot({ path: path.join(evidenceDir, `${caseName}-builder-ready.png`), fullPage: true })

      await page.goto(`/models/${fixture.id}/validation`)
      await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
      await expect(page.getByText(/keras.*pytorch/i).first()).toBeVisible({ timeout: 75_000 })
      const submitResponsePromise = page.waitForResponse(
        (response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
      const submitResponse = await submitResponsePromise
      expect(submitResponse.ok()).toBeTruthy()
      const submitted = await submitResponse.json() as { id: string }
      await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 90_000 })
      let finalJob: Record<string, unknown> = {}
      await expect.poll(async () => {
        const response = await request.get(`/api/execution/jobs/${submitted.id}`)
        if (!response.ok()) return 'missing'
        finalJob = await response.json() as Record<string, unknown>
        return String(finalJob['state'])
      }, { timeout: 90_000 }).toBe('succeeded')

      await page.screenshot({ path: testInfo.outputPath(`${caseName}-runtime-succeeded.png`), fullPage: true })
      await page.screenshot({ path: path.join(evidenceDir, `${caseName}-runtime-succeeded.png`), fullPage: true })
      await writeFile(path.join(evidenceDir, `${caseName}.json`), JSON.stringify({
        case: caseName,
        project_id: fixture.id,
        revision: saved.current_revision,
        validation: validationPayload,
        submitted_job_id: submitted.id,
        final_job: finalJob,
      }, null, 2))
      return { validation: validationPayload, saved, final_job: finalJob }
    }

    test.describe('NW-T3-01 risk-based catalog slice', () => {
      test.describe.configure({ mode: 'serial' })
      test.setTimeout(180_000)

      test('recurrent sequence/state ports build, persist, and execute from the catalog', async ({ page, request }, testInfo) => {
        const fixture = await createProject(
          request,
          `Tier 3 recurrent catalog ${Date.now()}`,
          graphShell([
            node('sequence-input', 'input.sequence', 'Sequence input', { x: 48, y: 160 }, { shape: [4, 3], dtype: 'float32' }),
            node('sequence-output', 'output.sequence', 'Sequence output', { x: 1040, y: 120 }, {}),
            node('state-output', 'output.custom', 'State output', { x: 1040, y: 360 }, {}),
          ]),
        )
        const browserErrors: string[] = []
        page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
        page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
        try {
          await page.goto(`/builder/${fixture.id}`)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
          const rnnNodeId = await addCatalogNode(page, 'Simple RNN')
          const rnn = page.locator(`[data-node-id="${rnnNodeId}"]`)
          await rnn.click()
          const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
          await inspector.locator('label.inspector-field').filter({ hasText: 'Units' }).locator('input').fill('8')
          await inspector.locator('label.inspector-field').filter({ hasText: 'Return sequences' }).locator('input[type="checkbox"]').check()
          await inspector.locator('label.inspector-field').filter({ hasText: 'Return state' }).locator('input[type="checkbox"]').check()
          await expect(rnn.getByRole('button', { name: /^Port Hidden state \(output\)/ })).toBeVisible()

          await connectByPorts(page, 'Sequence input', 'Output', 'Simple RNN', 'Input', 1)
          await connectByPorts(page, 'Simple RNN', 'Output', 'Sequence output', 'Input', 2)
          await connectByPorts(page, 'Simple RNN', 'Hidden state', 'State output', 'Input', 3)
          await saveValidateAndRun(page, request, fixture, 4, 3, 'nw-t3-01-recurrent', testInfo)
          expect(browserErrors, 'recurrent catalog workflow should not emit browser errors').toEqual([])
          await writeFile(path.join(evidenceDir, 'nw-t3-01-recurrent-browser-errors.json'), JSON.stringify(browserErrors, null, 2))
        } finally {
          await deleteProject(request, fixture.id)
        }
      })

      test('GRU and LSTM sequence/state variants build, persist, and execute from the catalog', async ({ page, request }, testInfo) => {
        const fixture = await createProject(
          request,
          `Tier 3 recurrent variants ${Date.now()}`,
          graphShell([
            node('gru-input', 'input.sequence', 'GRU input', { x: 48, y: 80 }, { shape: [4, 3], dtype: 'float32' }),
            node('gru-sequence-output', 'output.sequence', 'GRU sequence output', { x: 1040, y: 40 }, {}),
            node('gru-state-output', 'output.custom', 'GRU hidden state output', { x: 1040, y: 220 }, {}),
            node('lstm-input', 'input.sequence', 'LSTM input', { x: 48, y: 520 }, { shape: [4, 3], dtype: 'float32' }),
            node('lstm-sequence-output', 'output.sequence', 'LSTM sequence output', { x: 1040, y: 480 }, {}),
            node('lstm-hidden-output', 'output.custom', 'LSTM hidden state output', { x: 1040, y: 660 }, {}),
            node('lstm-cell-output', 'output.custom', 'LSTM cell state output', { x: 1040, y: 840 }, {}),
          ]),
        )
        const browserErrors: string[] = []
        page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
        page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
        try {
          await page.goto(`/builder/${fixture.id}`)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()

          const gruNodeId = await addCatalogNode(page, 'GRU')
          const gru = page.locator(`[data-node-id="${gruNodeId}"]`)
          await gru.click()
          let inspector = page.getByRole('complementary', { name: 'Properties inspector' })
          await inspector.locator('label.inspector-field').filter({ hasText: 'Units' }).locator('input').fill('8')
          await inspector.locator('label.inspector-field').filter({ hasText: 'Return sequences' }).locator('input[type="checkbox"]').check()
          await inspector.locator('label.inspector-field').filter({ hasText: 'Return state' }).locator('input[type="checkbox"]').check()
          await expect(gru.getByRole('button', { name: /^Port Hidden state \(output\)/ })).toBeVisible()

          await connectByPorts(page, 'GRU input', 'Output', 'GRU', 'Input', 1)
          await connectByPorts(page, 'GRU', 'Output', 'GRU sequence output', 'Input', 2)
          await connectByPorts(page, 'GRU', 'Hidden state', 'GRU hidden state output', 'Input', 3)

          const lstmNodeId = await addCatalogNode(page, 'LSTM')
          const lstm = page.locator(`[data-node-id="${lstmNodeId}"]`)
          await lstm.click()
          inspector = page.getByRole('complementary', { name: 'Properties inspector' })
          await inspector.locator('label.inspector-field').filter({ hasText: 'Units' }).locator('input').fill('8')
          await inspector.locator('label.inspector-field').filter({ hasText: 'Return sequences' }).locator('input[type="checkbox"]').check()
          await inspector.locator('label.inspector-field').filter({ hasText: 'Return state' }).locator('input[type="checkbox"]').check()
          await expect(lstm.getByRole('button', { name: /^Port Hidden state \(output\)/ })).toBeVisible()
          await expect(lstm.getByRole('button', { name: /^Port Cell state \(output\)/ })).toBeVisible()

          await connectByPorts(page, 'LSTM input', 'Output', 'LSTM', 'Input', 4)
          await connectByPorts(page, 'LSTM', 'Output', 'LSTM sequence output', 'Input', 5)
          await connectByPorts(page, 'LSTM', 'Hidden state', 'LSTM hidden state output', 'Input', 6)
          await connectByPorts(page, 'LSTM', 'Cell state', 'LSTM cell state output', 'Input', 7)

          const result = await saveValidateAndRun(page, request, fixture, 9, 7, 'nw-t3-02-recurrent-variants', testInfo)
          const validationPayload = result['validation'] as {
            shape_summaries?: Array<{ node_id: string; parameter_count: number; outputs: Record<string, { shape: unknown[] }> }>
            total_parameters?: number
          }
          const summaryByNode = new Map(validationPayload.shape_summaries?.map((summary) => [summary.node_id, summary]))
          expect(summaryByNode.get(gruNodeId)?.parameter_count).toBe(312)
          expect(summaryByNode.get(gruNodeId)?.outputs['output']?.shape).toEqual([null, 4, 8])
          expect(summaryByNode.get(gruNodeId)?.outputs['hidden_state']?.shape).toEqual([null, 8])
          expect(summaryByNode.get(lstmNodeId)?.parameter_count).toBe(384)
          expect(summaryByNode.get(lstmNodeId)?.outputs['output']?.shape).toEqual([null, 4, 8])
          expect(summaryByNode.get(lstmNodeId)?.outputs['hidden_state']?.shape).toEqual([null, 8])
          expect(summaryByNode.get(lstmNodeId)?.outputs['cell_state']?.shape).toEqual([null, 8])
          expect(validationPayload.total_parameters).toBe(696)

          const finalJob = result['final_job'] as { result?: { parameter_count?: number; outputs?: unknown[] } }
          expect(finalJob.result?.parameter_count).toBe(696)
          expect(finalJob.result?.outputs).toHaveLength(5)
          expect(browserErrors, 'recurrent variant workflow should not emit browser errors').toEqual([])
          await writeFile(path.join(evidenceDir, 'nw-t3-02-recurrent-variants-browser-errors.json'), JSON.stringify(browserErrors, null, 2))
        } finally {
          await deleteProject(request, fixture.id)
        }
      })

      test('tensor-list merge fan-in and axis parameter build, persist, and execute from the catalog', async ({ page, request }, testInfo) => {
        const fixture = await createProject(
          request,
          `Tier 3 merge catalog ${Date.now()}`,
          graphShell([
            node('left-input', 'input.tensor', 'Left input', { x: 48, y: 96 }, { shape: [4], dtype: 'float32' }),
            node('right-input', 'input.tensor', 'Right input', { x: 48, y: 320 }, { shape: [4], dtype: 'float32' }),
            node('merge-output', 'output.custom', 'Merged output', { x: 1040, y: 200 }, {}),
          ]),
        )
        const browserErrors: string[] = []
        page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
        page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
        try {
          await page.goto(`/builder/${fixture.id}`)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
          const mergeNodeId = await addCatalogNode(page, 'Concatenate')
          const merge = page.locator(`[data-node-id="${mergeNodeId}"]`)
          await merge.click()
          const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
          await inspector.locator('label.inspector-field').filter({ hasText: 'Axis' }).locator('input').fill('-1')

          await connectByPorts(page, 'Left input', 'Output', 'Concatenate', 'Inputs', 1)
          await connectByPorts(page, 'Right input', 'Output', 'Concatenate', 'Inputs', 2)
          await connectByPorts(page, 'Concatenate', 'Output', 'Merged output', 'Input', 3)
          const result = await saveValidateAndRun(page, request, fixture, 4, 3, 'nw-t3-01-merge', testInfo)
          const validationPayload = result['validation'] as {
            shape_summaries?: Array<{ node_id: string; outputs: Record<string, { shape: unknown[] }> }>
          }
          const mergeSummary = validationPayload.shape_summaries?.find((summary) => summary.node_id === mergeNodeId)
          expect(mergeSummary?.outputs['output']?.shape).toEqual([null, 8])
          expect(browserErrors, 'merge catalog workflow should not emit browser errors').toEqual([])
          await writeFile(path.join(evidenceDir, 'nw-t3-01-merge-browser-errors.json'), JSON.stringify(browserErrors, null, 2))
        } finally {
          await deleteProject(request, fixture.id)
        }
      })

      test('bidirectional recurrent merge modes infer focused output shapes', async ({ page, request }, testInfo) => {
        const fixture = await createProject(
          request,
          `Tier 3 bidirectional merge modes ${Date.now()}`,
          graphShell([
            node('sequence-input', 'input.sequence', 'Sequence input', { x: 48, y: 160 }, { shape: [4, 3], dtype: 'float32' }),
            node('sequence-output', 'output.sequence', 'Sequence output', { x: 1040, y: 160 }, {}),
          ]),
        )
        const browserErrors: string[] = []
        page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
        page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
        try {
          await page.goto(`/builder/${fixture.id}`)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
          const bidirectionalNodeId = await addCatalogNode(page, 'Bidirectional Recurrent Layer')
          const bidirectional = page.locator(`[data-node-id="${bidirectionalNodeId}"]`)
          await bidirectional.click()
          const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
          await inspector.locator('label.inspector-field').filter({ hasText: 'Cell' }).locator('select').selectOption('lstm')
          await inspector.locator('label.inspector-field').filter({ hasText: 'Units' }).locator('input').fill('4')
          await inspector.locator('label.inspector-field').filter({ hasText: 'Return sequences' }).locator('input[type="checkbox"]').check()
          await inspector.locator('label.inspector-field').filter({ hasText: 'Merge mode' }).locator('select').selectOption('concat')

          await connectByPorts(page, 'Sequence input', 'Output', 'Bidirectional Recurrent Layer', 'Input', 1)
          await connectByPorts(page, 'Bidirectional Recurrent Layer', 'Output', 'Sequence output', 'Input', 2)
          await expect(page.locator('.validation-summary')).toContainText('Ready', { timeout: 30_000 })

          const expectedShapes: Record<string, (number | null)[]> = {
            sum: [null, 4, 4],
            mul: [null, 4, 4],
            ave: [null, 4, 4],
            concat: [null, 4, 8],
          }
          const observed: Record<string, unknown> = {}
          for (const mergeMode of ['sum', 'mul', 'ave', 'concat']) {
            const validationResponsePromise = page.waitForResponse(
              (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
            )
            await inspector.locator('label.inspector-field').filter({ hasText: 'Merge mode' }).locator('select').selectOption(mergeMode)
            const validationResponse = await validationResponsePromise
            expect(validationResponse.ok()).toBeTruthy()
            const validationPayload = await validationResponse.json() as {
              valid: boolean
              total_parameters?: number
              shape_summaries?: Array<{ node_id: string; parameter_count: number; outputs: Record<string, { shape: unknown[] }> }>
            }
            expect(validationPayload.valid).toBe(true)
            const summary = validationPayload.shape_summaries?.find((item) => item.node_id === bidirectionalNodeId)
            expect(summary?.parameter_count).toBe(256)
            expect(summary?.outputs['output']?.shape).toEqual(expectedShapes[mergeMode])
            expect(validationPayload.total_parameters).toBe(256)
            observed[mergeMode] = { shape: summary?.outputs['output']?.shape, parameter_count: summary?.parameter_count }
          }
          await page.screenshot({ path: testInfo.outputPath('nw-t3-wrapper-bidirectional-merge-modes.png'), fullPage: true })
          await mkdir(evidenceDir, { recursive: true })
          await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-wrapper-bidirectional-merge-modes.png'), fullPage: true })
          await writeFile(path.join(evidenceDir, 'nw-t3-wrapper-bidirectional-merge-modes.json'), JSON.stringify({
            case: 'nw-t3-wrapper-bidirectional-merge-modes',
            project_id: fixture.id,
            observed,
          }, null, 2))
          expect(browserErrors, 'bidirectional merge-mode workflow should not emit browser errors').toEqual([])
          await writeFile(path.join(evidenceDir, 'nw-t3-wrapper-bidirectional-merge-modes-browser-errors.json'), JSON.stringify(browserErrors, null, 2))
        } finally {
          await deleteProject(request, fixture.id)
        }
      })

      test('bidirectional LSTM wrapper builds, persists, and executes on the selected CPU lane', async ({ page, request }, testInfo) => {
        const fixture = await createProject(
          request,
          `Tier 3 bidirectional LSTM ${Date.now()}`,
          graphShell([
            node('sequence-input', 'input.sequence', 'Sequence input', { x: 48, y: 160 }, { shape: [4, 3], dtype: 'float32' }),
            node('sequence-output', 'output.sequence', 'Sequence output', { x: 1040, y: 160 }, {}),
          ]),
        )
        const browserErrors: string[] = []
        page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
        page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
        try {
          await page.goto(`/builder/${fixture.id}`)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
          const bidirectionalNodeId = await addCatalogNode(page, 'Bidirectional Recurrent Layer')
          const bidirectional = page.locator(`[data-node-id="${bidirectionalNodeId}"]`)
          await bidirectional.click()
          const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
          await inspector.locator('label.inspector-field').filter({ hasText: 'Cell' }).locator('select').selectOption('lstm')
          await inspector.locator('label.inspector-field').filter({ hasText: 'Units' }).locator('input').fill('4')
          await inspector.locator('label.inspector-field').filter({ hasText: 'Return sequences' }).locator('input[type="checkbox"]').check()
          await inspector.locator('label.inspector-field').filter({ hasText: 'Merge mode' }).locator('select').selectOption('concat')

          await connectByPorts(page, 'Sequence input', 'Output', 'Bidirectional Recurrent Layer', 'Input', 1)
          await connectByPorts(page, 'Bidirectional Recurrent Layer', 'Output', 'Sequence output', 'Input', 2)
          const result = await saveValidateAndRun(page, request, fixture, 3, 2, 'nw-t3-wrapper-bidirectional-lstm', testInfo)
          const validationPayload = result['validation'] as {
            shape_summaries?: Array<{ node_id: string; parameter_count: number; outputs: Record<string, { shape: unknown[] }> }>
            total_parameters?: number
          }
          const summary = validationPayload.shape_summaries?.find((item) => item.node_id === bidirectionalNodeId)
          expect(summary?.parameter_count).toBe(256)
          expect(summary?.outputs['output']?.shape).toEqual([null, 4, 8])
          expect(validationPayload.total_parameters).toBe(256)

          const finalJob = result['final_job'] as { result?: { parameter_count?: number; outputs?: unknown[] } }
          expect(finalJob.result?.parameter_count).toBe(256)
          expect(finalJob.result?.outputs).toHaveLength(1)
          expect(browserErrors, 'bidirectional LSTM workflow should not emit browser errors').toEqual([])
          await writeFile(path.join(evidenceDir, 'nw-t3-wrapper-bidirectional-lstm-browser-errors.json'), JSON.stringify(browserErrors, null, 2))
        } finally {
          await deleteProject(request, fixture.id)
        }
      })

      test('recurrent stack repairs a blocked layer count, persists, and executes', async ({ page, request }, testInfo) => {
        const fixture = await createProject(
          request,
          `Tier 3 recurrent stack ${Date.now()}`,
          graphShell([
            node('sequence-input', 'input.sequence', 'Sequence input', { x: 48, y: 160 }, { shape: [4, 3], dtype: 'float32' }),
            node('sequence-output', 'output.sequence', 'Sequence output', { x: 1040, y: 160 }, {}),
          ]),
        )
        const browserErrors: string[] = []
        page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
        page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
        try {
          await page.goto(`/builder/${fixture.id}`)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
          const stackNodeId = await addCatalogNode(page, 'Recurrent Stack')
          const stack = page.locator(`[data-node-id="${stackNodeId}"]`)
          await stack.click()
          const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
          await inspector.locator('label.inspector-field').filter({ hasText: 'Cell' }).locator('select').selectOption('lstm')
          await inspector.locator('label.inspector-field').filter({ hasText: 'Units' }).locator('input').fill('4')
          await inspector.locator('label.inspector-field').filter({ hasText: 'Return sequences' }).locator('input[type="checkbox"]').check()
          await connectByPorts(page, 'Sequence input', 'Output', 'Recurrent Stack', 'Input', 1)
          await connectByPorts(page, 'Recurrent Stack', 'Output', 'Sequence output', 'Input', 2)
          await expect(page.locator('.validation-summary')).toContainText('Ready', { timeout: 30_000 })

          const layers = inspector.locator('label.inspector-field').filter({ hasText: 'Layers' }).locator('input')
          await layers.fill('0')
          await expect(layers).toHaveValue('0')
          await expect(page.locator('.validation-summary')).toContainText('Blocked', { timeout: 30_000 })
          const blockedNotice = page.getByRole('button', { name: 'Model graph has blocking findings.', exact: true })
          await expect(blockedNotice).toBeVisible()
          await blockedNotice.click()
          await page.locator('.validation-summary').click()
          await expect(page.locator('.diagnostic-list')).toContainText('Recurrent stack layers must be positive')
          const invalidDiagnostic = await page.locator('.diagnostic-list').innerText()
          await mkdir(evidenceDir, { recursive: true })
          await page.screenshot({ path: testInfo.outputPath('nw-t3-wrapper-stack-invalid.png'), fullPage: true })
          await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-wrapper-stack-invalid.png'), fullPage: true })
          await writeFile(path.join(evidenceDir, 'nw-t3-wrapper-stack-invalid.json'), JSON.stringify({
            case: 'nw-t3-wrapper-stack-invalid',
            project_id: fixture.id,
            layers: 0,
            validation_status: 'blocked',
            diagnostic: invalidDiagnostic,
          }, null, 2))

          await layers.fill('2')
          await expect(layers).toHaveValue('2')
          const result = await saveValidateAndRun(page, request, fixture, 3, 2, 'nw-t3-wrapper-stack-lstm', testInfo)
          const validationPayload = result['validation'] as {
            shape_summaries?: Array<{ node_id: string; parameter_count: number; outputs: Record<string, { shape: unknown[] }> }>
            total_parameters?: number
          }
          const summary = validationPayload.shape_summaries?.find((item) => item.node_id === stackNodeId)
          expect(summary?.parameter_count).toBe(272)
          expect(summary?.outputs['output']?.shape).toEqual([null, 4, 4])
          expect(validationPayload.total_parameters).toBe(272)

          const finalJob = result['final_job'] as { result?: { parameter_count?: number; outputs?: unknown[] } }
          expect(finalJob.result?.parameter_count).toBe(272)
          expect(finalJob.result?.outputs).toHaveLength(1)
          expect(browserErrors, 'recurrent stack workflow should not emit browser errors').toEqual([])
          await writeFile(path.join(evidenceDir, 'nw-t3-wrapper-stack-browser-errors.json'), JSON.stringify(browserErrors, null, 2))
        } finally {
          await deleteProject(request, fixture.id)
        }
      })
    })
  })
  test.describe('node functional matrix.spec', () => {
    type CreatedProject = { id: string; current_revision: number; name: string }
    type GraphNode = {
      id: string
      definition_id: string
      definition_version: number
      label: string | null
      position: { x: number; y: number }
      parameters: Record<string, unknown>
    }
    type GraphEdge = {
      id: string
      source_node_id: string
      source_port_id: string
      target_node_id: string
      target_port_id: string
    }

    type CaseDefinition = {
      name: string
      inputDefinition: 'input.sequence' | 'input.image' | 'input.tensor'
      inputShape: number[]
      catalogNodes: string[]
      kerasEnabled?: boolean
    }

    function graphShell(nodes: GraphNode[], edges: GraphEdge[] = []) {
      return {
        schema_version: 6,
        nodes,
        edges,
        model_configuration: {},
        export_settings: {
          include_comments: true,
          include_shape_assertions: true,
          preferred_binary_format: 'keras',
        },
        future_training_configuration: {},
      }
    }

    function inputOutputGraph(inputDefinition: CaseDefinition['inputDefinition'], inputShape: number[]) {
      return graphShell([
        {
          id: 'matrix-input',
          definition_id: inputDefinition,
          definition_version: 1,
          label: 'Matrix input',
          position: { x: 64, y: 180 },
          parameters: { shape: inputShape, dtype: 'float32' },
        },
        {
          id: 'matrix-output',
          definition_id: 'output.custom',
          definition_version: 1,
          label: 'Matrix output',
          position: { x: 1200, y: 180 },
          parameters: {},
        },
      ])
    }

    async function createProject(
      request: APIRequestContext,
      name: string,
      modelGraph: ReturnType<typeof graphShell>,
      kerasEnabled = true,
    ): Promise<CreatedProject> {
      const response = await request.post('/api/projects', {
        data: {
          name,
          description: 'Focused CNN and activation browser validation fixture',
          execution_backend: 'pytorch',
          keras_enabled: kerasEnabled,
          graph: modelGraph,
        },
      })
      expect(response.ok()).toBeTruthy()
      return response.json() as Promise<CreatedProject>
    }

    async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
      const response = await request.delete(`/api/projects/${id}`)
      expect([204, 404]).toContain(response.status())
    }

    async function addCatalogNode(page: Page, displayName: string): Promise<string> {
      const nodes = page.locator('[data-node-id]')
      const before = new Set(await nodes.evaluateAll((elements) => elements.map((element) => element.getAttribute('data-node-id')).filter(Boolean) as string[]))
      await page.getByRole('textbox', { name: 'Search nodes' }).fill(displayName)
      const item = page.locator('.catalog-item').filter({ hasText: displayName }).first()
      await expect(item, `Catalog node ${displayName} should be available for the selected profile`).toBeEnabled()
      await item.click()
      await expect(nodes).toHaveCount(before.size + 1)
      const after = await nodes.evaluateAll((elements) => elements.map((element) => element.getAttribute('data-node-id')).filter(Boolean) as string[])
      const created = after.find((nodeId) => !before.has(nodeId))
      expect(created, `Builder should materialize ${displayName}`).toBeTruthy()
      await page.getByRole('textbox', { name: 'Search nodes' }).fill('')
      return created!
    }

    async function connectByLabels(
      page: Page,
      sourceLabel: string,
      targetLabel: string,
      expectedConnectionCount: number,
    ): Promise<void> {
      const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
      const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
      const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
      const target = targetNode.getByRole('button', { name: /^Port Input \(input\)/ })
      await expect(source).toBeVisible()
      await expect(target).toBeVisible()
      await source.press('Enter')
      await expect(source).toHaveAttribute('aria-pressed', 'true')
      const validationResponsePromise = page.waitForResponse(
        (response) => response.url().includes('/api/validation/connection') && response.request().method() === 'POST',
      )
      await target.press('Enter')
      const response = await validationResponsePromise
      expect(response.ok()).toBeTruthy()
      const payload = await response.json() as { valid: boolean; diagnostics: unknown[] }
      console.log(`[NODE_MATRIX_CONNECTION] ${sourceLabel} -> ${targetLabel} ${JSON.stringify(payload)}`)
      expect(payload.valid, `${sourceLabel} → ${targetLabel} should be accepted by connection validation`).toBeTruthy()
      await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
    }

    async function capture(page: Page, testInfo: TestInfo, name: string): Promise<void> {
      await page.screenshot({ path: testInfo.outputPath(`${name}.png`), fullPage: true })
    }

    async function validateSaveAndRun(
      page: Page,
      request: APIRequestContext,
      fixture: CreatedProject,
      expectedNodes: number,
      expectedEdges: number,
      caseName: string,
      kerasEnabled: boolean,
      testInfo: TestInfo,
    ): Promise<void> {
      const validationResponsePromise = page.waitForResponse(
        (response) => response.url().includes('/api/validation') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Validate', exact: true }).click()
      const validationResponse = await validationResponsePromise
      expect(validationResponse.ok()).toBeTruthy()
      const validationPayload = await validationResponse.json()
      console.log(`[NODE_MATRIX_STATIC] ${caseName} ${JSON.stringify(validationPayload)}`)
      await expect(page.locator('.validation-summary')).toContainText('Ready')
      await capture(page, testInfo, `${caseName}-builder-ready`)

      const saveResponsePromise = page.waitForResponse(
        (response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT',
      )
      await page.getByRole('button', { name: 'Save model', exact: true }).click()
      const saveResponse = await saveResponsePromise
      expect(saveResponse.ok()).toBeTruthy()
      const saved = await saveResponse.json() as {
        current_revision: number
        validation_status: string
        graph: { nodes: unknown[]; edges: unknown[] }
      }
      expect(saved.validation_status).toBe('valid')
      expect(saved.graph.nodes).toHaveLength(expectedNodes)
      expect(saved.graph.edges).toHaveLength(expectedEdges)

      await page.reload()
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
      await expect(page.locator('[data-node-id]')).toHaveCount(expectedNodes)

      await page.goto(`/models/${fixture.id}/validation`)
      await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
      const profilePattern = kerasEnabled ? /keras.*pytorch/i : /native.*pytorch/i
      await expect(page.getByText(profilePattern).first()).toBeVisible({ timeout: 75_000 })

      const submitResponsePromise = page.waitForResponse(
        (response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
      const submitResponse = await submitResponsePromise
      expect(submitResponse.ok()).toBeTruthy()
      const submitted = await submitResponse.json() as { id: string }

      await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 90_000 })
      let finalJob: Record<string, unknown> = {}
      await expect.poll(async () => {
        const response = await request.get(`/api/execution/jobs/${submitted.id}`)
        if (!response.ok()) return 'missing'
        finalJob = await response.json() as Record<string, unknown>
        return String(finalJob['state'])
      }, { timeout: 90_000 }).toBe('succeeded')

      console.log(`[NODE_MATRIX_RUNTIME] ${caseName} ${JSON.stringify(finalJob)}`)
      await capture(page, testInfo, `${caseName}-runtime-succeeded`)
    }

    async function runCnnCase(
      page: Page,
      request: APIRequestContext,
      testInfo: TestInfo,
      definition: CaseDefinition,
    ): Promise<void> {
      const kerasEnabled = definition.kerasEnabled ?? true
      const fixture = await createProject(
        request,
        `Node matrix ${definition.name} ${Date.now()}`,
        inputOutputGraph(definition.inputDefinition, definition.inputShape),
        kerasEnabled,
      )
      const browserErrors: string[] = []
      page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
      page.on('console', (message) => {
        if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`)
      })

      try {
        await page.goto(`/builder/${fixture.id}`)
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        await expect(page.locator('[data-node-id]')).toHaveCount(2)

        for (const displayName of definition.catalogNodes) {
          await addCatalogNode(page, displayName)
        }

        const chainLabels = ['Matrix input', ...definition.catalogNodes, 'Matrix output']
        for (let index = 0; index < chainLabels.length - 1; index += 1) {
          await connectByLabels(page, chainLabels[index]!, chainLabels[index + 1]!, index + 1)
        }

        await validateSaveAndRun(
          page,
          request,
          fixture,
          chainLabels.length,
          chainLabels.length - 1,
          definition.name,
          kerasEnabled,
          testInfo,
        )
        expect(browserErrors, `${definition.name} should not emit browser errors`).toEqual([])
        console.log(`[NODE_MATRIX_BROWSER_ERRORS] ${definition.name} ${JSON.stringify(browserErrors)}`)
      } finally {
        await deleteProject(request, fixture.id)
      }
    }

    async function runActivationCase(
      page: Page,
      request: APIRequestContext,
      testInfo: TestInfo,
      activation: string,
    ): Promise<void> {
      const caseName = `activation-${activation}`
      const fixture = await createProject(
        request,
        `Node matrix ${caseName} ${Date.now()}`,
        inputOutputGraph('input.tensor', [8]),
        true,
      )
      const browserErrors: string[] = []
      page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
      page.on('console', (message) => {
        if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`)
      })

      try {
        await page.goto(`/builder/${fixture.id}`)
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        await expect(page.locator('[data-node-id]')).toHaveCount(2)
        const activationNodeId = await addCatalogNode(page, 'Activation')
        await page.locator(`[data-node-id="${activationNodeId}"]`).click()
        const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
        const activationField = inspector.locator('label.inspector-field').filter({ hasText: 'Activation' })
        await expect(activationField.locator('select')).toBeVisible()
        await activationField.locator('select').selectOption(activation)
        await connectByLabels(page, 'Matrix input', 'Activation', 1)
        await connectByLabels(page, 'Activation', 'Matrix output', 2)
        await validateSaveAndRun(page, request, fixture, 3, 2, caseName, true, testInfo)
        expect(browserErrors, `${caseName} should not emit browser errors`).toEqual([])
        console.log(`[NODE_MATRIX_BROWSER_ERRORS] ${caseName} ${JSON.stringify(browserErrors)}`)
      } finally {
        await deleteProject(request, fixture.id)
      }
    }

    test.describe('focused CNN and activation node functional matrix', () => {
      test.setTimeout(150_000)

      const cnnCases: CaseDefinition[] = [
        {
          name: 'cnn-1d-keras-pytorch',
          inputDefinition: 'input.sequence',
          inputShape: [12, 3],
          catalogNodes: ['Conv1D', 'Average Pooling 1D', 'Global Pooling 1D'],
        },
        {
          name: 'cnn-2d-keras-pytorch',
          inputDefinition: 'input.image',
          inputShape: [16, 16, 3],
          catalogNodes: ['Conv2D', 'Max Pooling 2D', 'Global Pooling 2D'],
        },
        {
          name: 'cnn-3d-keras-pytorch',
          inputDefinition: 'input.tensor',
          inputShape: [8, 8, 8, 1],
          catalogNodes: ['Conv3D', 'Average Pooling 3D', 'Global Pooling 3D'],
        },
        {
          name: 'cnn-2d-native-pytorch',
          inputDefinition: 'input.image',
          inputShape: [16, 16, 3],
          catalogNodes: ['Conv2D', 'Max Pooling 2D', 'Global Pooling 2D'],
          kerasEnabled: false,
        },
      ]

      for (const definition of cnnCases) {
        test(`${definition.name} builds in the Builder and completes a real functional validation`, async ({ page, request }, testInfo) => {
          await runCnnCase(page, request, testInfo, definition)
        })
      }

      for (const activation of ['relu', 'gelu', 'sigmoid', 'tanh', 'softmax', 'linear']) {
        test(`Activation ${activation} builds in the Builder and completes a real functional validation`, async ({ page, request }, testInfo) => {
          await runActivationCase(page, request, testInfo, activation)
        })
      }
    })
  })
  test.describe('recurrent nodes slice.spec', () => {
    type CreatedProject = { id: string; current_revision: number; name: string }
    type GraphNode = {
      id: string
      definition_id: string
      definition_version: number
      label: string
      position: { x: number; y: number }
      parameters: Record<string, unknown>
    }
    type ShapeSummary = {
      node_id: string
      parameter_count: number
      outputs: Record<string, { shape: Array<number | null> }>
    }
    type ValidationPayload = {
      valid: boolean
      diagnostics: unknown[]
      shape_summaries: ShapeSummary[]
      total_parameters: number
    }
    type RuntimeJob = {
      state: string
      worker?: { api: string; backend: string; device_id: string }
      result?: {
        status: string
        outputs: Array<{ shape: number[] }>
        parameter_count: number
      }
    }
    type NativeCase = {
      name: string
      catalogNode: string
      definitionId: string
      parameters: Record<string, unknown>
      staticOutputShape: Array<number | null>
      runtimeOutputShape: number[]
      staticParameterCount: number
      runtimeParameterCount: number
    }

    const evidenceDir = path.resolve(
      process.env.NW_VALIDATION_EVIDENCE_DIR
        ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260928', 'evidence', 'nw-t3-05-recurrent'),
    )

    function graphShell(nodes: GraphNode[]) {
      return {
        schema_version: 6,
        nodes,
        edges: [],
        model_configuration: {},
        export_settings: {
          include_comments: true,
          include_shape_assertions: true,
          preferred_binary_format: 'keras',
        },
        future_training_configuration: {},
      }
    }

    function node(
      id: string,
      definitionId: string,
      label: string,
      position: { x: number; y: number },
      parameters: Record<string, unknown>,
    ): GraphNode {
      return {
        id,
        definition_id: definitionId,
        definition_version: 1,
        label,
        position,
        parameters,
      }
    }

    async function createProject(
      request: APIRequestContext,
      name: string,
      modelGraph: ReturnType<typeof graphShell>,
    ): Promise<CreatedProject> {
      const response = await request.post('/api/projects', {
        data: {
          name,
          description: 'NW-T3-05 recurrent catalog validation fixture',
          execution_backend: 'pytorch',
          keras_enabled: false,
          graph: modelGraph,
        },
      })
      expect(response.ok()).toBeTruthy()
      return response.json() as Promise<CreatedProject>
    }

    async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
      const response = await request.delete(`/api/projects/${id}`)
      expect([204, 404]).toContain(response.status())
    }

    async function addCatalogNode(page: Page, displayName: string): Promise<string> {
      const nodes = page.locator('[data-node-id]')
      const before = new Set(
        await nodes.evaluateAll((elements) => elements
          .map((element) => element.getAttribute('data-node-id'))
          .filter(Boolean) as string[]),
      )
      await page.getByRole('textbox', { name: 'Search nodes' }).fill(displayName)
      const item = page.locator('.catalog-item').filter({ hasText: displayName }).first()
      await expect(item, `Catalog node ${displayName} should be available`).toBeEnabled()
      await item.click()
      await expect(nodes).toHaveCount(before.size + 1)
      const after = await nodes.evaluateAll((elements) => elements
        .map((element) => element.getAttribute('data-node-id'))
        .filter(Boolean) as string[])
      const created = after.find((nodeId) => !before.has(nodeId))
      expect(created, `Builder should materialize ${displayName}`).toBeTruthy()
      await page.getByRole('textbox', { name: 'Search nodes' }).fill('')
      return created!
    }

    async function connectByPorts(
      page: Page,
      sourceLabel: string,
      targetLabel: string,
      expectedConnectionCount: number,
    ): Promise<void> {
      const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
      const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
      const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
      const target = targetNode.getByRole('button', { name: /^Port Input \(input\)/ })
      await expect(source).toBeVisible()
      await expect(target).toBeVisible()
      await source.press('Enter')
      await expect(source).toHaveAttribute('aria-pressed', 'true')
      const responsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation/connection') && response.request().method() === 'POST',
      )
      await target.press('Enter')
      const response = await responsePromise
      expect(response.ok()).toBeTruthy()
      const payload = await response.json() as { valid: boolean }
      expect(payload.valid, `${sourceLabel} → ${targetLabel} should be accepted`).toBe(true)
      await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
    }

    const nativeCases: NativeCase[] = [
      {
        name: 'nw-t3-05-simple-rnn-native-pytorch',
        catalogNode: 'Simple RNN',
        definitionId: 'recurrent.simple_rnn',
        parameters: { units: 4, return_sequences: true },
        staticOutputShape: [null, 4, 4],
        runtimeOutputShape: [1, 4, 4],
        staticParameterCount: 32,
        runtimeParameterCount: 36,
      },
      {
        name: 'nw-t3-05-gru-native-pytorch',
        catalogNode: 'GRU',
        definitionId: 'recurrent.gru',
        parameters: { units: 4, return_sequences: true },
        staticOutputShape: [null, 4, 4],
        runtimeOutputShape: [1, 4, 4],
        staticParameterCount: 108,
        runtimeParameterCount: 108,
      },
      {
        name: 'nw-t3-05-lstm-native-pytorch',
        catalogNode: 'LSTM',
        definitionId: 'recurrent.lstm',
        parameters: { units: 4, return_sequences: true },
        staticOutputShape: [null, 4, 4],
        runtimeOutputShape: [1, 4, 4],
        staticParameterCount: 128,
        runtimeParameterCount: 144,
      },
      {
        name: 'nw-t3-05-bidirectional-native-pytorch',
        catalogNode: 'Bidirectional Recurrent Layer',
        definitionId: 'recurrent.bidirectional',
        parameters: { cell: 'simple_rnn', units: 4, return_sequences: true, merge_mode: 'concat' },
        staticOutputShape: [null, 4, 8],
        runtimeOutputShape: [1, 4, 8],
        staticParameterCount: 64,
        runtimeParameterCount: 72,
      },
      {
        name: 'nw-t3-05-stack-native-pytorch',
        catalogNode: 'Recurrent Stack',
        definitionId: 'recurrent.stack',
        parameters: { cell: 'simple_rnn', units: 4, layers: 2, return_sequences: true },
        staticOutputShape: [null, 4, 4],
        runtimeOutputShape: [1, 4, 4],
        staticParameterCount: 68,
        runtimeParameterCount: 76,
      },
    ]

    test.describe('NW-T3-05 recurrent catalog slice', () => {
      test.describe.configure({ mode: 'serial' })
      test.setTimeout(180_000)

      test('live catalog renders all recurrent definitions with native support mappings', async ({ page, request }) => {
        const response = await request.get('/api/nodes/catalog')
        expect(response.ok()).toBeTruthy()
        const payload = await response.json() as {
          schema_version: number
          nodes: Array<{
            id: string
            display_name: string
            category: string
            implementations: Record<string, { supported?: boolean }>
          }>
        }
        const selected = payload.nodes.filter((item) => item.category === 'recurrent')
        expect(selected.map((item) => item.id).sort()).toEqual(nativeCases.map((item) => item.definitionId).sort())
        expect(selected).toHaveLength(5)
        for (const definition of selected) {
          expect(definition.implementations.keras?.supported).toBe(true)
          expect(definition.implementations.pytorch?.supported).toBe(true)
        }

        await page.goto('/builder')
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        await page.getByRole('button', { name: /Recurrent 5 nodes/ }).click()
        for (const candidate of nativeCases) {
          await expect(page.getByRole('button', { name: new RegExp(`${candidate.catalogNode} recurrent\\.`) })).toBeEnabled()
        }
        await mkdir(evidenceDir, { recursive: true })
        await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-05-recurrent-catalog-rendered.png'), fullPage: true })
        await writeFile(path.join(evidenceDir, 'nw-t3-05-live-catalog.json'), JSON.stringify({
          schema_version: payload.schema_version,
          catalog_count: payload.nodes.length,
          recurrent: selected,
        }, null, 2))
      })

      for (const candidate of nativeCases) {
        test(`${candidate.name} builds, persists, and completes exact-profile functional validation`, async ({ page, request }, testInfo) => {
          const fixture = await createProject(
            request,
            `Tier 3 native recurrent ${candidate.catalogNode} ${Date.now()}`,
            graphShell([
              node('sequence-input', 'input.sequence', 'Sequence input', { x: 48, y: 160 }, { shape: [4, 3], dtype: 'float32' }),
              node('sequence-output', 'output.sequence', 'Sequence output', { x: 1040, y: 160 }, {}),
            ]),
          )
          const browserErrors: string[] = []
          page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
          page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
          try {
            await page.goto(`/builder/${fixture.id}`)
            await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
            await expect(page.getByLabel('Enable Keras', { exact: true })).not.toBeChecked()
            const nodeId = await addCatalogNode(page, candidate.catalogNode)
            const materialized = page.locator(`[data-node-id="${nodeId}"]`)
            await materialized.click()
            const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
            await inspector.locator('label.inspector-field').filter({ hasText: 'Units' }).locator('input').fill(String(candidate.parameters.units))
            if (candidate.parameters.return_sequences === true) {
              await inspector.locator('label.inspector-field').filter({ hasText: 'Return sequences' }).locator('input[type="checkbox"]').check()
            }
            if (candidate.definitionId === 'recurrent.bidirectional' || candidate.definitionId === 'recurrent.stack') {
              await inspector.locator('label.inspector-field').filter({ hasText: 'Cell' }).locator('select').selectOption(String(candidate.parameters.cell))
            }
            if (candidate.definitionId === 'recurrent.bidirectional') {
              await inspector.locator('label.inspector-field').filter({ hasText: 'Merge mode' }).locator('select').selectOption(String(candidate.parameters.merge_mode))
            }
            if (candidate.definitionId === 'recurrent.stack') {
              await inspector.locator('label.inspector-field').filter({ hasText: 'Layers' }).locator('input').fill(String(candidate.parameters.layers))
            }
            await connectByPorts(page, 'Sequence input', candidate.catalogNode, 1)
            await connectByPorts(page, candidate.catalogNode, 'Sequence output', 2)

            const validationResponsePromise = page.waitForResponse(
              (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
            )
            await page.getByRole('button', { name: 'Validate', exact: true }).click()
            const validationResponse = await validationResponsePromise
            expect(validationResponse.ok()).toBeTruthy()
            const validationPayload = await validationResponse.json() as ValidationPayload
            expect(validationPayload.valid).toBe(true)
            expect(validationPayload.diagnostics).toEqual([])
            const summary = validationPayload.shape_summaries.find((item) => item.node_id === nodeId)
            expect(summary?.outputs.output?.shape).toEqual(candidate.staticOutputShape)
            expect(summary?.parameter_count).toBe(candidate.staticParameterCount)
            expect(validationPayload.total_parameters).toBe(candidate.staticParameterCount)
            await expect(page.locator('.validation-summary')).toContainText('Ready')
            await page.screenshot({ path: testInfo.outputPath(`${candidate.name}-builder-ready.png`), fullPage: true })

            const saveResponsePromise = page.waitForResponse(
              (response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT',
            )
            await page.getByRole('button', { name: 'Save model', exact: true }).click()
            const saveResponse = await saveResponsePromise
            expect(saveResponse.ok()).toBeTruthy()
            const saved = await saveResponse.json() as {
              current_revision: number
              validation_status: string
              graph: { nodes: unknown[]; edges: unknown[] }
            }
            expect(saved.validation_status).toBe('valid')
            expect(saved.graph.nodes).toHaveLength(3)
            expect(saved.graph.edges).toHaveLength(2)

            await page.reload()
            await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
            await expect(page.getByLabel('Enable Keras', { exact: true })).not.toBeChecked()
            await expect(page.locator('[data-node-id]')).toHaveCount(3)
            await mkdir(evidenceDir, { recursive: true })
            await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-builder-ready.png`), fullPage: true })

            await page.goto(`/models/${fixture.id}/validation`)
            await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
            await expect(page.getByText(/native.*pytorch/i).first()).toBeVisible({ timeout: 75_000 })
            const submitResponsePromise = page.waitForResponse(
              (response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST',
            )
            await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
            const submitResponse = await submitResponsePromise
            expect(submitResponse.ok()).toBeTruthy()
            const submitted = await submitResponse.json() as { id: string }
            await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 90_000 })

            let finalJob: RuntimeJob = { state: 'missing' }
            await expect.poll(async () => {
              const jobResponse = await request.get(`/api/execution/jobs/${submitted.id}`)
              if (!jobResponse.ok()) return 'missing'
              finalJob = await jobResponse.json() as RuntimeJob
              return finalJob.state
            }, { timeout: 90_000 }).toBe('succeeded')

            expect(finalJob.worker).toMatchObject({ api: 'native', backend: 'pytorch', device_id: 'cpu' })
            expect(finalJob.result?.status).toBe('succeeded')
            expect(finalJob.result?.outputs[0]?.shape).toEqual(candidate.runtimeOutputShape)
            expect(finalJob.result?.parameter_count).toBe(candidate.runtimeParameterCount)
            expect(browserErrors, `${candidate.name} should not emit browser errors`).toEqual([])
            await page.screenshot({ path: testInfo.outputPath(`${candidate.name}-runtime-succeeded.png`), fullPage: true })
            await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-runtime-succeeded.png`), fullPage: true })
            await writeFile(path.join(evidenceDir, `${candidate.name}.json`), JSON.stringify({
              case: candidate.name,
              project_id: fixture.id,
              definition_id: candidate.definitionId,
              profile: { execution_backend: 'pytorch', keras_enabled: false, worker: finalJob.worker },
              revision: saved.current_revision,
              expected: {
                static_output_shape: candidate.staticOutputShape,
                runtime_output_shape: candidate.runtimeOutputShape,
                static_parameter_count: candidate.staticParameterCount,
                runtime_parameter_count: candidate.runtimeParameterCount,
              },
              validation: validationPayload,
              submitted_job_id: submitted.id,
              final_job: finalJob,
              browser_errors: browserErrors,
            }, null, 2))
            await writeFile(path.join(evidenceDir, `${candidate.name}-browser-errors.json`), JSON.stringify(browserErrors, null, 2))
          } finally {
            await deleteProject(request, fixture.id)
          }
        })
      }
    })
  })
  test.describe('structural merge slice.spec', () => {
    type CreatedProject = { id: string; current_revision: number; name: string }
    type GraphNode = {
      id: string
      definition_id: string
      definition_version: number
      label: string
      position: { x: number; y: number }
      parameters: Record<string, unknown>
    }
    type GraphEdge = {
      id: string
      source_node_id: string
      source_port_id: string
      target_node_id: string
      target_port_id: string
    }
    type ShapeSummary = {
      node_id: string
      parameter_count: number
      outputs: Record<string, { shape: Array<number | null> }>
    }
    type ValidationDiagnostic = { code?: string; message?: string; node_id?: string | null }
    type ValidationPayload = {
      valid: boolean
      diagnostics: ValidationDiagnostic[]
      shape_summaries: ShapeSummary[]
      total_parameters: number
    }
    type RuntimeJob = {
      state: string
      worker?: { api: string; backend: string; device_id: string }
      result?: { status: string; outputs: Array<{ shape: number[] }>; parameter_count: number }
    }
    type SavedProject = {
      current_revision: number
      validation_status: string
      graph: { nodes: GraphNode[]; edges: GraphEdge[] }
    }
    type StructuralCase = {
      name: string
      displayName: string
      definitionId: string
      kerasEnabled: boolean
      inputShapes: number[][]
      expectedStaticShape: Array<number | null>
      expectedRuntimeShape: number[]
      outputPortId?: string
      inputTargetLabel?: 'Input' | 'Inputs'
      parameterOverrides?: Record<string, unknown>
      worker: { api: string; backend: string; device_id: string }
    }

    const evidenceDir = path.resolve(
      process.env.NW_VALIDATION_EVIDENCE_DIR
        ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260929', 'evidence', 'nw-t3-09-structural-merge'),
    )

    function graphShell(nodes: GraphNode[], edges: GraphEdge[] = []) {
      return {
        schema_version: 6,
        nodes,
        edges,
        model_configuration: {},
        export_settings: {
          include_comments: true,
          include_shape_assertions: true,
          preferred_binary_format: 'keras',
        },
        future_training_configuration: {},
      }
    }

    function node(
      id: string,
      definitionId: string,
      label: string,
      position: { x: number; y: number },
      parameters: Record<string, unknown>,
    ): GraphNode {
      return {
        id,
        definition_id: definitionId,
        definition_version: 1,
        label,
        position,
        parameters,
      }
    }

    function graphForCase(candidate: StructuralCase) {
      const nodes = candidate.inputShapes.map((shape, index) => node(
        `input-${index + 1}`,
        'input.tensor',
        `Input ${index + 1}`,
        { x: 48, y: 80 + index * 160 },
        { shape, dtype: 'float32' },
      ))
      nodes.push(node('merged-output', 'output.custom', 'Merged output', { x: 1040, y: 240 }, {}))
      return graphShell(nodes)
    }

    function invalidGraph(definitionId: string, inputShapes: number[][], parameters: Record<string, unknown> = {}) {
      const nodes = inputShapes.map((shape, index) => node(
        `input-${index + 1}`,
        'input.tensor',
        `Input ${index + 1}`,
        { x: 48, y: 80 + index * 160 },
        { shape, dtype: 'float32' },
      ))
      nodes.push(node('merge', definitionId, 'Merge operation', { x: 560, y: 240 }, parameters))
      nodes.push(node('merged-output', 'output.custom', 'Merged output', { x: 1040, y: 240 }, {}))
      const edges: GraphEdge[] = inputShapes.map((_, index) => ({
        id: `input-${index + 1}-merge`,
        source_node_id: `input-${index + 1}`,
        source_port_id: 'output',
        target_node_id: 'merge',
        target_port_id: inputShapes.length === 1 ? 'input' : 'inputs',
      }))
      edges.push({
        id: 'merge-output',
        source_node_id: 'merge',
        source_port_id: parameters.sections ? 'output_0' : 'output',
        target_node_id: 'merged-output',
        target_port_id: 'input',
      })
      return graphShell(nodes, edges)
    }

    async function createProject(
      request: APIRequestContext,
      name: string,
      graph: ReturnType<typeof graphShell>,
      kerasEnabled: boolean,
    ): Promise<CreatedProject> {
      const response = await request.post('/api/projects', {
        data: {
          name,
          description: 'NW-T3-09 structural merge validation fixture',
          execution_backend: 'pytorch',
          keras_enabled: kerasEnabled,
          graph,
        },
      })
      expect(response.ok()).toBeTruthy()
      return response.json() as Promise<CreatedProject>
    }

    async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
      const response = await request.delete(`/api/projects/${id}`)
      expect([204, 404]).toContain(response.status())
    }

    async function addCatalogNode(page: Page, displayName: string, definitionId: string): Promise<string> {
      const nodes = page.locator('[data-node-id]')
      const before = new Set(
        await nodes.evaluateAll((elements) => elements
          .map((element) => element.getAttribute('data-node-id'))
          .filter(Boolean) as string[]),
      )
      await page.getByRole('textbox', { name: 'Search nodes' }).fill(definitionId)
      const item = page.locator('.catalog-item').filter({ hasText: definitionId }).first()
      await expect(item, `${definitionId} should be available in the live catalog`).toBeEnabled()
      await item.click()
      await expect(nodes).toHaveCount(before.size + 1)
      const after = await nodes.evaluateAll((elements) => elements
        .map((element) => element.getAttribute('data-node-id'))
        .filter(Boolean) as string[])
      const created = after.find((nodeId) => !before.has(nodeId))
      expect(created, `Builder should materialize ${definitionId}`).toBeTruthy()
      await page.getByRole('textbox', { name: 'Search nodes' }).fill('')
      return created!
    }

    async function configureParameters(page: Page, candidate: StructuralCase): Promise<void> {
      const sections = candidate.parameterOverrides?.sections
      if (sections === undefined) return
      const field = page.locator('label.inspector-field').filter({ hasText: /^Sections/ }).locator('input').first()
      await expect(field).toBeVisible()
      await field.fill(String(sections))
      await field.press('Tab')
      await expect(field).toHaveValue(String(sections))
      await expect(page.getByRole('button', { name: /Output 2 \(output\)/ })).toBeVisible()
    }

    async function connectByPorts(
      page: Page,
      sourceLabel: string,
      targetLabel: string,
      targetPort: string,
      expectedConnectionCount: number,
      sourcePort = 'Output',
      sourcePortId = 'output',
    ): Promise<void> {
      const sourceNode = page.getByRole('group', { name: `Node ${sourceLabel}`, exact: true })
      const targetNode = page.getByRole('group', { name: `Node ${targetLabel}`, exact: true })
      const source = sourceNode.getByRole('button', { name: new RegExp(`^Port ${sourcePort} \\(${sourcePortId}\\)`) })
      const target = targetNode.getByRole('button', { name: new RegExp(`^Port ${targetPort} \\(input\\)`) })
      await expect(source).toBeVisible()
      await expect(target).toBeVisible()
      await source.press('Enter')
      await expect(source).toHaveAttribute('aria-pressed', 'true')
      const responsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation/connection') && response.request().method() === 'POST',
      )
      await target.press('Enter')
      const response = await responsePromise
      expect(response.ok()).toBeTruthy()
      const payload = await response.json() as { valid: boolean }
      expect(payload.valid, `${sourceLabel} → ${targetLabel}.${targetPort} should be accepted: ${JSON.stringify(payload)}`).toBe(true)
      await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
    }

    async function validateSaveAndRun(
      page: Page,
      request: APIRequestContext,
      fixture: CreatedProject,
      candidate: StructuralCase,
      mergeNodeId: string,
      browserErrors: string[],
      failedRequests: string[],
      testInfo: TestInfo,
    ): Promise<void> {
      const targetPort = candidate.inputTargetLabel ?? 'Inputs'
      for (let index = 0; index < candidate.inputShapes.length; index += 1) {
        await connectByPorts(page, `Input ${index + 1}`, candidate.displayName, targetPort, index + 1)
      }
      const outputPortId = candidate.outputPortId ?? 'output'
      const outputPort = outputPortId === 'output' ? 'Output' : `Output ${outputPortId.replace('output_', '')}`
      await connectByPorts(page, candidate.displayName, 'Merged output', 'Input', candidate.inputShapes.length + 1, outputPort, 'output')

      const validationResponsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Validate', exact: true }).click()
      const validationResponse = await validationResponsePromise
      expect(validationResponse.ok()).toBeTruthy()
      const validation = await validationResponse.json() as ValidationPayload
      expect(validation.valid).toBe(true)
      expect(validation.diagnostics).toEqual([])
      const summary = validation.shape_summaries.find((item) => item.node_id === mergeNodeId)
      expect(summary?.outputs[outputPortId]?.shape).toEqual(candidate.expectedStaticShape)
      expect(summary?.parameter_count).toBe(0)
      expect(validation.total_parameters).toBe(0)
      await expect(page.locator('.validation-summary')).toContainText('Ready')
      await page.screenshot({ path: testInfo.outputPath(`${candidate.name}-builder-ready.png`), fullPage: true })

      const saveResponsePromise = page.waitForResponse(
        (response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT',
      )
      await page.getByRole('button', { name: 'Save model', exact: true }).click()
      const saveResponse = await saveResponsePromise
      expect(saveResponse.ok()).toBeTruthy()
      const saved = await saveResponse.json() as SavedProject
      expect(saved.validation_status).toBe('valid')
      expect(saved.graph.nodes).toHaveLength(candidate.inputShapes.length + 2)
      expect(saved.graph.edges).toHaveLength(candidate.inputShapes.length + 1)
      const savedMerge = saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)
      expect(savedMerge).toBeTruthy()
      if (candidate.parameterOverrides) expect(savedMerge?.parameters).toMatchObject(candidate.parameterOverrides)

      await page.reload()
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
      await expect(page.locator('[data-node-id]')).toHaveCount(candidate.inputShapes.length + 2)
      await expect(page.getByRole('group', { name: `Node ${candidate.displayName}`, exact: true })).toBeVisible()
      if (candidate.outputPortId) {
        await expect(page.getByRole('button', { name: /^Port Output 0 \(output\)/ })).toBeVisible()
      }
      await mkdir(evidenceDir, { recursive: true })
      await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-builder-ready.png`), fullPage: true })

      await page.goto(`/models/${fixture.id}/validation`)
      await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
      await expect(page.getByText(candidate.kerasEnabled ? /keras.*pytorch/i : /native.*pytorch/i).first()).toBeVisible({ timeout: 75_000 })
      const submitResponsePromise = page.waitForResponse(
        (response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
      const submitResponse = await submitResponsePromise
      expect(submitResponse.ok()).toBeTruthy()
      const submitted = await submitResponse.json() as { id: string }
      await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 90_000 })

      let finalJob: RuntimeJob = { state: 'missing' }
      await expect.poll(async () => {
        const jobResponse = await request.get(`/api/execution/jobs/${submitted.id}`)
        if (!jobResponse.ok()) return 'missing'
        finalJob = await jobResponse.json() as RuntimeJob
        return finalJob.state
      }, { timeout: 90_000 }).toBe('succeeded')
      expect(finalJob.worker).toMatchObject(candidate.worker)
      expect(finalJob.result?.status).toBe('succeeded')
      expect(finalJob.result?.outputs[0]?.shape).toEqual(candidate.expectedRuntimeShape)
      expect(finalJob.result?.parameter_count).toBe(0)
      expect(browserErrors, `${candidate.name} should not emit browser errors`).toEqual([])
      expect(failedRequests, `${candidate.name} should not emit failed product requests`).toEqual([])
      await page.screenshot({ path: testInfo.outputPath(`${candidate.name}-runtime-succeeded.png`), fullPage: true })
      await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}-runtime-succeeded.png`), fullPage: true })
      await writeFile(path.join(evidenceDir, `${candidate.name}.json`), JSON.stringify({
        case: candidate.name,
        project_id: fixture.id,
        definition_id: candidate.definitionId,
        profile: { execution_backend: 'pytorch', keras_enabled: candidate.kerasEnabled, worker: finalJob.worker },
        saved_revision: saved.current_revision,
        graph: { node_count: saved.graph.nodes.length, edge_count: saved.graph.edges.length },
        expected: {
          static_output_shape: candidate.expectedStaticShape,
          runtime_output_shape: candidate.expectedRuntimeShape,
          parameter_count: 0,
          output_port_id: outputPortId,
        },
        validation,
        submitted_job_id: submitted.id,
        final_job: finalJob,
        browser_errors: browserErrors,
        failed_requests: failedRequests,
        cleanup: 'project deleted in finally block',
      }, null, 2))
      await writeFile(path.join(evidenceDir, `${candidate.name}-browser-errors.json`), JSON.stringify({ browser_errors: browserErrors, failed_requests: failedRequests }, null, 2))
    }

    const structuralCases: StructuralCase[] = [
      { name: 'nw-t3-09-concatenate-keras-pytorch', displayName: 'Concatenate', definitionId: 'merge.concatenate', kerasEnabled: true, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 6], expectedRuntimeShape: [1, 2, 6], worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-09-concatenate-native-pytorch', displayName: 'Concatenate', definitionId: 'merge.concatenate', kerasEnabled: false, inputShapes: [[2, 3], [2, 3]], expectedStaticShape: [null, 2, 6], expectedRuntimeShape: [1, 2, 6], worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-09-stack-keras-pytorch', displayName: 'Stack', definitionId: 'merge.stack', kerasEnabled: true, inputShapes: [[4], [4]], expectedStaticShape: [2, null, 4], expectedRuntimeShape: [2, 1, 4], worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-09-stack-native-pytorch', displayName: 'Stack', definitionId: 'merge.stack', kerasEnabled: false, inputShapes: [[4], [4]], expectedStaticShape: [2, null, 4], expectedRuntimeShape: [2, 1, 4], worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-09-split-keras-pytorch', displayName: 'Split', definitionId: 'merge.split', kerasEnabled: true, inputShapes: [[6]], expectedStaticShape: [null, 2], expectedRuntimeShape: [1, 2], outputPortId: 'output_0', inputTargetLabel: 'Input', parameterOverrides: { sections: 3 }, worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-09-split-native-pytorch', displayName: 'Split', definitionId: 'merge.split', kerasEnabled: false, inputShapes: [[6]], expectedStaticShape: [null, 2], expectedRuntimeShape: [1, 2], outputPortId: 'output_0', inputTargetLabel: 'Input', parameterOverrides: { sections: 3 }, worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-09-dot-keras-pytorch', displayName: 'Dot Product', definitionId: 'merge.dot', kerasEnabled: true, inputShapes: [[2, 3], [3, 4]], expectedStaticShape: [null, 2, 4], expectedRuntimeShape: [1, 2, 4], worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-09-matmul-keras-pytorch', displayName: 'Matrix Multiplication', definitionId: 'merge.matmul', kerasEnabled: true, inputShapes: [[2, 3], [3, 4]], expectedStaticShape: [null, 2, 4], expectedRuntimeShape: [1, 2, 4], worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-09-matmul-native-pytorch', displayName: 'Matrix Multiplication', definitionId: 'merge.matmul', kerasEnabled: false, inputShapes: [[2, 3], [3, 4]], expectedStaticShape: [null, 2, 4], expectedRuntimeShape: [1, 2, 4], worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-09-residual-keras-pytorch', displayName: 'Residual Connection', definitionId: 'merge.residual', kerasEnabled: true, inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-09-residual-native-pytorch', displayName: 'Residual Connection', definitionId: 'merge.residual', kerasEnabled: false, inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-09-skip-keras-pytorch', displayName: 'Skip Connection', definitionId: 'merge.skip', kerasEnabled: true, inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' } },
      { name: 'nw-t3-09-skip-native-pytorch', displayName: 'Skip Connection', definitionId: 'merge.skip', kerasEnabled: false, inputShapes: [[4], [4]], expectedStaticShape: [null, 4], expectedRuntimeShape: [1, 4], worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' } },
    ]

    test.describe('NW-T3-09 structural merge catalog slice', () => {
      test.describe.configure({ mode: 'serial' })
      test.setTimeout(240_000)

      test('live catalog renders structural merge definitions and verified mappings', async ({ page, request }) => {
        const response = await request.get('/api/nodes/catalog')
        expect(response.ok()).toBeTruthy()
        const catalog = await response.json() as {
          schema_version: number
          nodes: Array<{
            id: string
            display_name: string
            category: string
            operation?: { family?: string; shape_behavior?: string; arity?: number; variadic?: boolean } | null
            implementations: Record<string, { supported?: boolean; adapter?: string | null; symbol?: string | null }>
          }>
        }
        const merge = catalog.nodes.filter((item) => item.category === 'merge').sort((left, right) => left.id.localeCompare(right.id))
        const expectedIds = [
          'merge.add', 'merge.average', 'merge.concatenate', 'merge.divide', 'merge.dot', 'merge.matmul',
          'merge.maximum', 'merge.minimum', 'merge.multiply', 'merge.residual', 'merge.skip', 'merge.split',
          'merge.stack', 'merge.subtract',
        ]
        expect(merge.map((item) => item.id)).toEqual(expectedIds)
        const expectedContracts: Record<string, { family: string; shape_behavior: string; arity: number; variadic: boolean }> = {
          'merge.concatenate': { family: 'tensor', shape_behavior: 'concatenate', arity: 1, variadic: true },
          'merge.dot': { family: 'binary', shape_behavior: 'matmul', arity: 2, variadic: false },
          'merge.matmul': { family: 'binary', shape_behavior: 'matmul', arity: 2, variadic: false },
          'merge.stack': { family: 'tensor', shape_behavior: 'stack', arity: 1, variadic: true },
        }
        for (const [id, contract] of Object.entries(expectedContracts)) {
          expect(merge.find((item) => item.id === id)?.operation).toMatchObject(contract)
        }
        for (const id of ['merge.residual', 'merge.skip', 'merge.split']) {
          const item = merge.find((entry) => entry.id === id)!
          expect(item.operation).toBeNull()
          expect(item.implementations.keras?.supported).toBe(true)
          expect(item.implementations.pytorch?.supported).toBe(true)
        }
        expect(merge.find((item) => item.id === 'merge.residual')?.implementations).toMatchObject({ keras: { adapter: 'keras_merge' }, pytorch: { adapter: 'add' } })
        expect(merge.find((item) => item.id === 'merge.skip')?.implementations).toMatchObject({ keras: { adapter: 'keras_merge' }, pytorch: { adapter: 'add' } })
        expect(merge.find((item) => item.id === 'merge.split')?.implementations).toMatchObject({ keras: { adapter: 'split' }, pytorch: { adapter: 'split' } })
        for (const item of merge) {
          expect(item.implementations.keras?.supported).toBe(true)
          expect(item.implementations.pytorch?.supported).toBe(true)
        }

        await page.goto('/builder')
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        await page.getByRole('button', { name: /Merge 14 nodes/ }).click()
        for (const item of merge) {
          await expect(page.getByRole('button', { name: new RegExp(`${item.display_name} ${item.id} ·`) })).toBeEnabled()
        }
        await mkdir(evidenceDir, { recursive: true })
        await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-09-structural-merge-catalog-rendered.png'), fullPage: true })
        await writeFile(path.join(evidenceDir, 'nw-t3-09-live-catalog.json'), JSON.stringify({
          schema_version: catalog.schema_version,
          catalog_count: catalog.nodes.length,
          catalog_fingerprint_sha256: createHash('sha256').update(JSON.stringify(catalog)).digest('hex'),
          merge,
          structural_ids: Object.keys(expectedContracts).concat(['merge.residual', 'merge.skip', 'merge.split']),
        }, null, 2))
      })

      for (const candidate of structuralCases) {
        test(`${candidate.name} builds, persists, and completes exact-profile functional validation`, async ({ page, request }, testInfo) => {
          const fixture = await createProject(
            request,
            `Tier 3 structural merge ${candidate.definitionId} ${candidate.kerasEnabled ? 'keras' : 'native'} ${Date.now()}`,
            graphForCase(candidate),
            candidate.kerasEnabled,
          )
          const browserErrors: string[] = []
          const failedRequests: string[] = []
          page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
          page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
          page.on('requestfailed', (event) => failedRequests.push(`${event.method()} ${event.url()} ${event.failure()?.errorText ?? 'unknown'}`))
          page.on('response', (event) => {
            if (event.status() >= 400 && event.url().includes('/api/')) failedRequests.push(`${event.status()} ${event.request().method()} ${event.url()}`)
          })
          try {
            await page.goto(`/builder/${fixture.id}`)
            await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
            if (!candidate.kerasEnabled) await expect(page.getByLabel('Enable Keras', { exact: true })).not.toBeChecked()
            const mergeNodeId = await addCatalogNode(page, candidate.displayName, candidate.definitionId)
            await configureParameters(page, candidate)
            await validateSaveAndRun(page, request, fixture, candidate, mergeNodeId, browserErrors, failedRequests, testInfo)
          } finally {
            await deleteProject(request, fixture.id)
          }
        })
      }

      test('matmul and split shape contracts remain fail-closed for invalid graphs', async ({ page, request }, testInfo) => {
        const cases = [
          {
            name: 'nw-t3-09-invalid-matmul',
            definitionId: 'merge.matmul',
            graph: invalidGraph('merge.matmul', [[2, 3], [5, 4]]),
            message: /inner|compatible|matmul|shape/i,
          },
          {
            name: 'nw-t3-09-invalid-split',
            definitionId: 'merge.split',
            graph: invalidGraph('merge.split', [[5]], { sections: 2, axis: -1 }),
            message: /split|divis|shape/i,
          },
        ]
        for (const candidate of cases) {
          const fixture = await createProject(request, `Tier 3 invalid structural merge ${candidate.name} ${Date.now()}`, candidate.graph, true)
          const browserErrors: string[] = []
          const failedRequests: string[] = []
          page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`))
          page.on('console', (message) => { if (message.type() === 'error') browserErrors.push(`console: ${message.text()}`) })
          page.on('requestfailed', (event) => failedRequests.push(`${event.method()} ${event.url()} ${event.failure()?.errorText ?? 'unknown'}`))
          try {
            await page.goto(`/builder/${fixture.id}`)
            await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
            const validationResponsePromise = page.waitForResponse(
              (response) => response.url().endsWith('/api/validation') && response.request().method() === 'POST',
            )
            await page.getByRole('button', { name: 'Validate', exact: true }).click()
            const validationResponse = await validationResponsePromise
            expect(validationResponse.ok()).toBeTruthy()
            const validation = await validationResponse.json() as ValidationPayload
            expect(validation.valid).toBe(false)
            expect(validation.diagnostics).toEqual(expect.arrayContaining([
              expect.objectContaining({ message: expect.stringMatching(candidate.message) }),
            ]))
            await expect(page.locator('.validation-summary')).toContainText(/Blocked/i)
            await mkdir(evidenceDir, { recursive: true })
            await page.screenshot({ path: testInfo.outputPath(`${candidate.name}.png`), fullPage: true })
            await page.screenshot({ path: path.join(evidenceDir, `${candidate.name}.png`), fullPage: true })
            await writeFile(path.join(evidenceDir, `${candidate.name}.json`), JSON.stringify({
              case: candidate.name,
              project_id: fixture.id,
              definition_id: candidate.definitionId,
              validation,
              browser_errors: browserErrors,
              failed_requests: failedRequests,
              execution_job_submissions: 0,
            }, null, 2))
            expect(browserErrors).toEqual([])
            expect(failedRequests).toEqual([])
          } finally {
            await deleteProject(request, fixture.id)
          }
        }
      })
    })
  })
  test.describe('transformer blocks slice.spec', () => {
    type CreatedProject = { id: string; current_revision: number; name: string }
    type GraphNode = {
      id: string
      definition_id: string
      definition_version: number
      label: string
      position: { x: number; y: number }
      parameters: Record<string, unknown>
    }
    type ValidationDiagnostic = {
      code?: string
      message?: string
      node_id?: string | null
      parameter_id?: string | null
    }
    type ShapeSummary = {
      node_id: string
      parameter_count: number
      outputs: Record<string, { shape: Array<number | null> }>
    }
    type ValidationPayload = {
      valid: boolean
      diagnostics: ValidationDiagnostic[]
      shape_summaries: ShapeSummary[]
      total_parameters: number
    }
    type RuntimeJob = {
      state: string
      worker?: { api: string; backend: string; device_id: string }
      result?: {
        status: string
        outputs: Array<{ shape: number[] }>
        parameter_count: number
      }
    }
    type SavedProject = {
      current_revision: number
      validation_status: string
      graph: { nodes: GraphNode[]; edges: unknown[] }
    }
    type TransformerKind = 'single' | 'decoder' | 'residual' | 'mask'
    type TransformerCase = {
      name: string
      catalogNode: string
      definitionId: string
      kind: TransformerKind
      parameters: Record<string, unknown>
      staticOutputShape: Array<number | null>
      runtimeOutputShape: number[]
      parameterCount: number
      nodeCount: number
      edgeCount: number
      kerasEnabled: boolean
      worker: { api: string; backend: string; device_id: string }
    }

    const evidenceDir = path.resolve(
      process.env.NW_VALIDATION_EVIDENCE_DIR
        ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260929', 'evidence', 'nw-t3-07-transformer'),
    )

    function graphShell(nodes: GraphNode[]) {
      return {
        schema_version: 6,
        nodes,
        edges: [],
        model_configuration: {},
        export_settings: {
          include_comments: true,
          include_shape_assertions: true,
          preferred_binary_format: 'keras',
        },
        future_training_configuration: {},
      }
    }

    function node(
      id: string,
      definitionId: string,
      label: string,
      position: { x: number; y: number },
      parameters: Record<string, unknown>,
    ): GraphNode {
      return {
        id,
        definition_id: definitionId,
        definition_version: 1,
        label,
        position,
        parameters,
      }
    }

    function initialGraph(kind: TransformerKind) {
      const nodes = kind === 'decoder'
        ? [
            node('query-input', 'input.sequence', 'Query Sequence', { x: 48, y: 120 }, { shape: [5, 8], dtype: 'float32' }),
            node('context-input', 'input.sequence', 'Context Sequence', { x: 48, y: 360 }, { shape: [3, 8], dtype: 'float32' }),
            node('custom-output', 'output.custom', 'Custom Output', { x: 1120, y: 240 }, {}),
          ]
        : kind === 'residual'
          ? [
              node('left-input', 'input.sequence', 'Left Sequence', { x: 48, y: 120 }, { shape: [5, 8], dtype: 'float32' }),
              node('right-input', 'input.sequence', 'Right Sequence', { x: 48, y: 360 }, { shape: [5, 8], dtype: 'float32' }),
              node('custom-output', 'output.custom', 'Custom Output', { x: 1120, y: 240 }, {}),
            ]
          : kind === 'mask'
            ? [
                node('mask-input', 'input.mask', 'Mask Input', { x: 48, y: 240 }, { shape: [5, 5], dtype: 'bool' }),
                node('custom-output', 'output.custom', 'Custom Output', { x: 1120, y: 240 }, {}),
              ]
            : [
                node('sequence-input', 'input.sequence', 'Input Sequence', { x: 48, y: 240 }, { shape: [5, 8], dtype: 'float32' }),
                node('custom-output', 'output.custom', 'Custom Output', { x: 1120, y: 240 }, {}),
              ]
      return graphShell(nodes)
    }

    async function createProject(
      request: APIRequestContext,
      name: string,
      kind: TransformerKind,
      kerasEnabled: boolean,
    ): Promise<CreatedProject> {
      const response = await request.post('/api/projects', {
        data: {
          name,
          description: 'NW-T3-07 Transformer block validation fixture',
          execution_backend: 'pytorch',
          keras_enabled: kerasEnabled,
          graph: initialGraph(kind),
        },
      })
      expect(response.ok()).toBeTruthy()
      return response.json() as Promise<CreatedProject>
    }

    async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
      const response = await request.delete('/api/projects/' + id)
      expect([204, 404]).toContain(response.status())
    }

    async function addCatalogNode(page: Page, displayName: string, definitionId: string): Promise<string> {
      const nodes = page.locator('[data-node-id]')
      const before = new Set(
        await nodes.evaluateAll((elements) => elements
          .map((element) => element.getAttribute('data-node-id'))
          .filter(Boolean) as string[]),
      )
      await page.getByRole('textbox', { name: 'Search nodes' }).fill(displayName)
      const item = page.locator('.catalog-item').filter({ hasText: definitionId }).first()
      await expect(item, 'Catalog node ' + displayName + ' should be available').toBeEnabled()
      await item.click()
      await expect(nodes).toHaveCount(before.size + 1)
      const after = await nodes.evaluateAll((elements) => elements
        .map((element) => element.getAttribute('data-node-id'))
        .filter(Boolean) as string[])
      const created = after.find((nodeId) => !before.has(nodeId))
      expect(created, 'Builder should materialize ' + displayName).toBeTruthy()
      await page.getByRole('textbox', { name: 'Search nodes' }).fill('')
      return created!
    }

    async function connectByPorts(
      page: Page,
      sourceLabel: string,
      targetLabel: string,
      targetPortLabel: string,
      expectedConnectionCount: number,
    ): Promise<void> {
      const sourceNode = page.getByRole('group', { name: 'Node ' + sourceLabel, exact: true })
      const targetNode = page.getByRole('group', { name: 'Node ' + targetLabel, exact: true })
      const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
      const target = targetNode.getByRole('button', { name: new RegExp('^Port ' + targetPortLabel + ' \\(input\\)') })
      await expect(source).toBeVisible()
      await expect(target).toBeVisible()
      await source.press('Enter')
      await expect(source).toHaveAttribute('aria-pressed', 'true')
      const responsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation/connection') && response.request().method() === 'POST',
      )
      await target.press('Enter')
      const response = await responsePromise
      expect(response.ok()).toBeTruthy()
      const payload = await response.json() as { valid: boolean }
      expect(payload.valid, sourceLabel + ' → ' + targetLabel + '.' + targetPortLabel + ' should be accepted').toBe(true)
      await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(expectedConnectionCount)
    }

    async function rejectConnectionByPorts(
      page: Page,
      sourceLabel: string,
      targetLabel: string,
      targetPortLabel: string,
    ): Promise<{ valid: boolean; diagnostics?: unknown[] }> {
      const sourceNode = page.getByRole('group', { name: 'Node ' + sourceLabel, exact: true })
      const targetNode = page.getByRole('group', { name: 'Node ' + targetLabel, exact: true })
      const source = sourceNode.getByRole('button', { name: /^Port Output \(output\)/ })
      const target = targetNode.getByRole('button', { name: new RegExp('^Port ' + targetPortLabel + ' \\(input\\)') })
      await expect(source).toBeVisible()
      await expect(target).toBeVisible()
      await source.press('Enter')
      await expect(source).toHaveAttribute('aria-pressed', 'true')
      const responsePromise = page.waitForResponse(
        (response) => response.url().endsWith('/api/validation/connection') && response.request().method() === 'POST',
      )
      await target.press('Enter')
      const response = await responsePromise
      expect(response.ok()).toBeTruthy()
      const payload = await response.json() as { valid: boolean; diagnostics?: unknown[] }
      expect(payload.valid, sourceLabel + ' → ' + targetLabel + '.' + targetPortLabel + ' should be rejected').toBe(false)
      return payload
    }

    async function parameterField(inspector: Locator, displayName: string | RegExp): Promise<Locator> {
      const field = inspector.locator('label.inspector-field').filter({ hasText: displayName }).first()
      if (!(await field.isVisible())) {
        const group = inspector.locator('details.parameter-group').filter({ hasText: displayName }).first()
        await group.locator('summary').click()
      }
      await expect(field, displayName.toString() + ' should be visible in the Inspector').toBeVisible()
      return field
    }

    async function setTransformerParameters(
      page: Page,
      nodeId: string,
      parameters: Record<string, unknown>,
    ): Promise<void> {
      await page.locator('[data-node-id="' + nodeId + '"]').click()
      const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
      const displayNames: Record<string, string | RegExp> = {
        num_heads: /Num heads/i,
        key_dim: /Key dimension/i,
        dropout: /Dropout/i,
        layers: /Layers/i,
        ffn_multiplier: /Feed-forward multiplier/i,
        pre_norm: /Pre-normalization/i,
      }
      for (const [id, value] of Object.entries(parameters)) {
        const displayName = displayNames[id]
        if (!displayName) continue
        const field = await parameterField(inspector, displayName)
        if (typeof value === 'boolean') {
          await field.locator('input[type="checkbox"]').setChecked(value)
        } else {
          const input = field.locator('input').first()
          if (await input.isEnabled()) await input.fill(String(value))
          await expect(input).toHaveValue(String(value))
        }
      }
    }

    async function validateGraph(page: Page, nodeId: string, edgeCount: number): Promise<ValidationPayload> {
      const validationResponsePromise = page.waitForResponse(
        (response) => {
          if (!response.url().endsWith('/api/validation') || response.request().method() !== 'POST') return false
          const body = response.request().postDataJSON() as {
            graph?: { nodes?: Array<{ id?: string }>; edges?: unknown[] }
          }
          return body.graph?.nodes?.some((node) => node.id === nodeId) === true
            && body.graph?.edges?.length === edgeCount
        },
      )
      await page.getByRole('button', { name: 'Validate', exact: true }).click()
      const validationResponse = await validationResponsePromise
      expect(validationResponse.ok()).toBeTruthy()
      return validationResponse.json() as Promise<ValidationPayload>
    }

    async function connectCase(page: Page, candidate: TransformerCase): Promise<void> {
      if (candidate.kind === 'decoder') {
        await connectByPorts(page, 'Query Sequence', candidate.catalogNode, 'Query', 1)
        await connectByPorts(page, 'Context Sequence', candidate.catalogNode, 'Context', 2)
      } else if (candidate.kind === 'residual') {
        await connectByPorts(page, 'Left Sequence', candidate.catalogNode, 'Inputs', 1)
        await connectByPorts(page, 'Right Sequence', candidate.catalogNode, 'Inputs', 2)
      } else if (candidate.kind === 'mask') {
        await connectByPorts(page, 'Mask Input', candidate.catalogNode, 'Input', 1)
      } else {
        const targetPort = candidate.definitionId === 'transformer.positional_encoding' ? 'Input' : 'Query'
        await connectByPorts(page, 'Input Sequence', candidate.catalogNode, targetPort, 1)
      }
      await connectByPorts(page, candidate.catalogNode, 'Custom Output', 'Input', candidate.edgeCount)
    }

    async function validateSaveAndRun(
      page: Page,
      request: APIRequestContext,
      fixture: CreatedProject,
      nodeId: string,
      candidate: TransformerCase,
      browserErrors: string[],
      failedRequests: string[],
    ): Promise<{ validation: ValidationPayload; saved: SavedProject; submittedJobId: string; finalJob: RuntimeJob }> {
      const validation = await validateGraph(page, nodeId, candidate.edgeCount)
      expect(validation.valid, JSON.stringify(validation.diagnostics)).toBe(true)
      expect(validation.diagnostics).toEqual([])
      const summary = validation.shape_summaries.find((item) => item.node_id === nodeId)
      expect(summary?.outputs.output?.shape).toEqual(candidate.staticOutputShape)
      expect(summary?.parameter_count).toBe(candidate.parameterCount)
      expect(validation.total_parameters).toBe(candidate.parameterCount)
      await expect(page.locator('.validation-summary')).toContainText('Ready')
      await mkdir(evidenceDir, { recursive: true })
      await page.screenshot({ path: path.join(evidenceDir, candidate.name + '-builder-ready.png'), fullPage: true })

      const saveResponsePromise = page.waitForResponse(
        (response) => response.url().includes('/api/projects/' + fixture.id) && response.request().method() === 'PUT',
      )
      await page.getByRole('button', { name: 'Save model', exact: true }).click()
      const saveResponse = await saveResponsePromise
      expect(saveResponse.ok()).toBeTruthy()
      const saved = await saveResponse.json() as SavedProject
      expect(saved.validation_status).toBe('valid')
      expect(saved.graph.nodes).toHaveLength(candidate.nodeCount)
      expect(saved.graph.edges).toHaveLength(candidate.edgeCount)
      const savedNode = saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)
      expect(savedNode?.parameters).toMatchObject(candidate.parameters)

      await page.reload()
      await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible({ timeout: 30_000 })
      await expect(page.locator('[data-node-id]')).toHaveCount(candidate.nodeCount)
      await expect(page.getByRole('group', { name: 'Node ' + candidate.catalogNode, exact: true })).toBeVisible()

      await page.goto('/models/' + fixture.id + '/validation')
      await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
      const profilePattern = candidate.kerasEnabled ? /keras.*pytorch/i : /native.*pytorch/i
      await expect(page.getByText(profilePattern).first()).toBeVisible({ timeout: 75_000 })
      await expect(page.getByRole('button', { name: 'Run functional validation', exact: true })).toBeEnabled()

      const submitResponsePromise = page.waitForResponse(
        (response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
      const submitResponse = await submitResponsePromise
      expect(submitResponse.ok()).toBeTruthy()
      const submitted = await submitResponse.json() as { id: string }
      await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 90_000 })

      let finalJob: RuntimeJob = { state: 'missing' }
      await expect.poll(async () => {
        const jobResponse = await request.get('/api/execution/jobs/' + submitted.id)
        if (!jobResponse.ok()) return 'missing'
        finalJob = await jobResponse.json() as RuntimeJob
        return finalJob.state
      }, { timeout: 90_000 }).toBe('succeeded')

      expect(finalJob.worker).toMatchObject(candidate.worker)
      expect(finalJob.result?.status).toBe('succeeded')
      expect(finalJob.result?.outputs[0]?.shape).toEqual(candidate.runtimeOutputShape)
      expect(finalJob.result?.parameter_count).toBe(candidate.parameterCount)
      await page.screenshot({ path: path.join(evidenceDir, candidate.name + '-runtime-succeeded.png'), fullPage: true })
      expect(browserErrors, candidate.name + ' should not emit browser errors').toEqual([])
      expect(failedRequests, candidate.name + ' should not emit failed product requests').toEqual([])

      await writeFile(path.join(evidenceDir, candidate.name + '.json'), JSON.stringify({
        case: candidate.name,
        project_id: fixture.id,
        definition_id: candidate.definitionId,
        authored_parameters: candidate.parameters,
        profile: {
          execution_backend: 'pytorch',
          keras_enabled: candidate.kerasEnabled,
          worker: finalJob.worker,
        },
        saved_revision: saved.current_revision,
        graph: { node_count: saved.graph.nodes.length, edge_count: saved.graph.edges.length },
        validation,
        submitted_job_id: submitted.id,
        final_job: finalJob,
        browser_errors: browserErrors,
        failed_requests: failedRequests,
        cleanup: 'project deleted in finally block',
      }, null, 2))
      await writeFile(path.join(evidenceDir, candidate.name + '-browser-errors.json'), JSON.stringify({
        browser_errors: browserErrors,
        failed_requests: failedRequests,
      }, null, 2))
      return { validation, saved, submittedJobId: submitted.id, finalJob }
    }

    const transformerCases: TransformerCase[] = [
      {
        name: 'nw-t3-07-encoder-keras-pytorch',
        catalogNode: 'Transformer Encoder Block',
        definitionId: 'transformer.encoder_block',
        kind: 'single',
        parameters: { num_heads: 2, key_dim: 4, layers: 2, dropout: 0, ffn_multiplier: 2, pre_norm: true },
        staticOutputShape: [null, 5, 8],
        runtimeOutputShape: [1, 5, 8],
        parameterCount: 1200,
        nodeCount: 3,
        edgeCount: 2,
        kerasEnabled: true,
        worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
      },
      {
        name: 'nw-t3-07-decoder-keras-pytorch',
        catalogNode: 'Transformer Decoder Block',
        definitionId: 'transformer.decoder_block',
        kind: 'decoder',
        parameters: { num_heads: 2, key_dim: 4, layers: 1, dropout: 0, ffn_multiplier: 2, pre_norm: true },
        staticOutputShape: [null, 5, 8],
        runtimeOutputShape: [1, 5, 8],
        parameterCount: 904,
        nodeCount: 4,
        edgeCount: 3,
        kerasEnabled: true,
        worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
      },
      {
        name: 'nw-t3-07-feed-forward-keras-pytorch',
        catalogNode: 'Feed-forward Transformer Block',
        definitionId: 'transformer.feed_forward',
        kind: 'single',
        parameters: { ffn_multiplier: 2, dropout: 0 },
        staticOutputShape: [null, 5, 8],
        runtimeOutputShape: [1, 5, 8],
        parameterCount: 280,
        nodeCount: 3,
        edgeCount: 2,
        kerasEnabled: true,
        worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
      },
      {
        name: 'nw-t3-07-stack-keras-pytorch',
        catalogNode: 'Complete Transformer Stack',
        definitionId: 'transformer.stack',
        kind: 'single',
        parameters: { num_heads: 2, key_dim: 4, layers: 2, dropout: 0, ffn_multiplier: 2, pre_norm: false },
        staticOutputShape: [null, 5, 8],
        runtimeOutputShape: [1, 5, 8],
        parameterCount: 1200,
        nodeCount: 3,
        edgeCount: 2,
        kerasEnabled: true,
        worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
      },
      {
        name: 'nw-t3-07-positional-encoding-keras-pytorch',
        catalogNode: 'Positional Encoding',
        definitionId: 'transformer.positional_encoding',
        kind: 'single',
        parameters: {},
        staticOutputShape: [null, 5, 8],
        runtimeOutputShape: [1, 5, 8],
        parameterCount: 0,
        nodeCount: 3,
        edgeCount: 2,
        kerasEnabled: true,
        worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
      },
      {
        name: 'nw-t3-07-residual-keras-pytorch',
        catalogNode: 'Residual Connection',
        definitionId: 'transformer.residual',
        kind: 'residual',
        parameters: {},
        staticOutputShape: [null, 5, 8],
        runtimeOutputShape: [1, 5, 8],
        parameterCount: 0,
        nodeCount: 4,
        edgeCount: 3,
        kerasEnabled: true,
        worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
      },
      {
        name: 'nw-t3-07-attention-mask-keras-pytorch',
        catalogNode: 'Attention Mask',
        definitionId: 'transformer.attention_mask',
        kind: 'mask',
        parameters: {},
        staticOutputShape: [null, 5, 5],
        runtimeOutputShape: [1, 5, 5],
        parameterCount: 0,
        nodeCount: 3,
        edgeCount: 2,
        kerasEnabled: true,
        worker: { api: 'keras', backend: 'pytorch', device_id: 'cpu' },
      },
      {
        name: 'nw-t3-07-encoder-native-pytorch',
        catalogNode: 'Transformer Encoder Block',
        definitionId: 'transformer.encoder_block',
        kind: 'single',
        parameters: { num_heads: 2, key_dim: 4, layers: 2, dropout: 0, ffn_multiplier: 2, pre_norm: true },
        staticOutputShape: [null, 5, 8],
        runtimeOutputShape: [1, 5, 8],
        parameterCount: 1200,
        nodeCount: 3,
        edgeCount: 2,
        kerasEnabled: false,
        worker: { api: 'native', backend: 'pytorch', device_id: 'cpu' },
      },
    ]

    test.describe('NW-T3-07 Transformer block catalog slice', () => {
      test.describe.configure({ mode: 'serial' })
      test.setTimeout(240_000)

      test('live catalog renders every Transformer definition with declared mappings', async ({ page, request }) => {
        const catalogResponse = await request.get('/api/nodes/catalog')
        expect(catalogResponse.ok()).toBeTruthy()
        const catalog = await catalogResponse.json() as {
          schema_version: number
          nodes: Array<{
            id: string
            display_name: string
            category: string
            implementations: Record<string, { supported?: boolean }>
          }>
        }
        const transformer = catalog.nodes
          .filter((item) => item.id.startsWith('transformer.'))
          .sort((left, right) => left.id.localeCompare(right.id))
        expect(transformer.map((item) => item.id)).toEqual([
          'transformer.attention_mask',
          'transformer.decoder_block',
          'transformer.encoder_block',
          'transformer.feed_forward',
          'transformer.positional_encoding',
          'transformer.residual',
          'transformer.stack',
        ])
        for (const item of transformer) {
          expect(item.implementations.keras?.supported).toBe(true)
          expect(item.implementations.tensorflow?.supported).toBe(true)
          expect(item.implementations.pytorch?.supported).toBe(true)
          expect(item.implementations.jax?.supported).toBe(true)
        }

        await page.goto('/builder')
        await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
        await page.getByRole('button', { name: /Transformer 10 nodes/ }).click()
        for (const candidate of transformerCases.slice(0, 7)) {
          await expect(page.getByRole('button', {
            name: new RegExp(candidate.catalogNode + ' ' + candidate.definitionId + ' · operation'),
          })).toBeEnabled()
        }
        await mkdir(evidenceDir, { recursive: true })
        await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-07-transformer-catalog-rendered.png'), fullPage: true })
        await writeFile(path.join(evidenceDir, 'nw-t3-07-live-catalog.json'), JSON.stringify({
          schema_version: catalog.schema_version,
          catalog_count: catalog.nodes.length,
          category_counts: catalog.nodes.reduce<Record<string, number>>((counts, item) => {
            counts[item.category] = (counts[item.category] ?? 0) + 1
            return counts
          }, {}),
          catalog_fingerprint_sha256: createHash('sha256').update(JSON.stringify(catalog)).digest('hex'),
          transformer,
        }, null, 2))
      })

      for (const candidate of transformerCases.filter((item) => item.kind !== 'mask')) {
        test(candidate.name + ' builds, persists, and completes exact-profile functional validation', async ({ page, request }) => {
          const fixture = await createProject(
            request,
            'Tier 3 Transformer ' + candidate.catalogNode + ' ' + Date.now(),
            candidate.kind,
            candidate.kerasEnabled,
          )
          const browserErrors: string[] = []
          const failedRequests: string[] = []
          page.on('pageerror', (error) => browserErrors.push('pageerror: ' + error.message))
          page.on('console', (message) => { if (message.type() === 'error') browserErrors.push('console: ' + message.text()) })
          page.on('requestfailed', (requestEvent) => failedRequests.push(requestEvent.method() + ' ' + requestEvent.url() + ' ' + (requestEvent.failure()?.errorText ?? 'unknown')))
          page.on('response', (response) => {
            if (response.status() >= 400 && response.url().includes('/api/')) failedRequests.push(response.status() + ' ' + response.request().method() + ' ' + response.url())
          })
          try {
            await page.goto('/builder/' + fixture.id)
            await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
            const transformerNodeId = await addCatalogNode(page, candidate.catalogNode, candidate.definitionId)
            await setTransformerParameters(page, transformerNodeId, candidate.parameters)
            await connectCase(page, candidate)
            await validateSaveAndRun(page, request, fixture, transformerNodeId, candidate, browserErrors, failedRequests)
          } finally {
            await deleteProject(request, fixture.id)
          }
        })
      }

      test('attention mask preserves its mask type and rejects a tensor-only output', async ({ page, request }) => {
        const candidate = transformerCases.find((item) => item.kind === 'mask')!
        const fixture = await createProject(
          request,
          'Tier 3 Transformer Attention Mask boundary ' + Date.now(),
          candidate.kind,
          candidate.kerasEnabled,
        )
        const browserErrors: string[] = []
        const failedRequests: string[] = []
        page.on('pageerror', (error) => browserErrors.push('pageerror: ' + error.message))
        page.on('console', (message) => { if (message.type() === 'error') browserErrors.push('console: ' + message.text()) })
        page.on('requestfailed', (requestEvent) => failedRequests.push(requestEvent.method() + ' ' + requestEvent.url() + ' ' + (requestEvent.failure()?.errorText ?? 'unknown')))
        page.on('response', (response) => {
          if (response.status() >= 400 && response.url().includes('/api/')) failedRequests.push(response.status() + ' ' + response.request().method() + ' ' + response.url())
        })
        try {
          await page.goto('/builder/' + fixture.id)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
          await addCatalogNode(page, candidate.catalogNode, candidate.definitionId)
          await connectByPorts(page, 'Mask Input', candidate.catalogNode, 'Input', 1)
          const rejected = await rejectConnectionByPorts(page, candidate.catalogNode, 'Custom Output', 'Input')
          await mkdir(evidenceDir, { recursive: true })
          await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-07-attention-mask-type-boundary.png'), fullPage: true })
          await writeFile(path.join(evidenceDir, 'nw-t3-07-attention-mask-type-boundary.json'), JSON.stringify({
            case: 'nw-t3-07-attention-mask-type-boundary',
            project_id: fixture.id,
            definition_id: candidate.definitionId,
            input_connection: 'Mask Input → Attention Mask.Input accepted',
            rejected_connection: 'Attention Mask → Custom Output.Input',
            validation: rejected,
            execution_job_submissions: 0,
            browser_errors: browserErrors,
            failed_requests: failedRequests,
            limitation: 'The current catalog has no mask-capable terminal consumer, so standalone attention-mask runtime execution is not certified in this slice.',
          }, null, 2))
          expect(browserErrors).toEqual([])
          expect(failedRequests).toEqual([])
        } finally {
          await deleteProject(request, fixture.id)
        }
      })

      test('encoder invalid layer count is blocked by the Builder, repaired, persisted, and executed', async ({ page, request }) => {
        const candidate = transformerCases[0]!
        const repaired: TransformerCase = {
          ...candidate,
          name: 'nw-t3-07-encoder-repaired',
          parameters: { ...candidate.parameters, layers: 1 },
          parameterCount: 600,
        }
        const fixture = await createProject(request, 'Tier 3 Transformer repair ' + Date.now(), candidate.kind, true)
        const browserErrors: string[] = []
        const failedRequests: string[] = []
        let jobSubmitCount = 0
        page.on('pageerror', (error) => browserErrors.push('pageerror: ' + error.message))
        page.on('console', (message) => { if (message.type() === 'error') browserErrors.push('console: ' + message.text()) })
        page.on('requestfailed', (requestEvent) => failedRequests.push(requestEvent.method() + ' ' + requestEvent.url() + ' ' + (requestEvent.failure()?.errorText ?? 'unknown')))
        page.on('request', (requestEvent) => {
          if (requestEvent.method() === 'POST' && requestEvent.url().includes('/api/execution/jobs')) jobSubmitCount += 1
        })
        try {
          await page.goto('/builder/' + fixture.id)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
          const transformerNodeId = await addCatalogNode(page, candidate.catalogNode, candidate.definitionId)
          await setTransformerParameters(page, transformerNodeId, candidate.parameters)
          await connectCase(page, candidate)

          const validBeforeRepair = await validateGraph(page, transformerNodeId, candidate.edgeCount)
          expect(validBeforeRepair.valid).toBe(true)
          const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
          const layers = (await parameterField(inspector, /Layers/i)).locator('input')
          await layers.fill('0')
          await expect(layers).toHaveValue('0')
          await expect(inspector.locator('.parameter-error').filter({ hasText: /valid layers/i })).toBeVisible()
          await expect(page.getByRole('button', { name: 'Validate', exact: true })).toBeDisabled()
          await expect(page.getByRole('button', { name: 'Save model', exact: true })).toBeDisabled()
          await mkdir(evidenceDir, { recursive: true })
          await page.screenshot({ path: path.join(evidenceDir, 'nw-t3-07-invalid-layers.png'), fullPage: true })
          await writeFile(path.join(evidenceDir, 'nw-t3-07-invalid-layers.json'), JSON.stringify({
            case: 'nw-t3-07-invalid-layers',
            project_id: fixture.id,
            authored_parameters: { ...candidate.parameters, layers: 0 },
            ui_parameter_error: 'Enter a valid layers.',
            validate_action: 'disabled',
            save_action: 'disabled',
            server_validation_submitted: false,
            execution_job_submissions: jobSubmitCount,
            browser_errors: browserErrors,
            failed_requests: failedRequests,
          }, null, 2))

          await layers.fill('1')
          await expect(layers).toHaveValue('1')
          await expect(page.getByRole('button', { name: 'Validate', exact: true })).toBeEnabled()
          const completed = await validateSaveAndRun(page, request, fixture, transformerNodeId, repaired, browserErrors, failedRequests)
          expect(completed.saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)?.parameters.layers).toBe(1)
          await writeFile(path.join(evidenceDir, 'nw-t3-07-repaired-layers.json'), JSON.stringify({
            case: 'nw-t3-07-repaired-layers',
            project_id: fixture.id,
            invalid_ui_state: {
              layers: 0,
              validate_action: 'disabled',
              save_action: 'disabled',
            },
            repaired_saved_revision: completed.saved.current_revision,
            repaired_graph: {
              node_count: completed.saved.graph.nodes.length,
              edge_count: completed.saved.graph.edges.length,
              layers: completed.saved.graph.nodes.find((item) => item.definition_id === candidate.definitionId)?.parameters.layers,
            },
            submitted_job_id: completed.submittedJobId,
            final_job: completed.finalJob,
            execution_job_submissions: jobSubmitCount,
            browser_errors: browserErrors,
            failed_requests: failedRequests,
          }, null, 2))
          await page.goto('/builder/' + fixture.id)
          await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
          const persistedNode = page.getByRole('group', { name: 'Node ' + candidate.catalogNode, exact: true })
          await persistedNode.click()
          const persistedInspector = page.getByRole('complementary', { name: 'Properties inspector' })
          await expect((await parameterField(persistedInspector, /Layers/i)).locator('input')).toHaveValue('1')
          expect(jobSubmitCount).toBe(1)
          expect(browserErrors).toEqual([])
          expect(failedRequests).toEqual([])
        } finally {
          await deleteProject(request, fixture.id)
        }
      })
    })
  })
