# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: workflows.spec.ts >> isolates builder tabs and exposes canvas and validation controls
- Location: e2e\workflows.spec.ts:784:1

# Error details

```
Error: expect(locator).toBeVisible() failed

Locator: locator('button[role="tab"]').filter({ hasText: 'Untitled model' })
Expected: visible
Timeout: 5000ms
Error: element(s) not found

Call log:
  - Expect "toBeVisible" with timeout 5000ms
  - waiting for locator('button[role="tab"]').filter({ hasText: 'Untitled model' })

```

```yaml
- banner:
  - link "NeuroWeave models":
    - /url: /models
    - strong: NeuroWeave
    - text: MODEL STUDIO
  - navigation "Primary navigation":
    - link "Models":
      - /url: /models
    - link "Model Builder":
      - /url: /builder
    - link "Nodes":
      - /url: /nodes
    - link "Datasets":
      - /url: /datasets
    - link "Training":
      - /url: /training
    - link "Metrics":
      - /url: /metrics
    - link "Environment":
      - /url: /environment
    - link "Settings":
      - /url: /settings
- main:
  - text: MODEL BUILDER Saved
  - textbox "Model name":
    - /placeholder: Untitled model
    - text: Angular tabs 1790583777352
  - button "Description"
  - text: Backend
  - combobox "Execution backend": PyTorch
  - text: Enable Keras
  - checkbox "Enable Keras" [checked]
  - strong: "On"
  - button "Undo" [disabled]
  - button "Redo" [disabled]
  - button "Validate"
  - button "Save model"
  - button "Open workspace"
  - complementary "Node catalog":
    - text: BUILDING BLOCKS
    - heading "Node catalog" [level=2]
    - text: 149 available
    - textbox "Search nodes":
      - /placeholder: Search layers, operations, templates
    - button "Filters"
    - button "Convolutional 18 nodes":
      - strong: Convolutional
      - text: 18 nodes
    - button "Core 14 nodes":
      - strong: Core
      - text: 14 nodes
    - button "Inputs 8 nodes":
      - strong: Inputs
      - text: 8 nodes
    - button "Mathematical 73 nodes":
      - strong: Mathematical
      - text: 73 nodes
    - button "Merge 14 nodes":
      - strong: Merge
      - text: 14 nodes
    - button "Outputs 7 nodes":
      - strong: Outputs
      - text: 7 nodes
    - button "Recurrent 5 nodes":
      - strong: Recurrent
      - text: 5 nodes
    - button "Transformer 10 nodes":
      - strong: Transformer
      - text: 10 nodes
  - navigation "Open models":
    - tablist "Model editor tabs":
      - tab "○ Angular tabs 1790583777352 r1" [selected]
      - button "Close Angular tabs 1790583777352"
    - button "New model"
  - region "Model graph canvas":
    - img
    - group "Connection from input-tabular-e2e|output|output to layer-dense-e2e-1|input|input":
      - img
    - group "Connection from layer-dense-e2e-1|output|output to layer-dense-e2e-2|input|input":
      - img
    - group "Connection from layer-dense-e2e-2|output|output to output-classification-e2e|input|input":
      - img
    - group "Node Features":
      - text: inputs
      - strong: Features
      - text: input.tabular Shape pending
      - button "Port Output (output) on Features"
    - group "Node Hidden layer":
      - text: core
      - strong: Hidden layer
      - text: layer.dense Shape pending
      - button "Port Input (input) on Hidden layer"
      - button "Port Output (output) on Hidden layer"
    - group "Node Classifier":
      - text: core
      - strong: Classifier
      - text: layer.dense Shape pending
      - button "Port Input (input) on Classifier"
      - button "Port Output (output) on Classifier"
    - group "Node Prediction":
      - text: outputs
      - strong: Prediction
      - text: output.classification Shape pending
      - button "Port Input (input) on Prediction"
      - button "Port Output (output) on Prediction"
    - group "Canvas zoom controls":
      - button "Zoom in": +
      - button "Zoom out": −
      - button "Reset view"
    - toolbar "Graph editor controls":
      - button "Hide minimap": Hide map
      - button "Lock canvas"
      - button "Enter full canvas": Focus
    - text: Drop a node to build the model • Ctrl+L auto-layout
  - button "GRAPH HEALTH Validation Not run":
    - text: GRAPH HEALTH
    - strong: Validation
    - text: Not run
  - complementary "Properties inspector":
    - text: INSPECTOR
    - heading "Properties" [level=2]
    - text: SELECTION
    - heading "No node selected" [level=2]
    - paragraph: Select a node to edit its label and parameters.
  - button "No operational execution backend is available for a new model."
```

# Test source

```ts
  692 |     await expect(page.getByText('succeeded', { exact: true })).toBeVisible({ timeout: 30_000 })
  693 |     await expect(page.getByText(/Completed on/)).toBeVisible()
  694 |
  695 |     const secondSubmit = page.waitForResponse((response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST')
  696 |     await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
  697 |     const secondJob = await (await secondSubmit).json() as { id: string }
  698 |     await page.reload()
  699 |
  700 |     await expect.poll(async () => {
  701 |       const response = await request.get(`/api/execution/jobs/${secondJob.id}`)
  702 |       return response.ok() ? String((await response.json() as { state: string }).state) : 'missing'
  703 |     }, { timeout: 30_000 }).not.toBe('missing')
  704 |     let secondResult = await (await request.get(`/api/execution/jobs/${secondJob.id}`)).json() as {
  705 |       state: string
  706 |       worker: { api: string; backend: string }
  707 |       cancellation_requested: boolean
  708 |     }
  709 |     expect(secondResult.worker).toMatchObject({ api: 'keras', backend: 'pytorch' })
  710 |     if (!['succeeded', 'terminated', 'failed'].includes(secondResult.state)) {
  711 |       await expect(page.getByRole('button', { name: 'Stop', exact: true })).toBeVisible({ timeout: 30_000 })
  712 |       const stopResponse = page.waitForResponse((response) => response.url().includes(`/api/execution/jobs/${secondJob.id}/stop`) && response.request().method() === 'POST')
  713 |       await page.getByRole('button', { name: 'Stop', exact: true }).click()
  714 |       expect((await stopResponse).ok()).toBeTruthy()
  715 |       await expect(page.getByText('terminated', { exact: true })).toBeVisible({ timeout: 30_000 })
  716 |       secondResult = await (await request.get(`/api/execution/jobs/${secondJob.id}`)).json() as typeof secondResult
  717 |       expect(secondResult.state).toBe('terminated')
  718 |       expect(secondResult.cancellation_requested).toBeTruthy()
  719 |     } else {
  720 |       expect(secondResult.state).toBe('succeeded')
  721 |       await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 30_000 })
  722 |     }
  723 |     await page.screenshot({ path: path.join(evidenceDir, 'nw-t2-04-validation-restored-after-reload.png'), fullPage: true })
  724 |   } finally {
  725 |     await deleteProject(request, fixture.id)
  726 |   }
  727 | })
  728 |
  729 | test('exposes keyboard-traversable graph nodes and ports', async ({ page, request }) => {
  730 |   const fixture = await createProject(request, `Angular graph accessibility ${Date.now()}`, fanOutGraph())
  731 |   try {
  732 |     await page.goto(`/builder/${fixture.id}`)
  733 |     const nodes = page.locator('[data-node-id]')
  734 |     const ports = page.locator('.nw-flow-port[role="button"]')
  735 |     await expect(nodes).toHaveCount(7)
  736 |     await expect(nodes.first()).toHaveAttribute('tabindex', '0')
  737 |     await expect(nodes.first()).toHaveAttribute('role', 'group')
  738 |     await expect(ports).not.toHaveCount(0)
  739 |
  740 |     const source = page.locator('[data-node-id="input-branch-e2e"] .nw-flow-port.output')
  741 |     const target = page.locator('[data-node-id="dense-fanout-e2e"] .nw-flow-port.input')
  742 |     await source.focus()
  743 |     await source.press('Enter')
  744 |     await expect(source).toHaveAttribute('aria-pressed', 'true')
  745 |     await expect(page.getByRole('status')).toContainText('Choose an input port')
  746 |
  747 |     await target.focus()
  748 |     const validationResponse = page.waitForResponse((response) => response.url().includes('/api/validation/connection') && response.request().method() === 'POST')
  749 |     await target.press('Enter')
  750 |     expect((await validationResponse).ok()).toBeTruthy()
  751 |     await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(5)
  752 |     await expect(source).toHaveAttribute('aria-pressed', 'false')
  753 |   } finally {
  754 |     await deleteProject(request, fixture.id)
  755 |   }
  756 | })
  757 |
  758 | test('keeps edge selection, inspector details, deletion, and undo synchronized', async ({ page, request }) => {
  759 |   const fixture = await createProject(request, `Angular edge state ${Date.now()}`)
  760 |   try {
  761 |     await page.goto(`/builder/${fixture.id}`)
  762 |     await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(3)
  763 |     await waitForCanvasFit(page)
  764 |     await page.locator('f-connection path.f-connection-selection').first().click({ force: true })
  765 |     await expect(page.locator('f-connection.is-selected')).toHaveCount(1)
  766 |     await expect(page.getByRole('heading', { name: 'Connection' })).toBeVisible()
  767 |     await expect(page.getByRole('button', { name: 'Delete connection' })).toBeVisible()
  768 |     await expect(page.getByRole('complementary', { name: 'Properties inspector' })).toContainText('Features')
  769 |     await expect(page.getByRole('complementary', { name: 'Properties inspector' })).toContainText('Hidden layer')
  770 |
  771 |     await page.locator('body').press('Delete')
  772 |     await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(2)
  773 |     await expect(page.getByRole('heading', { name: 'No node selected' })).toBeVisible()
  774 |     await page.getByRole('button', { name: 'Undo', exact: true }).click()
  775 |     await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(3)
  776 |
  777 |     const persisted = await (await request.get(`/api/projects/${fixture.id}`)).json() as { graph: { edges: unknown[] } }
  778 |     expect(persisted.graph.edges).toHaveLength(3)
  779 |   } finally {
  780 |     await deleteProject(request, fixture.id)
  781 |   }
  782 | })
  783 |
  784 | test('isolates builder tabs and exposes canvas and validation controls', async ({ page, request }) => {
  785 |   const fixture = await createProject(request, `Angular tabs ${Date.now()}`)
  786 |   try {
  787 |     await page.goto(`/builder/${fixture.id}`)
  788 |     const savedTab = page.locator('button[role="tab"]').filter({ hasText: fixture.name })
  789 |     await expect(savedTab).toBeVisible()
  790 |     await page.getByRole('button', { name: 'New model', exact: true }).click()
  791 |     const draftTab = page.locator('button[role="tab"]').filter({ hasText: 'Untitled model' })
> 792 |     await expect(draftTab).toBeVisible()
      |                            ^ Error: expect(locator).toBeVisible() failed
  793 |     await page.getByRole('textbox', { name: 'Search nodes' }).fill('Dense')
  794 |     await page.getByRole('button', { name: /Dense \/ Linear/ }).click()
  795 |     await expect(page.locator('.nw-flow-node')).toHaveCount(1)
  796 |
  797 |     await savedTab.click()
  798 |     await expect(page.locator('.nw-flow-node')).toHaveCount(4)
  799 |     await expect(page.locator('f-connection path.f-connection-selection')).toHaveCount(3)
  800 |     await page.getByRole('button', { name: 'Validate', exact: true }).click()
  801 |     await expect(page.locator('.validation-summary')).toContainText('Ready')
  802 |     await page.locator('.builder-notice').click()
  803 |     await page.locator('.validation-summary').click()
  804 |     await expect(page.getByRole('separator', { name: 'Resize validation panel' })).toBeVisible()
  805 |
  806 |     await draftTab.click()
  807 |     await expect(page.locator('.nw-flow-node')).toHaveCount(1)
  808 |     await page.getByRole('button', { name: 'Hide minimap' }).click()
  809 |     await expect(page.getByRole('button', { name: 'Show minimap' })).toBeVisible()
  810 |     await page.getByRole('button', { name: 'Enter full canvas' }).click()
  811 |     await expect(page.getByRole('button', { name: 'Exit full canvas' })).toBeVisible()
  812 |     await page.getByRole('button', { name: 'Exit full canvas' }).click()
  813 |   } finally {
  814 |     await deleteProject(request, fixture.id)
  815 |   }
  816 | })
  817 |
  818 | test('covers builder editing, history, focus mode, and dirty navigation', async ({ page, request }) => {
  819 |   const fixture = await createProject(request, `Angular builder ${Date.now()}`)
  820 |   try {
  821 |     await page.goto('/builder')
  822 |     await expect(page.getByRole('heading', { name: 'Node catalog' })).toBeVisible()
  823 |     await page.screenshot({ path: path.join(visualEvidenceDir, 'final-builder-empty.png'), fullPage: true })
  824 |
  825 |     await page.getByRole('textbox', { name: 'Search nodes' }).fill('Dense')
  826 |     await page.getByRole('button', { name: /Dense \/ Linear/ }).click()
  827 |     await expect(page.locator('.nw-flow-node')).toHaveCount(1)
  828 |     await page.locator('.nw-flow-node').first().click()
  829 |     await expect(page.getByRole('heading', { name: 'Dense / Linear' })).toBeVisible()
  830 |
  831 |     const label = page.getByLabel('Label')
  832 |     await label.fill('Regression layer')
  833 |     await expect(label).toHaveValue('Regression layer')
  834 |     await page.locator('.nw-flow-node').click()
  835 |     await page.keyboard.press('Control+c')
  836 |     await page.keyboard.press('Control+v')
  837 |     await expect(page.locator('.nw-flow-node')).toHaveCount(2)
  838 |     await page.getByRole('button', { name: 'Undo' }).click()
  839 |     await expect(page.locator('.nw-flow-node')).toHaveCount(1)
  840 |
  841 |     await page.getByRole('button', { name: 'Enter full canvas' }).click()
  842 |     await expect(page.getByRole('button', { name: 'Exit full canvas' })).toBeVisible()
  843 |     await page.keyboard.press('Escape')
  844 |     await expect(page.getByRole('button', { name: 'Enter full canvas' })).toBeVisible()
  845 |
  846 |     await page.getByRole('link', { name: 'Models', exact: true }).click()
  847 |     await expect(page.getByRole('dialog')).toContainText('Leave the Model Builder?')
  848 |     await page.getByRole('button', { name: 'Stay' }).click()
  849 |     await expect(page).toHaveURL(/\/builder$/)
  850 |     await page.getByRole('link', { name: 'Models', exact: true }).click()
  851 |     await page.getByRole('button', { name: 'Discard changes' }).click()
  852 |     await expect(page).toHaveURL(/\/models$/)
  853 |     await expect(page.getByRole('heading', { name: 'Saved models' })).toBeVisible()
  854 |   } finally {
  855 |     await deleteProject(request, fixture.id)
  856 |   }
  857 | })
  858 |
  859 | test('NW-T2-01 verifies Builder selection, editing, history, save/reopen, and invalid-draft recovery', async ({ page, request }) => {
  860 |   test.setTimeout(120_000)
  861 |   const diagnostics = {
  862 |     pageErrors: [] as string[],
  863 |     consoleErrors: [] as string[],
  864 |     failedRequests: [] as string[],
  865 |     httpErrors: [] as Array<{ method: string; path: string; status: number }>,
  866 |   }
  867 |   const projectWrites: string[] = []
  868 |   page.on('pageerror', (error) => diagnostics.pageErrors.push(error.message))
  869 |   page.on('console', (message) => {
  870 |     if (message.type() === 'error') diagnostics.consoleErrors.push(message.text())
  871 |   })
  872 |   page.on('requestfailed', (request) => {
  873 |     diagnostics.failedRequests.push(`${request.method()} ${request.url()}: ${request.failure()?.errorText ?? 'failed'}`)
  874 |   })
  875 |   page.on('request', (request) => {
  876 |     const pathname = new URL(request.url()).pathname
  877 |     if (request.method() === 'PUT' && pathname.startsWith('/api/projects/')) projectWrites.push(pathname)
  878 |   })
  879 |   page.on('response', (response) => {
  880 |     if (response.status() >= 400) {
  881 |       diagnostics.httpErrors.push({
  882 |         method: response.request().method(),
  883 |         path: new URL(response.url()).pathname,
  884 |         status: response.status(),
  885 |       })
  886 |     }
  887 |   })
  888 |
  889 |   await mkdir(campaignEvidenceDir, { recursive: true })
  890 |   const fixture = await createProject(request, `NW T2-01 Builder ${Date.now()}`)
  891 |   const projectUrl = `/api/projects/${fixture.id}`
  892 |   const hiddenLayer = page.locator('.nw-flow-node').filter({ hasText: 'Hidden layer' })
```