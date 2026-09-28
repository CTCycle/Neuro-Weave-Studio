# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: workflows.spec.ts >> NW-T2-04 restores functional-validation results and active jobs after reload
- Location: e2e\workflows.spec.ts:662:1

# Error details

```
Error: expect(locator).toBeVisible() failed

Locator: getByText(/Completed on/)
Expected: visible
Timeout: 75000ms
Error: element(s) not found

Call log:
  - Expect "toBeVisible" with timeout 75000ms
  - waiting for getByText(/Completed on/)

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
  - link "All models":
    - /url: /models
  - text: MODEL WORKSPACE
  - heading "NW-T2-04 validation reload 1790583775208" [level=1]
  - paragraph: Browser regression fixture
  - link "Edit architecture":
    - /url: /builder/6518bf3e-71a6-4313-9a15-c0e375cbaef9
  - button "Duplicate"
  - button "Delete model"
  - text: Ready Not compiled Revision 1 1 input · 1 output 4 nodes · 3 connections Keras · PyTorch backend
  - navigation "Model workspace sections":
    - link "Overview":
      - /url: /models/6518bf3e-71a6-4313-9a15-c0e375cbaef9
    - link "Compilation":
      - /url: /models/6518bf3e-71a6-4313-9a15-c0e375cbaef9/compilation
    - link "Validation":
      - /url: /models/6518bf3e-71a6-4313-9a15-c0e375cbaef9/validation
    - link "Export":
      - /url: /models/6518bf3e-71a6-4313-9a15-c0e375cbaef9/export
  - text: FUNCTIONAL VALIDATION
  - heading "Run a safe model check" [level=2]
  - paragraph: The worker creates compatible dummy inputs and performs one forward pass. It may run a temporary optimizer step in worker memory to verify the optimizer; it does not train or change the saved model, or serve user predictions.
  - text: Model backend preference
  - strong: keras · pytorch · cpu
  - text: Functional validation uses the model's persisted backend and its matching runtime preference. It never changes saved model data.
  - link "Configure runtime":
    - /url: /environment
  - text: FUNCTIONAL CHECK
  - heading "One forward pass" [level=3]
  - paragraph: The worker returns input/output tensor shapes, device information, and parameter count.
  - button "Run functional validation"
  - alert: Device 'cpu' already has an active ML job
```

# Test source

```ts
  579 |     await page.goto(`/builder/${fixture.id}`)
  580 |     const kerasToggle = page.getByLabel('Enable Keras', { exact: true })
  581 |     await expect(kerasToggle).toBeChecked()
  582 |     await kerasToggle.uncheck()
  583 |
  584 |     const incompatibleNode = page.locator('[data-node-id="simple-rnn-compat-e2e"]')
  585 |     await expect(incompatibleNode).toHaveClass(/has-validation-error/)
  586 |     await expect(incompatibleNode).toHaveAttribute('aria-label', 'Node Simple RNN')
  587 |
  588 |     await page.getByRole('textbox', { name: 'Search nodes' }).fill('Simple RNN')
  589 |     const catalogItem = page.locator('.catalog-item').filter({ hasText: 'Simple RNN' }).first()
  590 |     await expect(catalogItem).toBeEnabled()
  591 |
  592 |     await incompatibleNode.click()
  593 |     const inspector = page.getByRole('complementary', { name: 'Properties inspector' })
  594 |     await expect(inspector.locator('label.inspector-field').filter({ hasText: 'Units' }).locator('input')).toBeEnabled()
  595 |     await expect(inspector.locator('label.inspector-field').filter({ hasText: 'Return Sequences' }).locator('input')).toBeEnabled()
  596 |     await expect(inspector.locator('label.inspector-field').filter({ hasText: 'Return state' }).locator('input')).toBeDisabled()
  597 |     await page.screenshot({ path: path.join(evidenceDir, 'profile-parameter-incompatible-builder.png'), fullPage: true })
  598 |
  599 |     const saveResponse = page.waitForResponse((response) => response.url().includes(`/api/projects/${fixture.id}`) && response.request().method() === 'PUT')
  600 |     await page.getByRole('button', { name: 'Save model', exact: true }).click()
  601 |     const saved = await (await saveResponse).json() as { current_revision: number; keras_enabled: boolean; validation_status: string; graph: { nodes: unknown[] } }
  602 |     expect(saved.keras_enabled).toBe(false)
  603 |     expect(saved.validation_status).toBe('invalid')
  604 |     expect(saved.graph.nodes).toHaveLength(3)
  605 |     await expect(page.locator('.builder-badge')).toContainText('Saved')
  606 |
  607 |     const persisted = await (await request.get(`/api/projects/${fixture.id}`)).json() as { keras_enabled: boolean; validation_status: string; graph: { nodes: Array<{ id: string }> } }
  608 |     expect(persisted).toMatchObject({ keras_enabled: false, validation_status: 'invalid' })
  609 |     expect(persisted.graph.nodes.map((node) => node.id)).toContain('simple-rnn-compat-e2e')
  610 |
  611 |     await page.goto(`/models/${fixture.id}/compilation`)
  612 |     await expect(page.getByRole('button', { name: 'Compile source', exact: true })).toBeDisabled()
  613 |     await page.getByRole('button', { name: 'Validate profile', exact: true }).click()
  614 |     await expect(page.getByText(/Parameter 'Return state' is not supported for Native PyTorch/i)).toBeVisible()
  615 |     await page.screenshot({ path: path.join(evidenceDir, 'invalid-compilation.png'), fullPage: true })
  616 |
  617 |     await page.goto(`/models/${fixture.id}/export`)
  618 |     await expect(page.getByRole('button', { name: /Project bundle/ })).toBeDisabled()
  619 |     await expect(page.getByRole('button', { name: /Project bundle/ })).toContainText('Fix the validation diagnostics')
  620 |     const pytorchExport = page.getByRole('button', { name: /PyTorch ExportedProgram/ })
  621 |     await expect(pytorchExport).toBeDisabled()
  622 |     await expect(pytorchExport).toHaveAttribute('title', /(?:Unavailable|Native runtime|profile|probed)/i)
  623 |     await expect(pytorchExport).toContainText(/(?:Unavailable for this profile|Native runtime is unavailable|capability has not been probed)/i)
  624 |     await page.screenshot({ path: path.join(evidenceDir, 'invalid-export-capabilities.png'), fullPage: true })
  625 |
  626 |     await page.goto(`/builder/${fixture.id}`)
  627 |     await expect(page.getByLabel('Enable Keras', { exact: true })).not.toBeChecked()
  628 |     await page.getByLabel('Enable Keras', { exact: true }).check()
  629 |     await expect(page.locator('[data-node-id="simple-rnn-compat-e2e"]')).not.toHaveClass(/has-validation-error/)
  630 |   } finally {
  631 |     await deleteProject(request, fixture.id)
  632 |   }
  633 | })
  634 |
  635 | test('polls a compatible functional-validation job to a verified result', async ({ page, request }) => {
  636 |   test.setTimeout(120_000)
  637 |   const fixture = await createProject(request, `Angular compatibility smoke ${Date.now()}`)
  638 |   const evidenceDir = process.env.NW_VALIDATION_EVIDENCE_DIR
  639 |     ? campaignEvidenceDir
  640 |     : path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'backend-compatibility-consolidation')
  641 |   await mkdir(evidenceDir, { recursive: true })
  642 |   try {
  643 |     await page.goto(`/models/${fixture.id}/validation`)
  644 |     await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
  645 |     await expect(page.getByText(/keras · pytorch ·/i)).toBeVisible({ timeout: 75_000 })
  646 |     const submitResponse = page.waitForResponse((response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST')
  647 |     await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
  648 |     const submitted = await submitResponse
  649 |     expect(submitted.ok()).toBeTruthy()
  650 |     const job = await submitted.json() as { id: string }
  651 |     await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 75_000 })
  652 |     await expect.poll(async () => {
  653 |       const response = await request.get(`/api/execution/jobs/${job.id}`)
  654 |       return response.ok() ? (await response.json() as { state: string }).state : 'missing'
  655 |     }, { timeout: 75_000 }).toBe('succeeded')
  656 |     await page.screenshot({ path: path.join(evidenceDir, 'smoke-job-succeeded.png'), fullPage: true })
  657 |   } finally {
  658 |     await deleteProject(request, fixture.id)
  659 |   }
  660 | })
  661 |
  662 | test('NW-T2-04 restores functional-validation results and active jobs after reload', async ({ page, request }) => {
  663 |   test.setTimeout(120_000)
  664 |   const fixture = await createProject(request, `NW-T2-04 validation reload ${Date.now()}`)
  665 |   const evidenceDir = process.env.NW_VALIDATION_EVIDENCE_DIR
  666 |     ? campaignEvidenceDir
  667 |     : path.resolve(process.cwd(), '..', '..', 'assets', 'QA', 'comprehensive-validation-20260923', 'evidence')
  668 |   await mkdir(evidenceDir, { recursive: true })
  669 |   try {
  670 |     await page.goto(`/models/${fixture.id}/validation`)
  671 |     await expect(page.getByRole('heading', { name: 'Run a safe model check' })).toBeVisible({ timeout: 30_000 })
  672 |     await expect(page.getByText(/temporary optimizer step in worker memory/i)).toBeVisible()
  673 |     await expect(page.getByText(/does not train or change the saved model/i)).toBeVisible()
  674 |     await expect(page.getByText(/keras · pytorch ·/i)).toBeVisible({ timeout: 75_000 })
  675 |
  676 |     const firstSubmit = page.waitForResponse((response) => response.url().includes('/api/execution/jobs') && response.request().method() === 'POST')
  677 |     await page.getByRole('button', { name: 'Run functional validation', exact: true }).click()
  678 |     const firstJob = await (await firstSubmit).json() as { id: string }
> 679 |     await expect(page.getByText(/Completed on/)).toBeVisible({ timeout: 75_000 })
      |                                                  ^ Error: expect(locator).toBeVisible() failed
  680 |     const firstResult = await (await request.get(`/api/execution/jobs/${firstJob.id}`)).json() as {
  681 |       state: string
  682 |       worker: { api: string; backend: string }
  683 |       result: { status: string; inputs: unknown[]; outputs: unknown[]; parameter_count: number }
  684 |     }
  685 |     expect(firstResult).toMatchObject({
  686 |       state: 'succeeded',
  687 |       worker: { api: 'keras', backend: 'pytorch' },
  688 |       result: { status: 'succeeded', inputs: expect.any(Array), outputs: expect.any(Array), parameter_count: expect.any(Number) },
  689 |     })
  690 |
  691 |     await page.reload()
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
```