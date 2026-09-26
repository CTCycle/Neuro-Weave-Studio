import fs from 'node:fs/promises'
import path from 'node:path'
import { expect, test } from '@playwright/test'

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
