import { access, mkdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test } from '@playwright/test'

const fixtureRoot = path.resolve(
  process.env.NEURO_WEAVE_E2E_FIXTURE_DIR ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA'),
)
const fixturePath = path.join(fixtureRoot, `dataset-registration-${process.pid}.csv`)
const datasetName = `E2E patient outcomes ${process.pid}`

test('registers, reloads, inspects, and deletes a tabular dataset without deleting its source', async ({ page, request }) => {
  test.setTimeout(90_000)
  let datasetId: string | undefined
  await mkdir(fixtureRoot, { recursive: true })
  await writeFile(fixturePath, 'age,weight,diagnosis\n42,71,positive\n50,80,negative\n61,90,positive\n', 'utf8')

  try {
    await page.goto('/datasets')
    await page.locator('header').getByRole('button', { name: 'Register dataset' }).click()
    await page.getByRole('button', { name: 'Continue' }).click()

    await page.getByPlaceholder('e.g. Patient outcomes').fill(datasetName)
    await page.getByRole('button', { name: 'Browse…' }).click()
    const browser = page.getByRole('dialog').filter({ hasText: 'Select CSV table' })
    await browser.getByLabel('Filesystem path').fill(path.dirname(fixturePath))
    await browser.getByRole('button', { name: 'Browse' }).click()
    await browser.locator('.browser-item').filter({ hasText: path.basename(fixturePath) }).click()
    await browser.getByRole('button', { name: 'Use selected file' }).click()

    await page.getByRole('button', { name: 'Inspect & continue' }).click()
    await page.locator('.role-row').filter({ hasText: /^age/ }).locator('select').selectOption('input')
    await page.locator('.role-row').filter({ hasText: /^weight/ }).locator('select').selectOption('input')
    await page.locator('.role-row').filter({ hasText: /^diagnosis/ }).locator('select').selectOption('output')
    await page.getByRole('button', { name: 'Continue' }).click()
    await page.getByRole('button', { name: 'Continue' }).click()
    await page.getByRole('button', { name: 'Validate definition' }).click()

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
    await page.getByRole('link', { name: 'Structure', exact: true }).click()
    await expect(page.locator('body')).toContainText('diagnosis')
    await expect(page.locator('body')).toContainText('Output')
    await page.reload()
    await expect(page.getByRole('heading', { name: datasetName })).toBeVisible()

    await page.getByRole('link', { name: 'All datasets' }).click()
    const catalogCard = page.locator('.dataset-card').filter({ hasText: datasetName })
    await catalogCard.getByRole('button', { name: `Delete registration for ${datasetName}` }).click()
    await page.getByRole('button', { name: 'Delete registration', exact: true }).click()
    await expect(catalogCard).toHaveCount(0)
    await access(fixturePath)
  } finally {
    if (datasetId) {
      const response = await request.delete(`/api/datasets/${datasetId}`)
      expect([204, 404]).toContain(response.status())
    }
    await rm(fixturePath, { force: true })
  }
})
