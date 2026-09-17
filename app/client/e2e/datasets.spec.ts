import { access, mkdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'

const fixtureRoot = path.resolve(
  process.env.NEURO_WEAVE_E2E_FIXTURE_DIR ?? path.resolve(process.cwd(), '..', '..', 'assets', 'QA'),
)
const fixturePath = path.join(fixtureRoot, `dataset-registration-${process.pid}.csv`)
const datasetName = `E2E patient outcomes ${process.pid}`

test.describe.configure({ mode: 'serial' })

function trackDatasetPosts(page: Page): string[] {
  const requests: string[] = []
  page.on('request', (request) => {
    const pathname = new URL(request.url()).pathname
    if (request.method() === 'POST' && ['/api/datasets', '/api/datasets/inspect', '/api/datasets/validate'].includes(pathname)) {
      requests.push(`${request.method()} ${pathname}`)
    }
  })
  return requests
}

async function prepareTabularWizard(page: Page, sourcePath: string, name: string): Promise<void> {
  await page.goto('/datasets')
  await page.locator('header').getByRole('button', { name: 'Register dataset' }).click()
  await page.getByRole('button', { name: 'Continue' }).click()

  await page.getByPlaceholder('e.g. Patient outcomes').fill(name)
  await page.getByRole('button', { name: 'Browse…' }).click()
  const browser = page.getByRole('dialog').filter({ hasText: 'Select CSV table' })
  const filesystemPath = browser.getByLabel('Filesystem path')
  await filesystemPath.fill(path.dirname(sourcePath))
  await expect(browser.locator('.browser-body')).toHaveAttribute('aria-busy', 'false', { timeout: 30_000 })
  await expect(filesystemPath).toHaveValue(path.dirname(sourcePath))
  const browseResponse = page.waitForResponse((response) => response.url().includes('/api/filesystem/browse') && response.request().method() === 'GET')
  await browser.getByRole('button', { name: 'Browse' }).click()
  expect((await browseResponse).ok()).toBeTruthy()
  await expect(browser.locator('.browser-body')).toHaveAttribute('aria-busy', 'false', { timeout: 30_000 })
  const fixtureEntry = browser.locator('.browser-item').filter({ hasText: path.basename(sourcePath) })
  await expect(fixtureEntry).toBeVisible({ timeout: 30_000 })
  await fixtureEntry.click()
  await browser.getByRole('button', { name: 'Use selected file' }).click()

  await page.getByRole('button', { name: 'Inspect & continue' }).click()
  await page.locator('.role-row').filter({ hasText: /^age/ }).locator('select').selectOption('input')
  await page.locator('.role-row').filter({ hasText: /^weight/ }).locator('select').selectOption('input')
  await page.locator('.role-row').filter({ hasText: /^diagnosis/ }).locator('select').selectOption('output')
  await page.getByRole('button', { name: 'Continue' }).click()
  await page.getByRole('button', { name: 'Continue' }).click()
}

test('registers, reloads, inspects, and deletes a tabular dataset without deleting its source', async ({ page, request }) => {
  test.setTimeout(90_000)
  let datasetId: string | undefined
  const datasetPosts = trackDatasetPosts(page)
  await mkdir(fixtureRoot, { recursive: true })
  await writeFile(fixturePath, 'age,weight,diagnosis\n42,71,positive\n50,80,negative\n61,90,positive\n', 'utf8')

  try {
    await prepareTabularWizard(page, fixturePath, datasetName)
    await page.getByRole('button', { name: 'Continue to registration' }).click()

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
    expect(datasetPosts).toEqual(['POST /api/datasets/inspect', 'POST /api/datasets'])
  } finally {
    if (datasetId) {
      const response = await request.delete(`/api/datasets/${datasetId}`)
      expect([204, 404]).toContain(response.status())
    }
    await rm(fixturePath, { force: true })
  }
})

test('returns authoritative registration findings to Review without a pre-validation request', async ({ page }) => {
  test.setTimeout(90_000)
  const invalidFixturePath = path.join(fixtureRoot, `dataset-registration-invalid-${process.pid}.csv`)
  const invalidDatasetName = `E2E invalid patient outcomes ${process.pid}`
  const datasetPosts = trackDatasetPosts(page)
  await mkdir(fixtureRoot, { recursive: true })
  await writeFile(invalidFixturePath, 'age,weight,diagnosis\n42,71,positive\n', 'utf8')

  try {
    await prepareTabularWizard(page, invalidFixturePath, invalidDatasetName)
    await writeFile(invalidFixturePath, 'age,diagnosis\n42,positive\n', 'utf8')
    await page.getByRole('button', { name: 'Continue to registration' }).click()

    const rejectedResponse = page.waitForResponse((response) => response.url().endsWith('/api/datasets') && response.request().method() === 'POST')
    await page.locator('nw-dataset-wizard').getByRole('button', { name: 'Register dataset' }).click()
    const rejected = await rejectedResponse

    expect(rejected.status()).toBe(422)
    await expect(page.getByRole('heading', { name: 'Review' })).toBeVisible()
    await expect(page.locator('.validation-result')).toContainText('Definition needs attention')
    await expect(page.locator('.validation-result')).toContainText('missing_columns')
    await expect(page.locator('.validation-result')).toContainText('Columns do not exist: weight')
    expect(datasetPosts).toEqual(['POST /api/datasets/inspect', 'POST /api/datasets'])
  } finally {
    await rm(invalidFixturePath, { force: true })
  }
})
