// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { expect, test } from '@playwright/test'

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
