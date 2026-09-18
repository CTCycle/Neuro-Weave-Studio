import { expect, test } from '@playwright/test'

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
  await expect(page.getByText('There are no editable application-level settings defined yet.', { exact: true })).toBeVisible()
  await expect(page.locator('.settings-content input, .settings-content select')).toHaveCount(0)
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
