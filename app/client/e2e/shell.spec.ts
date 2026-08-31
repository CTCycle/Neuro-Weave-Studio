import { expect, test } from '@playwright/test'

test('starts the Angular shell at the model catalog', async ({ page }) => {
  await page.goto('/models')
  await expect(page.locator('nw-app-shell')).toBeVisible()
  await expect(page.getByRole('link', { name: 'Models', exact: true })).toHaveClass(/active/)
  await expect(page.locator('app-root')).toContainText('Saved models')
})

test('serves Angular deep links and preserves the shell on refresh', async ({ page }) => {
  test.setTimeout(60_000)
  for (const route of ['/environment', '/metrics', '/builder', '/models/missing/compilation']) {
    await page.goto(route)
    await expect(page.locator('nw-app-shell')).toBeVisible()
    await page.reload()
    await expect(page.locator('nw-app-shell')).toBeVisible()
  }
})
