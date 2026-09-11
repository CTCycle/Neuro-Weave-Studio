import { expect, test } from '@playwright/test'

const coreRoutes = ['/models', '/nodes', '/datasets', '/training', '/metrics', '/environment', '/settings', '/builder']

type AccessibilityAudit = {
  language: string
  title: string
  landmarkCount: { main: number; navigation: number }
  missingNames: Array<{ tag: string; role: string | null; html: string }>
}

test('runs the in-browser accessibility baseline across core routes', async ({ page }) => {
  test.setTimeout(120_000)
  for (const route of coreRoutes) {
    await page.goto(route)
    await expect(page.locator('nw-app-shell')).toBeVisible()
    const audit = await page.evaluate((): AccessibilityAudit => {
      const visible = (element: Element): boolean => {
        if (!(element instanceof HTMLElement)) return false
        const style = getComputedStyle(element)
        const bounds = element.getBoundingClientRect()
        return style.display !== 'none'
          && style.visibility !== 'hidden'
          && style.opacity !== '0'
          && bounds.width > 0
          && bounds.height > 0
      }
      const referencedName = (element: Element): string => {
        const ids = element.getAttribute('aria-labelledby')?.split(/\s+/).filter(Boolean) ?? []
        return ids.map((id) => document.getElementById(id)?.textContent ?? '').join(' ')
      }
      const controlName = (element: Element): string => {
        const ariaLabel = element.getAttribute('aria-label')?.trim()
        if (ariaLabel) return ariaLabel
        const labelledBy = referencedName(element).trim()
        if (labelledBy) return labelledBy
        if (element instanceof HTMLInputElement || element instanceof HTMLSelectElement || element instanceof HTMLTextAreaElement) {
          const label = element.labels?.[0]?.textContent?.trim()
          if (label) return label
        }
        const title = element.getAttribute('title')?.trim()
        if (title) return title
        return (element.textContent ?? '').replace(/\s+/g, ' ').trim()
      }
      const candidates = Array.from(document.querySelectorAll(
        'a,button,input,select,textarea,summary,[contenteditable="true"],[role="button"],[role="link"],[role="tab"],[role="checkbox"],[role="combobox"],[role="menuitem"]',
      )).filter(visible)
      return {
        language: document.documentElement.lang,
        title: document.title,
        landmarkCount: { main: document.querySelectorAll('main').length, navigation: document.querySelectorAll('nav').length },
        missingNames: candidates
          .filter((element) => !controlName(element))
          .map((element) => ({ tag: element.tagName.toLowerCase(), role: element.getAttribute('role'), html: element.outerHTML.slice(0, 180) })),
      }
    })
    expect(audit.language, route).toBe('en')
    expect(audit.title, route).toBe('NeuroWeave Model Studio')
    expect(audit.landmarkCount.main, route).toBe(1)
    expect(audit.landmarkCount.navigation, route).toBeGreaterThan(0)
    expect(audit.missingNames, route).toEqual([])
  }
})

test('keeps keyboard focus visible on core controls and graph landmarks', async ({ page }) => {
  await page.goto('/builder')
  await expect(page.locator('nw-app-shell')).toBeVisible()
  const focusSamples: Array<{ tag: string; outlineWidth: string; outlineStyle: string; boxShadow: string }> = []
  for (let index = 0; index < 12; index += 1) {
    await page.keyboard.press('Tab')
    const sample = await page.evaluate(() => {
      const element = document.activeElement
      if (!(element instanceof HTMLElement) || element === document.body) return null
      const style = getComputedStyle(element)
      const bounds = element.getBoundingClientRect()
      return bounds.width > 0 && bounds.height > 0
        ? { tag: element.tagName.toLowerCase(), outlineWidth: style.outlineWidth, outlineStyle: style.outlineStyle, boxShadow: style.boxShadow }
        : null
    })
    if (sample) focusSamples.push(sample)
  }
  expect(focusSamples.length).toBeGreaterThan(0)
  expect(focusSamples.some((sample) => (
    (sample.outlineStyle !== 'none' && sample.outlineWidth !== '0px') || sample.boxShadow !== 'none'
  ))).toBeTruthy()
})
