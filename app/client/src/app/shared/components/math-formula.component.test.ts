import { describe, expect, it } from 'vitest'
import { fittedFormulaFontSize } from './math-formula.component'

describe('fittedFormulaFontSize', () => {
  it('keeps formulas at the normal size when they fit', () => {
    expect(fittedFormulaFontSize(200, 180)).toBe(12)
  })

  it('leaves an inset for intrinsic KaTeX rounding overflow', () => {
    const fitted = fittedFormulaFontSize(176, 178)

    expect(fitted).toBeLessThan(12)
    expect(fitted).toBeGreaterThanOrEqual(7.5)
  })

  it('does not shrink below the readable minimum', () => {
    expect(fittedFormulaFontSize(40, 1000)).toBe(7.5)
  })

  it('supports a larger readable presentation variant', () => {
    expect(fittedFormulaFontSize(240, 180, 16, 9)).toBe(16)
    expect(fittedFormulaFontSize(40, 1000, 16, 9)).toBe(9)
  })
})
