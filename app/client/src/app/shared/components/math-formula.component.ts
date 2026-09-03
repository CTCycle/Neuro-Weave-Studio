import { ChangeDetectionStrategy, Component, DestroyRef, effect, inject, input, viewChild, type ElementRef } from '@angular/core'
import katex from 'katex'

const MAX_FORMULA_FONT_SIZE = 12
const MIN_FORMULA_FONT_SIZE = 7.5
const MAX_PROMINENT_FORMULA_FONT_SIZE = 16
const MIN_PROMINENT_FORMULA_FONT_SIZE = 9

export function fittedFormulaFontSize(
  availableWidth: number,
  requiredWidth: number,
  maxFontSize = MAX_FORMULA_FONT_SIZE,
  minFontSize = MIN_FORMULA_FONT_SIZE,
): number {
  if (!availableWidth || requiredWidth <= availableWidth) return maxFontSize
  const safeWidth = Math.max(0, availableWidth - 2)
  return Math.max(minFontSize, Math.min(maxFontSize, maxFontSize * safeWidth / requiredWidth))
}

@Component({
  selector: 'nw-math-formula',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: '<span #formulaElement class="math-formula" [class.prominent]="variant() === \'prominent\'" aria-label="Rendered mathematical formula"></span>',
  styles: [`
    :host { display:block; min-width:0; max-width:100%; }
    .math-formula { display:block; width:100%; max-width:100%; overflow:visible; padding:6px 0; color:var(--cyan); font-size:12px; line-height:1.35; white-space:nowrap; }
    .math-formula.prominent { padding:8px 0 7px; font-size:16px; line-height:1.4; }
    .math-formula :where(.katex, .katex-html) { max-width:100%; }
  `],
})
export class MathFormulaComponent {
  readonly formula = input.required<string>()
  readonly variant = input<'compact' | 'prominent'>('compact')
  private readonly formulaElement = viewChild<ElementRef<HTMLElement>>('formulaElement')
  private readonly destroyRef = inject(DestroyRef)
  private resizeObserver?: ResizeObserver

  constructor() {
    const formulaElement = this.formulaElement
    const formula = this.formula
    effect(() => {
      const element = formulaElement()?.nativeElement
      if (!element) return
      katex.render(formula(), element, { displayMode: false, output: 'html', throwOnError: false })
      this.observeAndFit(element)
    })
    this.destroyRef.onDestroy(() => this.resizeObserver?.disconnect())
  }

  private observeAndFit(element: HTMLElement): void {
    this.resizeObserver?.disconnect()
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.fitFormula(element))
      this.resizeObserver.observe(element)
    }
    queueMicrotask(() => this.fitFormula(element))
  }

  private fitFormula(element: HTMLElement): void {
    const prominent = this.variant() === 'prominent'
    const maxFontSize = prominent ? MAX_PROMINENT_FORMULA_FONT_SIZE : MAX_FORMULA_FONT_SIZE
    const minFontSize = prominent ? MIN_PROMINENT_FORMULA_FONT_SIZE : MIN_FORMULA_FONT_SIZE
    element.style.fontSize = `${maxFontSize}px`
    const available = element.clientWidth
    const required = element.scrollWidth
    if (!available || required <= available) return
    const fitted = fittedFormulaFontSize(available, required, maxFontSize, minFontSize)
    element.style.fontSize = `${fitted}px`
  }
}
