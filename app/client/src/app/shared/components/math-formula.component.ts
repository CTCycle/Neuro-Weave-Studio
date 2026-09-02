import { ChangeDetectionStrategy, Component, DestroyRef, effect, inject, input, viewChild, type ElementRef } from '@angular/core'
import katex from 'katex'

@Component({
  selector: 'nw-math-formula',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: '<span #formulaElement class="math-formula" aria-label="Rendered mathematical formula"></span>',
  styles: [`
    :host { display:block; min-width:0; max-width:100%; }
    .math-formula { display:block; width:100%; max-width:100%; overflow:visible; padding:6px 0; color:var(--cyan); font-size:12px; line-height:1.35; white-space:nowrap; }
    .math-formula :where(.katex, .katex-html) { max-width:100%; }
  `],
})
export class MathFormulaComponent {
  readonly formula = input.required<string>()
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
    element.style.fontSize = '12px'
    const available = element.clientWidth
    const required = element.scrollWidth
    if (!available || required <= available) return
    const fitted = Math.max(7.5, Math.min(12, 12 * available / required))
    element.style.fontSize = `${fitted}px`
  }
}
