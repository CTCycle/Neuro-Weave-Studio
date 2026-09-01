import { ChangeDetectionStrategy, Component, effect, input, viewChild, type ElementRef } from '@angular/core'
import katex from 'katex'

@Component({
  selector: 'nw-math-formula',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: '<span #formulaElement class="math-formula" aria-label="Rendered mathematical formula"></span>',
})
export class MathFormulaComponent {
  readonly formula = input.required<string>()
  private readonly formulaElement = viewChild<ElementRef<HTMLElement>>('formulaElement')

  constructor() {
    const formulaElement = this.formulaElement
    const formula = this.formula
    // The view-child signal makes this safe for both the initial render and
    // later catalog updates without binding trusted HTML into the template.
    effect(() => {
      const element = formulaElement()?.nativeElement
      if (!element) return
      katex.render(formula(), element, { displayMode: false, output: 'html', throwOnError: false })
    })
  }
}
