import { Component, computed, forwardRef, input, output, signal } from '@angular/core'
import type { ParameterCapability, ParameterDefinition } from '../../../domain/types'
import { displayParameterValue, isFiniteParameter, parameterOptions, parseParameterValue, type ParameterDraftValue } from '../graph/parameter-value'
import { StructuredParameterEditorComponent } from './structured-parameter-editor.component'

export type ParameterEditorState = 'editable' | 'fixed' | 'backend-inapplicable' | 'invalid'

@Component({
  selector: 'nw-parameter-editor',
  standalone: true,
  imports: [forwardRef(() => StructuredParameterEditorComponent)],
  template: `
    <div class="parameter-editor" [class.is-disabled]="readOnly()" [class.is-invalid]="invalid()">
      <span class="parameter-state" [class.is-fixed]="state() === 'fixed'" [class.is-unavailable]="state() === 'backend-inapplicable'" [class.is-invalid]="state() === 'invalid' || invalid()">{{ stateLabel() }}</span>
      @if (isCheckbox()) {
        <label class="parameter-checkbox"><input type="checkbox" [checked]="currentValue() === true" [disabled]="readOnly()" (change)="commit(($any($event.target)).checked)" /><span>{{ currentValue() === true ? 'Enabled' : 'Disabled' }}</span></label>
      } @else if (isChoice()) {
        <select [value]="displayValue()" [disabled]="readOnly()" [attr.aria-invalid]="invalid() ? 'true' : null" (change)="commit(($any($event.target)).value)">
          @if (definition().nullable) { <option value="">None</option> }
          @for (option of options(); track $index) { <option [value]="optionValue(option)">{{ optionLabel(option) }}</option> }
        </select>
      } @else if (isStructured()) {
        <nw-structured-parameter-editor [definition]="definition()" [value]="currentValue()" [capability]="capability()" [disabled]="readOnly()" [state]="state()" (valueChange)="commit($event)" (validChange)="invalid.set(!$event)" />
      } @else if (isMultiline()) {
        <textarea [value]="displayValue()" [disabled]="readOnly()" rows="2" [attr.aria-invalid]="invalid() ? 'true' : null" (input)="commit(($any($event.target)).value)"></textarea>
      } @else {
        <input [type]="inputType()" [value]="displayValue()" [disabled]="readOnly()" [attr.min]="bounds().min" [attr.max]="bounds().max" [attr.step]="bounds().step" [attr.aria-invalid]="invalid() ? 'true' : null" (input)="commit(($any($event.target)).value)" />
      }
      @if (invalid()) { <small class="parameter-error">Enter a valid {{ definition().display_name.toLowerCase() }}.</small> }
    </div>
  `,
  styles: [`
    :host { display:block; min-width:0; }
    .parameter-editor { display:grid; gap:5px; min-width:0; }
    .parameter-editor > input, .parameter-editor > select, .parameter-editor > textarea { width:100%; min-width:0; }
    .parameter-checkbox { display:flex; align-items:center; gap:7px; color:var(--text); font-size:10px; }
    .parameter-checkbox input { accent-color:var(--accent-bright); }
    .parameter-state { color:var(--accent-bright); font:500 8px/1.2 'IBM Plex Mono', Consolas, monospace; letter-spacing:.05em; text-transform:uppercase; }
    .parameter-state.is-fixed { color:var(--muted); }
    .parameter-state.is-unavailable, .parameter-state.is-invalid, .parameter-error { color:var(--error); }
    .parameter-error { line-height:1.35; }
    .is-disabled { opacity:.75; }
  `],
})
export class ParameterEditorComponent {
  readonly definition = input.required<ParameterDefinition>()
  readonly value = input<unknown>()
  readonly capability = input<ParameterCapability | null>(null)
  readonly disabled = input(false)
  readonly state = input<ParameterEditorState>('editable')
  readonly valueChange = output<unknown>()
  readonly validChange = output<boolean>()
  readonly invalid = signal(false)
  readonly currentValue = computed(() => this.value() === undefined ? this.definition().default : this.value())
  readonly options = computed(() => {
    const capability = this.capability()
    return parameterOptions(this.definition()).filter((option) => !capability?.allowed_values || capability.allowed_values.some((allowed) => JSON.stringify(allowed) === JSON.stringify(option)))
  })
  readonly readOnly = computed(() => this.disabled() || this.state() !== 'editable')

  isCheckbox(): boolean { return this.definition().type === 'boolean' && !this.definition().nullable }
  isChoice(): boolean { return isFiniteParameter(this.definition()) }
  isStructured(): boolean {
    const definition = this.definition()
    return definition.type === 'shape' || definition.type === 'list' || definition.type === 'tuple' || definition.type === 'object'
      || ((definition.type === 'initializer' || definition.type === 'regularizer' || definition.type === 'constraint') && typeof this.currentValue() === 'object' && this.currentValue() !== null)
  }
  isMultiline(): boolean { return this.definition().type === 'expression' }
  inputType(): 'number' | 'text' { return this.definition().type === 'number' || this.definition().type === 'integer' ? 'number' : 'text' }
  displayValue(): string { return displayParameterValue(this.currentValue()) }
  optionValue(value: unknown): string { return displayParameterValue(value) }
  optionLabel(value: unknown): string { return value === null ? 'None' : typeof value === 'boolean' ? value ? 'Enabled' : 'Disabled' : String(value) }
  stateLabel(): string { return this.invalid() || this.state() === 'invalid' ? 'Invalid' : this.state().replaceAll('-', ' ') }
  bounds(): { min?: number; max?: number; step?: number } {
    const constraints = this.definition().constraints
    return {
      min: typeof constraints['min'] === 'number' ? constraints['min'] : undefined,
      max: typeof constraints['max'] === 'number' ? constraints['max'] : undefined,
      step: typeof constraints['step'] === 'number' ? constraints['step'] : this.definition().type === 'integer' ? 1 : undefined,
    }
  }

  commit(raw: unknown): void {
    const parsed = parseParameterValue(this.definition(), raw as ParameterDraftValue, this.capability() ?? undefined)
    this.invalid.set(!parsed.valid)
    this.validChange.emit(parsed.valid)
    if (parsed.valid) this.valueChange.emit(parsed.value)
  }
}
