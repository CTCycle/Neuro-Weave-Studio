import { Component, computed, input, output } from '@angular/core'
import type { CompilationImplementation, CompilationObject, CompilationOption, ParameterDefinition } from '../../domain/types'
import { ParameterEditorComponent, type ParameterEditorState } from '../builder/components/parameter-editor.component'

@Component({
  selector: 'nw-compilation-option-editor',
  standalone: true,
  imports: [ParameterEditorComponent],
  template: `
    <div class="compilation-option-editor">
      @for (parameter of option().parameters; track parameter.id) {
        <label class="field compilation-parameter-field">
          <span>{{ parameter.display_name }}<small>{{ parameter.description }}</small>@if (mappingName(parameter); as name) { <code class="parameter-mapping">{{ name }}</code> }</span>
          <nw-parameter-editor [definition]="parameter" [value]="parameterValue(parameter)" [state]="parameterState()" (valueChange)="updateParameter(parameter, $event)" />
        </label>
      } @empty {
        <p class="compilation-description">This option uses its framework defaults.</p>
      }
    </div>
  `,
  styles: [`
    .compilation-option-editor { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
    .compilation-parameter-field { min-width:0; }
    .compilation-parameter-field > span { display:grid; gap:3px; }
    .compilation-parameter-field small { min-height:0; }
    .parameter-mapping { color:var(--cyan); font:500 8px/1.2 'IBM Plex Mono', Consolas, monospace; overflow-wrap:anywhere; }
    @media (max-width:760px) { .compilation-option-editor { grid-template-columns:1fr; } }
  `],
})
export class CompilationOptionEditorComponent {
  readonly option = input.required<CompilationOption>()
  readonly value = input.required<CompilationObject>()
  readonly backendMapping = input<CompilationImplementation | null>(null)
  readonly state = input<ParameterEditorState>('editable')
  readonly valueChange = output<CompilationObject>()
  readonly parameterState = computed<ParameterEditorState>(() => this.backendMapping()?.supported === false ? 'backend-inapplicable' : this.state())

  parameterValue(parameter: ParameterDefinition): unknown {
    return Object.prototype.hasOwnProperty.call(this.value().parameters, parameter.id)
      ? this.value().parameters[parameter.id]
      : parameter.default
  }

  mappingName(parameter: ParameterDefinition): string | null {
    return this.backendMapping()?.parameter_mapping?.[parameter.id] ?? null
  }

  updateParameter(parameter: ParameterDefinition, value: unknown): void {
    this.valueChange.emit({
      ...this.value(),
      parameters: { ...this.value().parameters, [parameter.id]: value },
    })
  }
}
