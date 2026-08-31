import { Component, computed, effect, inject, input, signal } from '@angular/core'
import { ReactiveFormsModule, UntypedFormControl, UntypedFormGroup } from '@angular/forms'
import { LucideTrash2 } from '@lucide/angular'
import type { ExecutionBackend, NodeManifest, ParameterCapability, ParameterDefinition } from '../../../domain/types'
import { implementationKey } from '../../../domain/types'
import { parseParameterValue, displayParameterValue, helpfulParameterDescription, type ParameterDraftValue } from '../graph/parameter-value'
import { GraphEditorStore } from '../state/graph-editor.store'

@Component({
  selector: 'nw-properties-inspector',
  standalone: true,
  imports: [ReactiveFormsModule, LucideTrash2],
  template: `
    <aside class="builder-inspector panel" aria-label="Properties inspector">
      @if (selectedNode(); as node) {
        <div class="inspector-heading"><div><span class="eyebrow">INSPECTOR</span><h2>Properties</h2></div></div>
        <form [formGroup]="form" (submit)="$event.preventDefault()">
          <div class="inspector-node-heading"><div><span class="eyebrow">{{ node.data.manifest.category }}</span><strong>{{ node.data.label }}</strong><small>{{ node.data.manifest.id }}</small></div></div>
          <label class="field-label" for="node-label">Label<input id="node-label" formControlName="label" (input)="updateLabel(node.id, $any($event.target).value)" /></label>
          <p class="inspector-description">{{ node.data.manifest.description }}</p>
          @for (group of groupedParameters(); track group[0]) {
            <details class="parameter-group" open><summary>{{ group[0] }}<span>{{ group[1].length }}</span></summary>
              @for (definition of group[1]; track definition.id) {
                @let capability = capabilityFor(node.data.manifest, definition);
                @let help = helpfulParameterDescription(definition);
                @if (definition.type === 'boolean') {
                  <label class="toggle-field inspector-field"><span>{{ definition.display_name }}@if (help) { <small>{{ help }}</small> }</span><input type="checkbox" [formControlName]="definition.id" (change)="updateParameter(definition, $any($event.target).checked)" /></label>
                } @else if (definition.type === 'enum') {
                  <label class="field inspector-field"><span>{{ definition.display_name }}@if (help) { <small>{{ help }}</small> }</span><select [formControlName]="definition.id" [attr.aria-invalid]="invalid()[definition.id] ? 'true' : null" (change)="updateParameter(definition, $any($event.target).value)">@if (definition.nullable) { <option value="">None</option> }@for (option of optionsFor(definition, capability); track $index) { <option [value]="stringValue(option)">{{ option }}</option> }</select>@if (invalid()[definition.id]) { <em class="field-error">Choose a supported option.</em> }</label>
                } @else {
                  <label class="field inspector-field"><span>{{ definition.display_name }}@if (help) { <small>{{ help }}</small> }</span><input [type]="inputType(definition)" [formControlName]="definition.id" [attr.min]="bounds(definition).min" [attr.max]="bounds(definition).max" [attr.step]="bounds(definition).step" [attr.aria-invalid]="invalid()[definition.id] ? 'true' : null" (input)="updateParameter(definition, $any($event.target).value)" />@if (invalid()[definition.id]) { <em class="field-error">Enter a complete value within the supported range.</em> }</label>
                }
              }
            </details>
          }
          <div class="inspector-section"><span class="eyebrow">PORTS</span>@for (port of node.data.manifest.inputs; track port.id) { <div class="inspector-port"><span>Input</span><strong>{{ port.display_name }}</strong><small>{{ port.kind }}{{ port.required ? ' · required' : '' }}</small></div> }@for (port of node.data.manifest.outputs; track port.id) { <div class="inspector-port"><span>Output</span><strong>{{ port.display_name }}</strong><small>{{ port.kind }}</small></div> }</div>
          <button class="secondary danger-button" type="button" (click)="deleteNode()"><svg lucideTrash2 [size]="14"></svg>Delete node</button>
        </form>
      } @else if (selectedEdge(); as edge) {
        <div class="inspector-heading"><div><span class="eyebrow">INSPECTOR</span><h2>Connection</h2></div></div>
        <div class="inspector-edge-heading"><span class="eyebrow">SELECTED EDGE</span><strong>{{ edge.id }}</strong></div>
        <div class="inspector-section inspector-connection-section"><span class="eyebrow">ROUTE</span><div class="inspector-connection-route"><div class="inspector-port"><span>Source</span><strong>{{ nodeLabel(edge.source) }}</strong><small>{{ portLabel(edge.source, 'output', edge.sourceHandle || 'output') }}</small></div><span class="inspector-connection-arrow" aria-hidden="true">→</span><div class="inspector-port"><span>Target</span><strong>{{ nodeLabel(edge.target) }}</strong><small>{{ portLabel(edge.target, 'input', edge.targetHandle || 'input') }}</small></div></div></div>
        <button class="secondary danger-button" type="button" aria-label="Delete connection" (click)="deleteEdge()"><svg lucideTrash2 [size]="14"></svg>Delete connection</button>
      } @else { <div class="inspector-empty"><span class="eyebrow">INSPECTOR</span><h2>No node selected</h2><p>Select a node to edit its label and parameters.</p></div> }
    </aside>
  `,
})
export class PropertiesInspectorComponent {
  readonly store = inject(GraphEditorStore)
  readonly executionBackend = input<ExecutionBackend>('pytorch')
  readonly kerasEnabled = input(true)
  readonly selectedNode = computed(() => this.store.nodes().find((node) => node.id === this.store.selectedNodeId()) ?? null)
  readonly selectedEdge = computed(() => this.store.edges().find((edge) => edge.id === this.store.selectedEdgeId()) ?? null)
  readonly groupedParameters = computed(() => {
    const groups = new Map<string, ParameterDefinition[]>()
    for (const definition of this.selectedNode()?.data.manifest.parameters ?? []) {
      const group = definition.group || 'General'
      groups.set(group, [...(groups.get(group) ?? []), definition])
    }
    return [...groups.entries()]
  })
  readonly form = new UntypedFormGroup({ label: new UntypedFormControl('') })
  readonly invalid = signal<Record<string, boolean>>({})
  private configuredControlIds = new Set<string>()

  constructor() {
    effect(() => {
      const node = this.selectedNode()
      for (const id of this.configuredControlIds) this.form.removeControl(id)
      this.configuredControlIds.clear()
      this.form.controls['label'].setValue(node?.data.label ?? '', { emitEvent: false })
      if (!node) { this.invalid.set({}); return }
      for (const definition of node.data.manifest.parameters) {
        const value = definition.type === 'boolean'
          ? Boolean(node.data.parameters[definition.id] ?? definition.default)
          : displayParameterValue(node.data.parameters[definition.id] ?? definition.default)
        const control = new UntypedFormControl(value)
        if (this.isDisabled(this.capabilityFor(node.data.manifest, definition))) control.disable({ emitEvent: false })
        this.form.addControl(definition.id, control)
        this.configuredControlIds.add(definition.id)
      }
      this.invalid.set({})
    })
  }

  updateLabel(nodeId: string, label: string): void { this.store.updateNode(nodeId, { label }) }

  updateParameter(definition: ParameterDefinition, raw: ParameterDraftValue): void {
    const parsed = parseParameterValue(definition, raw, this.capabilityFor(this.selectedNode()?.data.manifest, definition))
    this.invalid.update((state) => ({ ...state, [definition.id]: !parsed.valid }))
    if (parsed.valid) {
      const node = this.selectedNode()
      if (!node) return
      const parameters = { ...node.data.parameters }
      if (parsed.value === undefined) delete parameters[definition.id]
      else parameters[definition.id] = parsed.value
      this.store.updateNode(node.id, { parameters })
    }
  }

  capabilityFor(manifest: NodeManifest | undefined, definition: ParameterDefinition): ParameterCapability | undefined {
    return manifest?.implementations[implementationKey(this.executionBackend(), this.kerasEnabled())]?.parameters?.[definition.id]
  }

  helpfulParameterDescription(definition: ParameterDefinition): string | null { return helpfulParameterDescription(definition) }
  stringValue(value: unknown): string { return String(value) }

  isDisabled(capability: ParameterCapability | undefined): boolean {
    return capability?.supported === false || Boolean(capability?.allowed_values && capability.allowed_values.length === 1)
  }

  optionsFor(definition: ParameterDefinition, capability?: ParameterCapability): unknown[] {
    return definition.options.filter((option) => !capability?.allowed_values || capability.allowed_values.some((allowed) => String(allowed) === String(option)))
  }

  inputType(definition: ParameterDefinition): string {
    return definition.type === 'number' || definition.type === 'integer' || definition.type === 'axis' ? 'number' : 'text'
  }

  bounds(definition: ParameterDefinition): { min?: number; max?: number; step?: number } {
    return {
      min: typeof definition.constraints['min'] === 'number' ? definition.constraints['min'] as number : undefined,
      max: typeof definition.constraints['max'] === 'number' ? definition.constraints['max'] as number : undefined,
      step: typeof definition.constraints['step'] === 'number' ? definition.constraints['step'] as number : definition.type === 'integer' || definition.type === 'axis' ? 1 : undefined,
    }
  }

  deleteNode(): void { this.store.deleteSelected() }
  deleteEdge(): void { this.store.deleteSelected() }

  nodeLabel(nodeId: string): string {
    return this.store.nodes().find((node) => node.id === nodeId)?.data.label ?? nodeId
  }

  portLabel(nodeId: string, direction: 'input' | 'output', portId: string): string {
    const node = this.store.nodes().find((item) => item.id === nodeId)
    const port = (direction === 'input' ? node?.data.manifest.inputs : node?.data.manifest.outputs)?.find((item) => item.id === portId)
    return port ? `${port.display_name} · ${port.kind}` : portId
  }
}
