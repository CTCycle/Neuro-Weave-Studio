import { Component, HostListener, computed, effect, inject, input, output, signal, viewChild, type ElementRef } from '@angular/core'
import { ReactiveFormsModule, UntypedFormControl, UntypedFormGroup } from '@angular/forms'
import { LucideTrash2, LucideX } from '@lucide/angular'
import type { ExecutionBackend, NodeManifest, ParameterCapability, ParameterDefinition, PortDefinition } from '../../../domain/types'
import { implementationAvailable, implementationKey, implementationLabel } from '../../../domain/types'
import { displayParameterValue, helpfulParameterDescription, isFiniteParameter, parameterOptions, parseParameterValue, type ParameterDraftValue } from '../graph/parameter-value'
import { GraphEditorStore } from '../state/graph-editor.store'

@Component({
  selector: 'nw-properties-inspector',
  standalone: true,
  imports: [ReactiveFormsModule, LucideTrash2, LucideX],
  template: `
    <aside class="builder-inspector panel" [attr.role]="mobileOpen() ? 'dialog' : 'complementary'" [attr.aria-modal]="mobileOpen() ? 'true' : null" aria-label="Properties inspector" (click)="$event.stopPropagation()">
      @if (selectedNode(); as node) {
        @let status = nodeStatus(node);
        <header class="inspector-node-header">
          <div class="inspector-node-header-copy"><span class="eyebrow">{{ node.data.manifest.category }} · NODE</span><h2>{{ node.data.label }}</h2><code>{{ node.data.manifest.id }}</code></div>
          <div class="inspector-node-header-actions"><span class="inspector-status" [class.is-warning]="status.tone === 'warning'" [class.is-error]="status.tone === 'error'">{{ status.label }}</span><button #closeButton type="button" class="panel-close" (click)="close.emit()" aria-label="Close properties inspector"><svg lucideX [size]="14"></svg></button></div>
        </header>
        <form [formGroup]="form" (submit)="$event.preventDefault()">
          <label class="field-label" for="node-label">Label<input id="node-label" formControlName="label" (input)="updateLabel(node.id, $any($event.target).value)" /></label>
          <p class="inspector-description">{{ node.data.manifest.description }}</p>
          @for (group of groupedParameters(); track group[0]) {
            <details class="parameter-group" open><summary>{{ group[0] }}<span>{{ group[1].length }}</span></summary>
              @for (definition of group[1]; track definition.id) {
                @if (isVisible(definition, node)) {
                  @let capability = capabilityFor(node.data.manifest, definition);
                  @let help = helpfulParameterDescription(definition);
                  @let availability = availabilityMessage(node.data.manifest, definition, capability);
                  @if (isBinaryCheckbox(definition)) {
                    <label class="toggle-field inspector-field" [class.is-unavailable]="isDisabled(capability, definition, node, node.data.manifest)"><span>{{ definition.display_name }}@if (help) { <small>{{ help }}</small> }@if (availability) { <small class="parameter-availability">{{ availability }}</small> }</span><span class="compact-toggle"><input type="checkbox" [formControlName]="definition.id" [attr.aria-label]="definition.display_name" (change)="updateParameter(definition, $any($event.target).checked)" /><strong>{{ binaryStateLabel(node, definition) }}</strong></span></label>
                  } @else if (isChoice(definition)) {
                    <label class="field inspector-field" [class.is-unavailable]="isDisabled(capability, definition, node, node.data.manifest)"><span>{{ definition.display_name }}@if (help) { <small>{{ help }}</small> }@if (availability) { <small class="parameter-availability">{{ availability }}</small> }</span><select [formControlName]="definition.id" [attr.aria-invalid]="invalid()[definition.id] ? 'true' : null" (change)="updateParameter(definition, $any($event.target).value)">@if (definition.nullable) { <option value="">None</option> }@if (hasUnsupportedChoiceValue(definition, node, capability)) { <option [value]="currentParameterDisplay(node, definition)" disabled>Unavailable: {{ currentParameterDisplay(node, definition) }}</option> }@for (option of optionsFor(definition, capability); track $index) { <option [value]="stringValue(option)">{{ optionLabel(definition, option) }}</option> }</select>@if (invalid()[definition.id]) { <em class="field-error">Choose a supported option.</em> }</label>
                  } @else if (isStructured(definition)) {
                    <label class="field inspector-field" [class.is-unavailable]="isDisabled(capability, definition, node, node.data.manifest)"><span>{{ definition.display_name }}@if (help) { <small>{{ help }}</small> }@if (availability) { <small class="parameter-availability">{{ availability }}</small> }</span><textarea [formControlName]="definition.id" rows="2" [attr.aria-invalid]="invalid()[definition.id] ? 'true' : null" (input)="updateParameter(definition, $any($event.target).value)" [attr.placeholder]="structuredPlaceholder(definition)"></textarea>@if (invalid()[definition.id]) { <em class="field-error">Enter valid JSON or a supported value.</em> }</label>
                  } @else {
                    <label class="field inspector-field" [class.is-unavailable]="isDisabled(capability, definition, node, node.data.manifest)"><span>{{ definition.display_name }}@if (help) { <small>{{ help }}</small> }@if (availability) { <small class="parameter-availability">{{ availability }}</small> }</span><input [type]="inputType(definition)" [formControlName]="definition.id" [attr.min]="bounds(definition).min" [attr.max]="bounds(definition).max" [attr.step]="bounds(definition).step" [attr.aria-invalid]="invalid()[definition.id] ? 'true' : null" (input)="updateParameter(definition, $any($event.target).value)" />@if (invalid()[definition.id]) { <em class="field-error">Enter a complete value within the supported range.</em> }</label>
                  }
                }
              }
            </details>
          }
          <section class="inspector-section port-spec-section" aria-labelledby="port-specifications-title">
            <div class="inspector-section-heading"><span class="eyebrow" id="port-specifications-title">PORTS</span><span class="port-count">{{ node.data.manifest.inputs.length + node.data.manifest.outputs.length }}</span></div>
            <div class="port-spec-list">
              @for (port of node.data.manifest.inputs; track port.id) {
                <div class="port-spec-row input" [class.is-missing]="portStatus(node, 'input', port) === 'Missing required input'" [class.is-connected]="portConnectionCount(node.id, 'input', port.id) > 0" [class.is-disabled]="portStatus(node, 'input', port) === 'Disabled'" [title]="port.description">
                  <span class="port-direction input">IN</span><div class="port-spec-main"><strong>{{ port.display_name }}</strong><small>{{ portKindLabel(port) }} · {{ portShapeLabel(node, port, 'input') }}</small></div><div class="port-spec-meta"><span class="port-badge">{{ port.required ? 'REQ' : 'OPT' }}</span><span class="port-badge">{{ multiplicityLabel(port, 'input') }}</span><span class="port-status">{{ portStatus(node, 'input', port) }}</span></div>
                </div>
              }
              @for (port of node.data.manifest.outputs; track port.id) {
                <div class="port-spec-row output" [class.is-connected]="portConnectionCount(node.id, 'output', port.id) > 0" [class.is-disabled]="portStatus(node, 'output', port) === 'Disabled'" [title]="port.description">
                  <span class="port-direction output">OUT</span><div class="port-spec-main"><strong>{{ port.display_name }}</strong><small>{{ portKindLabel(port) }} · {{ portShapeLabel(node, port, 'output') }}</small></div><div class="port-spec-meta"><span class="port-badge">{{ port.required ? 'REQ' : 'OPT' }}</span><span class="port-badge">{{ multiplicityLabel(port, 'output') }}</span><span class="port-status">{{ portStatus(node, 'output', port) }}</span></div>
                </div>
              }
            </div>
          </section>
        </form>
      } @else if (selectedEdge(); as edge) {
        <header class="inspector-node-header edge"><div class="inspector-node-header-copy"><span class="eyebrow">INSPECTOR · CONNECTION</span><h2>Connection</h2><code>{{ edge.id }}</code></div><button #closeButton type="button" class="panel-close" (click)="close.emit()" aria-label="Close properties inspector"><svg lucideX [size]="14"></svg></button></header>
        <section class="inspector-section inspector-connection-section" aria-labelledby="connection-route-title"><div class="inspector-section-heading"><span class="eyebrow" id="connection-route-title">ROUTE</span></div><div class="inspector-connection-route"><div class="connection-route-port output"><span class="port-direction output">OUT</span><strong>{{ nodeLabel(edge.source) }}</strong><small>{{ portLabel(edge.source, 'output', edge.sourceHandle || 'output') }}</small></div><span class="inspector-connection-arrow" aria-hidden="true">→</span><div class="connection-route-port input"><span class="port-direction input">IN</span><strong>{{ nodeLabel(edge.target) }}</strong><small>{{ portLabel(edge.target, 'input', edge.targetHandle || 'input') }}</small></div></div></section>
        <button class="secondary danger-button" type="button" aria-label="Delete connection" (click)="deleteEdge()"><svg lucideTrash2 [size]="14"></svg>Delete connection</button>
      } @else {
        <div class="inspector-heading"><div><span class="eyebrow">INSPECTOR</span><h2>Properties</h2></div><button #closeButton type="button" class="panel-close" (click)="close.emit()" aria-label="Close properties inspector"><svg lucideX [size]="14"></svg></button></div><div class="inspector-empty"><span class="eyebrow">SELECTION</span><h2>No node selected</h2><p>Select a node to edit its label and parameters.</p></div>
      }
    </aside>
  `,
})
export class PropertiesInspectorComponent {
  readonly store = inject(GraphEditorStore)
  readonly executionBackend = input<ExecutionBackend>('pytorch')
  readonly kerasEnabled = input(true)
  readonly mobileOpen = input(false)
  readonly close = output<void>()
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
  private readonly closeButton = viewChild<ElementRef<HTMLButtonElement>>('closeButton')

  constructor() {
    effect(() => {
      if (this.mobileOpen()) queueMicrotask(() => this.closeButton()?.nativeElement.focus())
    })
    effect(() => {
      const node = this.selectedNode()
      for (const id of this.configuredControlIds) this.form.removeControl(id)
      this.configuredControlIds.clear()
      this.form.controls['label'].setValue(node?.data.label ?? '', { emitEvent: false })
      if (!node) { this.invalid.set({}); return }
      for (const definition of node.data.manifest.parameters) {
        const value = this.parameterValue(node, definition)
        const controlValue = this.isBinaryCheckbox(definition) ? Boolean(value) : displayParameterValue(value)
        const control = new UntypedFormControl(controlValue)
        if (this.isDisabled(this.capabilityFor(node.data.manifest, definition), definition, node, node.data.manifest)) control.disable({ emitEvent: false })
        this.form.addControl(definition.id, control)
        this.configuredControlIds.add(definition.id)
      }
      this.invalid.set({})
    })
  }

  @HostListener('keydown', ['$event'])
  keepFocusInside(event: KeyboardEvent): void {
    if (!this.mobileOpen() || event.key !== 'Tab') return
    const focusable = [...this.closeButton()?.nativeElement.closest('aside')?.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href]') ?? []].filter((element) => element.offsetParent !== null)
    if (!focusable.length) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
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
  stringValue(value: unknown): string { return displayParameterValue(value) }
  optionLabel(definition: ParameterDefinition, value: unknown): string { return definition.type === 'boolean' ? value === true ? 'Enabled' : 'Disabled' : value === null ? 'None' : String(value) }

  isDisabled(capability: ParameterCapability | undefined, definition?: ParameterDefinition, node?: { data: { parameters: Record<string, unknown> } }, manifest?: NodeManifest): boolean {
    return !implementationAvailable(manifest, this.executionBackend(), this.kerasEnabled())
      || capability?.supported === false
      || capability?.status === 'unsupported'
      || Boolean(capability?.allowed_values && capability.allowed_values.length === 1)
      || Boolean(definition?.enabled_when && node && !conditionMatches(definition.enabled_when, node.data.parameters))
  }

  optionsFor(definition: ParameterDefinition, capability?: ParameterCapability): unknown[] {
    return parameterOptions(definition).filter((option) => !capability?.allowed_values || capability.allowed_values.some((allowed) => sameValue(allowed, option)))
  }

  hasUnsupportedChoiceValue(definition: ParameterDefinition, node: { data: { parameters: Record<string, unknown> } }, capability?: ParameterCapability): boolean {
    const value = this.parameterValue(node, definition)
    if (value === undefined || value === null || (definition.nullable && value === '')) return false
    return !this.optionsFor(definition, capability).some((option) => String(option) === String(value))
  }

  currentParameterDisplay(node: { data: { parameters: Record<string, unknown> } }, definition: ParameterDefinition): string { return displayParameterValue(this.parameterValue(node, definition)) }
  inputType(definition: ParameterDefinition): string { return definition.type === 'number' || definition.type === 'integer' || definition.type === 'axis' ? 'number' : 'text' }
  isBinaryCheckbox(definition: ParameterDefinition): boolean { return definition.type === 'boolean' && !definition.nullable }
  binaryStateLabel(node: { data: { parameters: Record<string, unknown> } }, definition: ParameterDefinition): string { return this.parameterValue(node, definition) === true ? 'On' : 'Off' }
  isChoice(definition: ParameterDefinition): boolean { return isFiniteParameter(definition) }
  isStructured(definition: ParameterDefinition): boolean { return definition.type === 'shape' || definition.type === 'list' || definition.type === 'tuple' || definition.type === 'object' }
  structuredPlaceholder(definition: ParameterDefinition): string { return definition.type === 'shape' || definition.type === 'list' || definition.type === 'tuple' ? '[ … ]' : '{ "class_name": "…", "config": {} }' }
  isVisible(definition: ParameterDefinition, node: { data: { parameters: Record<string, unknown> } }): boolean { return !definition.visible_when || conditionMatches(definition.visible_when, node.data.parameters) }

  availabilityMessage(manifest: NodeManifest, definition: ParameterDefinition, capability?: ParameterCapability): string | null {
    const mapping = manifest.implementations[implementationKey(this.executionBackend(), this.kerasEnabled())]
    if (!mapping || !implementationAvailable(manifest, this.executionBackend(), this.kerasEnabled())) return `Unavailable for ${implementationLabel(this.executionBackend(), this.kerasEnabled())}`
    if (capability?.supported === false || capability?.status === 'unsupported') return `Unavailable for ${this.kerasEnabled() ? 'Keras' : this.executionBackend()}`
    if (capability?.status === 'unverified') return 'Backend mapping is not verified'
    if (capability?.allowed_values?.length === 1) return `Fixed to ${String(capability.allowed_values[0])} by this backend`
    return null
  }

  nodeStatus(node: { id: string; data: { manifest: NodeManifest; parameters: Record<string, unknown> } }): { label: string; tone: 'ready' | 'warning' | 'error' } {
    if (!implementationAvailable(node.data.manifest, this.executionBackend(), this.kerasEnabled())) return { label: 'Unavailable', tone: 'error' }
    if (this.invalidParameters()) return { label: 'Check parameters', tone: 'warning' }
    const missing = node.data.manifest.inputs.some((port) => port.required && this.isPortEnabled(node, port) && this.portConnectionCount(node.id, 'input', port.id) === 0)
    return missing ? { label: 'Needs input', tone: 'warning' } : { label: 'Ready', tone: 'ready' }
  }

  portConnectionCount(nodeId: string, direction: 'input' | 'output', portId: string): number {
    return this.store.edges().filter((edge) => direction === 'input'
      ? edge.target === nodeId && (edge.targetHandle ?? 'input') === portId
      : edge.source === nodeId && (edge.sourceHandle ?? 'output') === portId).length
  }

  portStatus(node: { id: string; data: { parameters: Record<string, unknown> } }, direction: 'input' | 'output', port: PortDefinition): string {
    if (!this.isPortEnabled(node, port)) return 'Disabled'
    const count = this.portConnectionCount(node.id, direction, port.id)
    if (count > 1) return `${count} connections`
    if (count === 1) return 'Connected'
    if (direction === 'input' && port.required) return 'Missing required input'
    return 'Not connected'
  }

  portKindLabel(port: PortDefinition): string { return port.kind === 'tensor_list' ? `tensor list · ${port.element_kind ?? 'tensor'}` : port.kind }
  portShapeLabel(node: { data: { shapeSummary?: { outputs: Record<string, { shape: Array<number | null | { name: string }>; dtype: string }> } } }, port: PortDefinition, direction: 'input' | 'output'): string {
    const summary = direction === 'output' ? node.data.shapeSummary?.outputs[port.id] : undefined
    if (summary) return `${summary.shape.map((value) => value === null ? '?' : typeof value === 'object' ? value.name : value).join(' × ')} · ${summary.dtype}`
    if (port.rank !== undefined && port.rank !== null) return `rank ${port.rank}`
    return port.kind === 'scalar' || port.kind === 'mask' ? 'value' : 'shape unknown'
  }
  multiplicityLabel(port: PortDefinition, direction: 'input' | 'output'): string { return direction === 'output' ? 'fan-out' : port.accepts_multiple ? 'many' : '1' }
  isPortEnabled(node: { data: { parameters: Record<string, unknown> } }, port: PortDefinition): boolean { return conditionMatches(port.enabled_when, node.data.parameters) }

  private parameterValue(node: { data: { parameters: Record<string, unknown> } }, definition: ParameterDefinition): unknown {
    if (Object.prototype.hasOwnProperty.call(node.data.parameters, definition.id)) return node.data.parameters[definition.id]
    const capability = this.capabilityFor(this.selectedNode()?.data.manifest, definition)
    if (capability?.default !== undefined && capability.default !== null) return capability.default
    return definition.editor_default !== undefined ? definition.editor_default : definition.default
  }

  private invalidParameters(): boolean { return Object.values(this.invalid()).some(Boolean) }

  bounds(definition: ParameterDefinition): { min?: number; max?: number; step?: number } {
    return {
      min: typeof definition.constraints['min'] === 'number' ? definition.constraints['min'] as number : undefined,
      max: typeof definition.constraints['max'] === 'number' ? definition.constraints['max'] as number : undefined,
      step: typeof definition.constraints['step'] === 'number' ? definition.constraints['step'] as number : definition.type === 'integer' || definition.type === 'axis' ? 1 : undefined,
    }
  }

  deleteNode(): void { this.store.deleteSelected() }
  deleteEdge(): void { this.store.deleteSelected() }
  nodeLabel(nodeId: string): string { return this.store.nodes().find((node) => node.id === nodeId)?.data.label ?? nodeId }
  portLabel(nodeId: string, direction: 'input' | 'output', portId: string): string {
    const node = this.store.nodes().find((item) => item.id === nodeId)
    const port = (direction === 'input' ? node?.data.manifest.inputs : node?.data.manifest.outputs)?.find((item) => item.id === portId)
    return port ? `${port.display_name} · ${this.portKindLabel(port)}` : portId
  }
}

function conditionMatches(condition: Record<string, unknown> | undefined, parameters: Record<string, unknown>): boolean {
  if (!condition) return true
  return Object.entries(condition).every(([parameterId, expected]) => {
    const actual = parameters[parameterId]
    if (typeof expected === 'object' && expected !== null && !Array.isArray(expected)) {
      const rule = expected as Record<string, unknown>
      if ('equals' in rule && !sameValue(actual, rule['equals'])) return false
      if ('not_equals' in rule && sameValue(actual, rule['not_equals'])) return false
      if ('in' in rule && Array.isArray(rule['in']) && !rule['in'].some((item) => sameValue(actual, item))) return false
      if ('not_in' in rule && Array.isArray(rule['not_in']) && rule['not_in'].some((item) => sameValue(actual, item))) return false
      return true
    }
    if (Array.isArray(expected)) return expected.some((item) => sameValue(actual, item))
    return sameValue(actual, expected)
  })
}

function sameValue(left: unknown, right: unknown): boolean { return JSON.stringify(left) === JSON.stringify(right) }
