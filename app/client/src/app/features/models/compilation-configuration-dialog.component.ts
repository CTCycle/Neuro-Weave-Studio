import { Component, computed, effect, input, output, signal } from '@angular/core'
import type {
  CompilationCatalog,
  CompilationOption,
  CompilationParameter,
  ModelConfiguration,
  NativeCompilationCapability,
} from '../../domain/types'
import { emptyModelConfiguration } from '../../domain/types'
import { MathFormulaComponent } from '../../shared/components/math-formula.component'

type CompilationTab = 'optimizer' | 'objective' | 'metrics' | 'outputs' | 'advanced' | 'native-optimizer' | 'native-objective' | 'native-metrics' | 'native-outputs' | 'native'

interface CompilationTabDefinition {
  id: CompilationTab
  label: string
  detail: string
}

@Component({
  selector: 'nw-compilation-configuration-dialog',
  standalone: true,
  imports: [MathFormulaComponent],
  template: `
    <div class="dialog-backdrop compilation-modal-backdrop" role="presentation" (mousedown)="closeFromBackdrop($event)">
      <section class="compilation-modal panel" role="dialog" aria-modal="true" aria-labelledby="compilation-modal-title" (mousedown)="$event.stopPropagation()">
        <header class="compilation-modal-header">
          <div>
            <span class="eyebrow">COMPILATION CONFIGURATION</span>
            <h2 id="compilation-modal-title">{{ kerasEnabled() ? 'Configure Keras compilation' : 'Configure native ' + (nativeCapability()?.backend ?? 'backend') }}</h2>
            <p>{{ kerasEnabled() ? 'These controls are validated and serialized by the Keras catalogue.' : nativeCatalog() ? 'These controls configure native PyTorch helpers and torch.compile. Smoke checks remain inference-only.' : 'These controls map directly to the selected backend compiler.' }}</p>
          </div>
          <button type="button" class="panel-close" (click)="close.emit()" aria-label="Close compilation configuration">×</button>
        </header>

        <div class="compilation-modal-tabs" role="tablist" aria-label="Compilation configuration sections">
          @for (tab of tabs(); track tab.id) {
            <button type="button" role="tab" class="compilation-tab" [class.active]="activeTab() === tab.id" [attr.aria-selected]="activeTab() === tab.id" [attr.aria-controls]="panelId(tab.id)" (click)="activeTab.set(tab.id)">
              <span>{{ tab.label }}</span><small>{{ tab.detail }}</small>
            </button>
          }
        </div>

        <div class="compilation-modal-body">
          @if (kerasEnabled() && kerasCatalog(); as catalog) {
            @if (activeTab() === 'optimizer') {
              <section class="compilation-section compilation-tab-panel" role="tabpanel">
                <div class="section-heading"><div><span class="eyebrow">OPTIMIZER</span><h3>Update rule</h3></div></div>
                <label class="field"><span>Optimizer</span><select [value]="kerasOptimizerId()" (change)="chooseKerasOptimizer($event, catalog)">@for (item of catalog.optimizers; track item.id) { <option [value]="item.id">{{ item.label }}</option> }</select></label>
                @if (kerasOptimizer(); as optimizer) { <p class="compilation-description">{{ optimizer.description }}</p> }
                <div class="compilation-parameter-grid">
                  @for (parameter of kerasOptimizerParameters(); track parameter.id) {
                    <label class="field"><span>{{ parameter.label }}<small>{{ parameter.description }}</small></span>
                      @if (parameter.type === 'boolean') { <input type="checkbox" [checked]="booleanParameterValue(parameter, draft().keras.optimizer_parameters)" (change)="setKerasOptimizerParameter(parameter, $event)" /> }
                      @else { <input type="number" [value]="parameterValue(parameter, draft().keras.optimizer_parameters)" [min]="numericConstraint(parameter, 'min')" [max]="numericConstraint(parameter, 'max')" step="any" (change)="setKerasOptimizerParameter(parameter, $event)" /> }
                    </label>
                  }
                </div>
              </section>
            }
            @if (activeTab() === 'objective') {
              <section class="compilation-section compilation-tab-panel" role="tabpanel">
                <div class="section-heading"><div><span class="eyebrow">LOSS</span><h3>Default objective</h3></div></div>
                <label class="field"><span>Loss</span><select [value]="kerasLossId()" (change)="setKerasLoss($event)">@for (item of catalog.losses; track item.id) { <option [value]="item.id">{{ item.label }}</option> }</select></label>
                @if (kerasLoss(); as loss) { <p class="compilation-description">{{ loss.description }}</p>@if (loss.formula) { <nw-math-formula [formula]="loss.formula"></nw-math-formula> } }
              </section>
            }
            @if (activeTab() === 'metrics') {
              <section class="compilation-section compilation-tab-panel" role="tabpanel">
                <div class="section-heading"><div><span class="eyebrow">METRICS</span><h3>Default measurements</h3></div></div>
                <div class="compilation-option-grid">@for (metric of catalog.metrics; track metric.id) { <label class="compilation-option"><input type="checkbox" [checked]="selectedMetricIds().includes(metric.id)" (change)="toggleKerasMetric(metric.id, $event)" /><span><strong>{{ metric.label }}</strong><small>{{ metric.description }}</small>@if (metric.formula) { <nw-math-formula [formula]="metric.formula"></nw-math-formula> }</span></label> }</div>
              </section>
            }
            @if (activeTab() === 'outputs') {
              <section class="compilation-section compilation-tab-panel" role="tabpanel">
                <div class="section-heading"><div><span class="eyebrow">OUTPUT OVERRIDES</span><h3>Per-output loss</h3></div></div>
                @for (outputId of outputIds(); track outputId) { <label class="field"><span>{{ outputId }}</span><select [value]="outputLoss(draft().keras.losses, outputId)" (change)="setKerasOutputLoss(outputId, $event)"><option value="">Use default loss</option>@for (item of catalog.losses; track item.id) { <option [value]="item.id">{{ item.label }}</option> }</select></label> }
              </section>
            }
            @if (activeTab() === 'advanced') {
              <section class="compilation-section compilation-tab-panel" role="tabpanel">
                <div class="section-heading"><div><span class="eyebrow">ADVANCED OPTIONS</span><h3>Model defaults</h3></div><small class="compilation-description">Initializers, regularization, precision</small></div>
                <div class="compilation-parameter-grid">
                  <label class="field"><span>Initializer</span><select [value]="draft().keras.default_initializer" (change)="setKerasText('default_initializer', $event)">@for (item of catalog.initializers; track item) { <option [value]="item">{{ item }}</option> }</select></label>
                  <label class="field"><span>Regularizer</span><select [value]="draft().keras.default_regularizer" (change)="setKerasText('default_regularizer', $event)">@for (item of catalog.regularizers; track item) { <option [value]="item">{{ item }}</option> }</select></label>
                  <label class="field"><span>Constraint</span><select [value]="draft().keras.default_constraint" (change)="setKerasText('default_constraint', $event)">@for (item of catalog.constraints; track item) { <option [value]="item">{{ item }}</option> }</select></label>
                  <label class="toggle-field"><span>Mixed precision</span><input type="checkbox" [checked]="draft().keras.mixed_precision" (change)="setKerasMixedPrecision($event)" /></label>
                </div>
              </section>
            }
          }

          @if (!kerasEnabled() && nativeCatalog(); as catalog) {
            @if (activeTab() === 'native-optimizer') {
              <section class="compilation-section compilation-tab-panel" role="tabpanel">
                <div class="section-heading"><div><span class="eyebrow">NATIVE PYTORCH OPTIMIZER</span><h3>Runtime update rule</h3></div></div>
                <p class="compilation-description">The optimizer is instantiated for helper verification only. No training step or weight update runs during inference.</p>
                <label class="field"><span>Optimizer</span><select [value]="nativeOptimizerId()" (change)="chooseNativeOptimizer($event, catalog)">@for (item of catalog.optimizers; track item.id) { <option [value]="item.id">{{ item.label }}</option> }</select></label>
                @if (nativeOptimizer(); as optimizer) { <p class="compilation-description">{{ optimizer.description }}</p> }
                <div class="compilation-parameter-grid">@for (parameter of nativeOptimizerParameters(); track parameter.id) { <label class="field"><span>{{ parameter.label }}<small>{{ parameter.description }}</small></span>@if (parameter.type === 'boolean') { <input type="checkbox" [checked]="booleanParameterValue(parameter, draft().native.pytorch.optimizer_parameters)" (change)="setNativeOptimizerParameter(parameter, $event)" /> } @else { <input type="number" [value]="parameterValue(parameter, draft().native.pytorch.optimizer_parameters)" [min]="numericConstraint(parameter, 'min')" [max]="numericConstraint(parameter, 'max')" step="any" (change)="setNativeOptimizerParameter(parameter, $event)" /> }</label> }</div>
              </section>
            }
            @if (activeTab() === 'native-objective') {
              <section class="compilation-section compilation-tab-panel" role="tabpanel"><div class="section-heading"><div><span class="eyebrow">NATIVE PYTORCH LOSS</span><h3>Default objective</h3></div></div><p class="compilation-description">Loss helpers are evaluated against synthetic targets during the isolated smoke check; they do not train the model.</p><label class="field"><span>Loss</span><select [value]="nativeLossId()" (change)="setNativeLoss($event)"><option value="">No default loss</option>@for (item of catalog.losses; track item.id) { <option [value]="item.id">{{ item.label }}</option> }</select></label>@if (nativeLoss(); as loss) { <p class="compilation-description">{{ loss.description }}</p>@if (loss.formula) { <nw-math-formula [formula]="loss.formula"></nw-math-formula> } }</section>
            }
            @if (activeTab() === 'native-metrics') {
              <section class="compilation-section compilation-tab-panel" role="tabpanel"><div class="section-heading"><div><span class="eyebrow">NATIVE PYTORCH METRICS</span><h3>Default measurements</h3></div></div><div class="compilation-option-grid">@for (metric of catalog.metrics; track metric.id) { <label class="compilation-option"><input type="checkbox" [checked]="selectedNativeMetricIds().includes(metric.id)" (change)="toggleNativeMetric(metric.id, $event)" /><span><strong>{{ metric.label }}</strong><small>{{ metric.description }}</small>@if (metric.formula) { <nw-math-formula [formula]="metric.formula"></nw-math-formula> }</span></label> }</div></section>
            }
            @if (activeTab() === 'native-outputs') {
              <section class="compilation-section compilation-tab-panel" role="tabpanel"><div class="section-heading"><div><span class="eyebrow">NATIVE OUTPUT OVERRIDES</span><h3>Per-output loss</h3></div></div>@for (outputId of outputIds(); track outputId) { <label class="field"><span>{{ outputId }}</span><select [value]="outputLoss(draft().native.pytorch.losses, outputId)" (change)="setNativeOutputLoss(outputId, $event)"><option value="">Use default loss</option>@for (item of catalog.losses; track item.id) { <option [value]="item.id">{{ item.label }}</option> }</select></label> }</section>
            }
          }

          @if (!kerasEnabled() && nativeCapability(); as capability) {
            @if (activeTab() === 'native') {
              <section class="compilation-section compilation-tab-panel" role="tabpanel">
                <div class="section-heading"><div><span class="eyebrow">{{ capability.compiler }}</span><h3>Backend-native execution</h3></div></div>
                @if (!capability.available) { <p class="compilation-unavailable">{{ capability.reason ?? 'This compiler is not operational in the current environment. Settings can be saved for a later runtime.' }}</p> }
                <div class="compilation-parameter-grid">@for (parameter of capability.parameters; track parameter.id) { <label [class.toggle-field]="parameter.type === 'boolean'" [class.field]="parameter.type !== 'boolean'"><span>{{ parameter.label }}<small>{{ parameter.description }}</small></span>@if (parameter.type === 'boolean') { <input type="checkbox" [checked]="nativeParameterValue(parameter) === true" (change)="setNativeParameter(parameter, $event)" /> } @else if (parameter.type === 'enum') { <select [value]="nativeParameterValue(parameter)" (change)="setNativeParameter(parameter, $event)">@for (option of parameter.options ?? []; track option) { <option [value]="option">{{ option }}</option> }</select> } @else { <input type="number" [value]="nativeParameterValue(parameter)" (change)="setNativeParameter(parameter, $event)" /> }</label> }</div>
              </section>
            }
          }
        </div>
        <footer class="dialog-actions"><button type="button" class="secondary" (click)="close.emit()">Cancel</button><button type="button" class="primary" (click)="apply.emit(draft())">Apply configuration</button></footer>
      </section>
    </div>
  `,
})
export class CompilationConfigurationDialogComponent {
  readonly configuration = input.required<ModelConfiguration>()
  readonly kerasEnabled = input(false)
  readonly kerasCatalog = input<CompilationCatalog | null>(null)
  readonly nativeCapability = input<NativeCompilationCapability | null>(null)
  readonly outputIds = input<string[]>([])
  readonly close = output<void>()
  readonly apply = output<ModelConfiguration>()
  readonly draft = signal<ModelConfiguration>(emptyModelConfiguration())
  readonly activeTab = signal<CompilationTab>('optimizer')

  readonly nativeCatalog = computed(() => this.nativeCapability()?.catalog ?? null)
  readonly kerasOptimizer = computed(() => this.findOption(this.kerasCatalog(), this.draft().keras.optimizer, 'optimizers'))
  readonly kerasOptimizerId = computed(() => this.kerasOptimizer()?.id ?? this.draft().keras.optimizer)
  readonly kerasOptimizerParameters = computed(() => this.kerasOptimizer()?.parameters ?? [])
  readonly kerasLoss = computed(() => this.findOption(this.kerasCatalog(), this.draft().keras.losses['__default__'], 'losses'))
  readonly kerasLossId = computed(() => this.kerasLoss()?.id ?? this.draft().keras.losses['__default__'] ?? '')
  readonly selectedMetricIds = computed(() => this.draft().keras.metrics['__default__'] ?? [])
  readonly nativeOptimizer = computed(() => this.findOption(this.nativeCatalog(), this.draft().native.pytorch.optimizer, 'optimizers'))
  readonly nativeOptimizerId = computed(() => this.nativeOptimizer()?.id ?? this.draft().native.pytorch.optimizer)
  readonly nativeOptimizerParameters = computed(() => this.nativeOptimizer()?.parameters ?? [])
  readonly nativeLoss = computed(() => this.findOption(this.nativeCatalog(), this.draft().native.pytorch.losses['__default__'], 'losses'))
  readonly nativeLossId = computed(() => this.nativeLoss()?.id ?? this.draft().native.pytorch.losses['__default__'] ?? '')
  readonly selectedNativeMetricIds = computed(() => this.draft().native.pytorch.metrics['__default__'] ?? [])
  readonly tabs = computed<CompilationTabDefinition[]>(() => {
    if (this.kerasEnabled()) {
      return [
        { id: 'optimizer', label: 'Optimizer', detail: this.kerasOptimizer()?.label ?? 'Update rule' },
        { id: 'objective', label: 'Loss', detail: this.kerasLoss()?.label ?? 'Default objective' },
        { id: 'metrics', label: 'Metrics', detail: this.selectedMetricIds().length ? `${this.selectedMetricIds().length} selected` : 'None selected' },
        ...(this.outputIds().length ? [{ id: 'outputs' as const, label: 'Outputs', detail: `${this.outputIds().length} available` }] : []),
        { id: 'advanced', label: 'Advanced', detail: 'Model defaults' },
      ]
    }
    return [
      ...(this.nativeCatalog() ? [
        { id: 'native-optimizer' as const, label: 'Optimizer', detail: this.nativeOptimizer()?.label ?? 'Update rule' },
        { id: 'native-objective' as const, label: 'Loss', detail: this.nativeLoss()?.label ?? 'Not configured' },
        { id: 'native-metrics' as const, label: 'Metrics', detail: this.selectedNativeMetricIds().length ? `${this.selectedNativeMetricIds().length} selected` : 'None selected' },
        ...(this.outputIds().length ? [{ id: 'native-outputs' as const, label: 'Outputs', detail: `${this.outputIds().length} available` }] : []),
      ] : []),
      { id: 'native', label: 'Native compiler', detail: this.nativeCapability()?.compiler ?? 'Backend settings' },
    ]
  })

  constructor() {
    effect(() => {
      const configuration = this.configuration()
      this.draft.set(structuredClone(configuration))
      this.activeTab.set(this.kerasEnabled() ? 'optimizer' : this.nativeCatalog() ? 'native-optimizer' : 'native')
    })
  }

  panelId(tab: CompilationTab): string { return `compilation-tabpanel-${tab}` }

  closeFromBackdrop(event: MouseEvent): void {
    if (event.target === event.currentTarget) this.close.emit()
  }

  chooseKerasOptimizer(event: Event, catalog: CompilationCatalog): void {
    const id = this.value(event)
    const next = catalog.optimizers.find((item) => item.id === id)
    if (!next) return
    const parameters = Object.fromEntries(next.parameters.map((parameter) => [parameter.id, this.draft().keras.optimizer_parameters[parameter.id] ?? parameter.default]))
    this.updateDraft((current) => ({ ...current, keras: { ...current.keras, optimizer: id, optimizer_parameters: parameters } }))
  }

  setKerasOptimizerParameter(parameter: CompilationParameter, event: Event): void {
    const value = parameter.type === 'boolean' ? this.checked(event) : Number(this.value(event))
    this.updateDraft((current) => ({ ...current, keras: { ...current.keras, optimizer_parameters: { ...current.keras.optimizer_parameters, [parameter.id]: value } } }))
  }

  setKerasLoss(event: Event): void { this.updateDraft((current) => ({ ...current, keras: { ...current.keras, losses: { ...current.keras.losses, __default__: this.value(event) } } })) }

  toggleKerasMetric(metricId: string, event: Event): void {
    const metrics = this.toggleValue(this.selectedMetricIds(), metricId, this.checked(event))
    this.updateDraft((current) => ({ ...current, keras: { ...current.keras, metrics: { ...current.keras.metrics, __default__: metrics } } }))
  }

  setKerasOutputLoss(outputId: string, event: Event): void {
    const value = this.value(event)
    this.updateDraft((current) => {
      const losses = { ...current.keras.losses }
      if (value) losses[outputId] = value
      else delete losses[outputId]
      return { ...current, keras: { ...current.keras, losses } }
    })
  }

  setKerasText(key: 'default_initializer' | 'default_regularizer' | 'default_constraint', event: Event): void { this.updateDraft((current) => ({ ...current, keras: { ...current.keras, [key]: this.value(event) } })) }
  setKerasMixedPrecision(event: Event): void { this.updateDraft((current) => ({ ...current, keras: { ...current.keras, mixed_precision: this.checked(event) } })) }

  chooseNativeOptimizer(event: Event, catalog: CompilationCatalog): void {
    const id = this.value(event)
    const next = catalog.optimizers.find((item) => item.id === id)
    if (!next) return
    const parameters = Object.fromEntries(next.parameters.map((parameter) => [parameter.id, this.draft().native.pytorch.optimizer_parameters[parameter.id] ?? parameter.default]))
    this.updateDraft((current) => ({ ...current, native: { ...current.native, pytorch: { ...current.native.pytorch, optimizer: id, optimizer_parameters: parameters } } }))
  }

  setNativeOptimizerParameter(parameter: CompilationParameter, event: Event): void {
    const value = parameter.type === 'boolean' ? this.checked(event) : Number(this.value(event))
    this.updateDraft((current) => ({ ...current, native: { ...current.native, pytorch: { ...current.native.pytorch, optimizer_parameters: { ...current.native.pytorch.optimizer_parameters, [parameter.id]: value } } } }))
  }

  setNativeLoss(event: Event): void {
    const value = this.value(event)
    this.updateDraft((current) => {
      const losses = { ...current.native.pytorch.losses }
      if (value) losses['__default__'] = value
      else delete losses['__default__']
      return { ...current, native: { ...current.native, pytorch: { ...current.native.pytorch, losses } } }
    })
  }

  toggleNativeMetric(metricId: string, event: Event): void {
    const metrics = this.toggleValue(this.selectedNativeMetricIds(), metricId, this.checked(event))
    this.updateDraft((current) => ({ ...current, native: { ...current.native, pytorch: { ...current.native.pytorch, metrics: { ...current.native.pytorch.metrics, __default__: metrics } } } }))
  }

  setNativeOutputLoss(outputId: string, event: Event): void {
    const value = this.value(event)
    this.updateDraft((current) => {
      const losses = { ...current.native.pytorch.losses }
      if (value) losses[outputId] = value
      else delete losses[outputId]
      return { ...current, native: { ...current.native, pytorch: { ...current.native.pytorch, losses } } }
    })
  }

  nativeParameterValue(parameter: CompilationParameter): string | number | boolean {
    const capability = this.nativeCapability()
    if (!capability) return parameter.default
    const branch = this.draft().native[capability.backend] as unknown as Record<string, unknown>
    return this.scalarValue(branch[parameter.id] ?? parameter.default, parameter.default)
  }

  setNativeParameter(parameter: CompilationParameter, event: Event): void {
    const capability = this.nativeCapability()
    if (!capability) return
    const raw = parameter.type === 'boolean' ? this.checked(event) : this.value(event)
    const value = parameter.type === 'number' || parameter.type === 'integer' ? Number(raw) : raw
    this.updateDraft((current) => {
      const branch = { ...(current.native[capability.backend] as unknown as Record<string, unknown>), [parameter.id]: value }
      return { ...current, native: { ...current.native, [capability.backend]: branch } }
    })
  }

  outputLoss(losses: Record<string, string>, outputId: string): string { return losses[outputId] || '' }
  parameterValue(parameter: CompilationParameter, values: Record<string, unknown>): string | number | boolean { return this.scalarValue(values[parameter.id] ?? parameter.default, parameter.default) }
  booleanParameterValue(parameter: CompilationParameter, values: Record<string, unknown>): boolean { return Boolean(values[parameter.id] ?? parameter.default) }
  numericConstraint(parameter: CompilationParameter, key: 'min' | 'max'): number | null { const value = parameter.constraints[key]; return typeof value === 'number' ? value : null }

  private updateDraft(update: (current: ModelConfiguration) => ModelConfiguration): void { this.draft.update((current) => update(structuredClone(current))) }
  private findOption(catalog: CompilationCatalog | null, id: string | undefined, key: 'optimizers' | 'losses'): CompilationOption | undefined { return catalog?.[key].find((item) => item.id === id) ?? catalog?.[key][0] }
  private scalarValue(value: unknown, fallback: unknown): string | number | boolean { return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? value : typeof fallback === 'string' || typeof fallback === 'number' || typeof fallback === 'boolean' ? fallback : String(fallback) }
  private toggleValue(values: string[], value: string, enabled: boolean): string[] { return enabled ? [...new Set([...values, value])] : values.filter((item) => item !== value) }
  private value(event: Event): string { return (event.target as HTMLInputElement | HTMLSelectElement).value }
  private checked(event: Event): boolean { return (event.target as HTMLInputElement).checked }
}
