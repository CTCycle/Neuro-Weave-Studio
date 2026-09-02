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

export interface CompilationOutputDescriptor {
  id: string
  label: string
  task_hint?: string | null
}

type CompilationTab = 'optimizer' | 'outputs' | 'advanced' | 'native'

@Component({
  selector: 'nw-compilation-configuration-dialog',
  standalone: true,
  imports: [MathFormulaComponent],
  template: `
    <div class="dialog-backdrop compilation-modal-backdrop" role="presentation" (mousedown)="closeFromBackdrop($event)">
      <section class="compilation-modal panel" role="dialog" aria-modal="true" aria-labelledby="compilation-modal-title" (mousedown)="$event.stopPropagation()">
        <header class="compilation-modal-header">
          <div><span class="eyebrow">COMPILATION CONFIGURATION</span><h2 id="compilation-modal-title">{{ kerasEnabled() ? 'Configure Keras compilation' : 'Configure native ' + (nativeCapability()?.backend ?? 'backend') }}</h2><p>Optimizer, output objectives, metrics, and compiler settings are stored against the saved graph revision.</p></div>
          <button type="button" class="panel-close" (click)="close.emit()" aria-label="Close compilation configuration">×</button>
        </header>

        <div class="compilation-modal-tabs" role="tablist" aria-label="Compilation configuration sections">
          @if (activeCatalog()?.optimizers?.length) { <button type="button" role="tab" class="compilation-tab" [class.active]="activeTab() === 'optimizer'" [attr.aria-selected]="activeTab() === 'optimizer'" (click)="activeTab.set('optimizer')"><span>Optimizer</span><small>{{ optimizerLabel() }}</small></button> }
          @if (outputs().length && activeCatalog()) { <button type="button" role="tab" class="compilation-tab" [class.active]="activeTab() === 'outputs'" [attr.aria-selected]="activeTab() === 'outputs'" (click)="activeTab.set('outputs')"><span>{{ outputs().length === 1 ? 'Objective' : 'Outputs' }}</span><small>{{ outputs().length === 1 ? outputs()[0].label : outputs().length + ' configured heads' }}</small></button> }
          @if (kerasEnabled()) { <button type="button" role="tab" class="compilation-tab" [class.active]="activeTab() === 'advanced'" [attr.aria-selected]="activeTab() === 'advanced'" (click)="activeTab.set('advanced')"><span>Advanced</span><small>Defaults and precision</small></button> }
          @if (nativeCapability()) { <button type="button" role="tab" class="compilation-tab" [class.active]="activeTab() === 'native'" [attr.aria-selected]="activeTab() === 'native'" (click)="activeTab.set('native')"><span>Compiler</span><small>{{ nativeCapability()?.compiler }}</small></button> }
        </div>

        <div class="compilation-modal-body">
          @if (activeTab() === 'optimizer' && activeCatalog(); as catalog) {
            <section class="compilation-section compilation-tab-panel" role="tabpanel">
              <div class="section-heading"><div><span class="eyebrow">OPTIMIZER</span><h3>Update rule</h3></div></div>
              <label class="field"><span>Optimizer</span><select [value]="optimizerId()" (change)="chooseOptimizer($event, catalog)">@for (item of catalog.optimizers; track item.id) { <option [value]="item.id">{{ item.label }}</option> }</select></label>
              @if (optimizer(); as selected) { <p class="compilation-description">{{ selected.description }}</p> }
              <div class="compilation-parameter-grid">@for (parameter of optimizerParameters(); track parameter.id) { <label class="field"><span>{{ parameter.label }}<small>{{ parameter.description }}</small></span>@if (parameter.type === 'boolean') { <input type="checkbox" [checked]="booleanParameterValue(parameter, optimizerParameterValues())" (change)="setOptimizerParameter(parameter, $event)" /> } @else { <input type="number" [value]="parameterValue(parameter, optimizerParameterValues())" [min]="numericConstraint(parameter, 'min')" [max]="numericConstraint(parameter, 'max')" step="any" (change)="setOptimizerParameter(parameter, $event)" /> }</label> }</div>
            </section>
          }

          @if (activeTab() === 'outputs' && activeCatalog(); as catalog) {
            <section class="compilation-section compilation-tab-panel output-configuration" role="tabpanel">
              <div class="section-heading"><div><span class="eyebrow">MODEL OUTPUTS</span><h3>{{ outputs().length === 1 ? 'Objective and metrics' : 'Per-output objectives and metrics' }}</h3></div><small class="compilation-description">{{ outputs().length === 1 ? 'A single output keeps one compact objective.' : 'Each graph output is configured independently.' }}</small></div>
              <div class="output-config-list">
                @for (modelOutput of outputs(); track modelOutput.id) {
                  <section class="output-config" [class.single-output]="outputs().length === 1">
                    <header><div><span class="eyebrow">{{ outputs().length === 1 ? 'OUTPUT' : 'OUTPUT HEAD' }}</span><h4>{{ modelOutput.label }}</h4></div>@if (taskLabel(modelOutput.task_hint); as task) { <span class="output-task">{{ task }}</span> }</header>
                    <div class="output-config-grid">
                      <div class="output-loss">
                        <label class="field"><span>Loss</span><select [value]="selectedLoss(modelOutput.id)" (change)="setOutputLoss(modelOutput.id, $event)"><option value="">Select loss</option>@for (item of compatibleOptions(catalog.losses, modelOutput); track item.id) { <option [value]="item.id">{{ item.label }}</option> }</select></label>
                        @if (lossOption(catalog, selectedLoss(modelOutput.id)); as loss) { <p class="compilation-description">{{ loss.description }}</p>@if (loss.formula) { <nw-math-formula [formula]="loss.formula"></nw-math-formula> } }
                      </div>
                      <div class="output-metrics"><span class="field-caption">Metrics</span><div class="compilation-option-grid compact">@for (metric of compatibleOptions(catalog.metrics, modelOutput); track metric.id) { <label class="compilation-option"><input type="checkbox" [checked]="selectedMetrics(modelOutput.id).includes(metric.id)" (change)="toggleOutputMetric(modelOutput.id, metric.id, $event)" /><span><strong>{{ metric.label }}</strong><small>{{ metric.description }}</small>@if (metric.formula) { <nw-math-formula [formula]="metric.formula"></nw-math-formula> }</span></label> }</div></div>
                    </div>
                  </section>
                }
              </div>
            </section>
          }

          @if (activeTab() === 'advanced' && kerasEnabled() && kerasCatalog(); as catalog) {
            <section class="compilation-section compilation-tab-panel" role="tabpanel">
              <div class="section-heading"><div><span class="eyebrow">ADVANCED OPTIONS</span><h3>Model defaults</h3></div></div>
              <div class="compilation-parameter-grid"><label class="field"><span>Initializer</span><select [value]="draft().keras.default_initializer" (change)="setKerasText('default_initializer', $event)">@for (item of catalog.initializers; track item) { <option [value]="item">{{ item }}</option> }</select></label><label class="field"><span>Regularizer</span><select [value]="draft().keras.default_regularizer" (change)="setKerasText('default_regularizer', $event)">@for (item of catalog.regularizers; track item) { <option [value]="item">{{ item }}</option> }</select></label><label class="field"><span>Constraint</span><select [value]="draft().keras.default_constraint" (change)="setKerasText('default_constraint', $event)">@for (item of catalog.constraints; track item) { <option [value]="item">{{ item }}</option> }</select></label><label class="toggle-field"><span>Mixed precision</span><input type="checkbox" [checked]="draft().keras.mixed_precision" (change)="setKerasMixedPrecision($event)" /></label></div>
            </section>
          }

          @if (activeTab() === 'native' && nativeCapability(); as capability) {
            <section class="compilation-section compilation-tab-panel" role="tabpanel">
              <div class="section-heading"><div><span class="eyebrow">{{ capability.compiler }}</span><h3>Backend-native execution</h3></div></div>
              @if (!capability.available) { <p class="compilation-unavailable">{{ capability.reason ?? 'This compiler is not operational in the current environment. Settings can still be saved for a later runtime.' }}</p> }
              <div class="compilation-parameter-grid">@for (parameter of capability.parameters; track parameter.id) { <label [class.toggle-field]="parameter.type === 'boolean'" [class.field]="parameter.type !== 'boolean'"><span>{{ parameter.label }}<small>{{ parameter.description }}</small></span>@if (parameter.type === 'boolean') { <input type="checkbox" [checked]="nativeParameterValue(parameter) === true" (change)="setNativeParameter(parameter, $event)" /> } @else if (parameter.type === 'enum') { <select [value]="nativeParameterValue(parameter)" (change)="setNativeParameter(parameter, $event)">@for (option of parameter.options ?? []; track option) { <option [value]="option">{{ option }}</option> }</select> } @else { <input type="number" [value]="nativeParameterValue(parameter)" (change)="setNativeParameter(parameter, $event)" /> }</label> }</div>
            </section>
          }
        </div>
        <footer class="dialog-actions"><button type="button" class="secondary" (click)="close.emit()">Cancel</button><button type="button" class="primary" (click)="apply.emit(draft())">Apply configuration</button></footer>
      </section>
    </div>
  `,
  styles: [`
    .compilation-modal { width:min(980px, calc(100vw - 32px)); max-height:min(860px, calc(100vh - 32px)); }
    .output-config-list { display:grid; gap:14px; }
    .output-config { padding:14px 0 2px; border-top:1px solid var(--border); }
    .output-config:first-child { border-top:0; padding-top:0; }
    .output-config > header { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:12px; }
    .output-config h4 { margin:3px 0 0; font-size:15px; }
    .output-task { border:1px solid var(--border); border-radius:4px; padding:4px 6px; color:var(--cyan); font:600 9px/1 'IBM Plex Mono', Consolas, monospace; text-transform:uppercase; }
    .output-config-grid { display:grid; grid-template-columns:minmax(220px,.8fr) minmax(300px,1.2fr); gap:18px; align-items:start; }
    .single-output .output-config-grid { grid-template-columns:minmax(250px,.9fr) minmax(320px,1.1fr); }
    .field-caption { display:block; margin-bottom:7px; color:var(--muted); font-size:11px; font-weight:600; }
    .compilation-option-grid.compact { grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; }
    .compilation-option { min-width:0; }
    .compilation-option span { min-width:0; }
    @media (max-width: 760px) { .output-config-grid, .single-output .output-config-grid { grid-template-columns:1fr; } .compilation-option-grid.compact { grid-template-columns:1fr; } }
  `],
})
export class CompilationConfigurationDialogComponent {
  readonly configuration = input.required<ModelConfiguration>()
  readonly kerasEnabled = input(false)
  readonly kerasCatalog = input<CompilationCatalog | null>(null)
  readonly nativeCapability = input<NativeCompilationCapability | null>(null)
  readonly outputs = input<CompilationOutputDescriptor[]>([])
  readonly close = output<void>()
  readonly apply = output<ModelConfiguration>()
  readonly draft = signal<ModelConfiguration>(emptyModelConfiguration())
  readonly activeTab = signal<CompilationTab>('outputs')

  readonly nativeCatalog = computed(() => this.nativeCapability()?.catalog ?? null)
  readonly activeCatalog = computed(() => this.kerasEnabled() ? this.kerasCatalog() : this.nativeCatalog())
  readonly optimizer = computed(() => this.findOption(this.activeCatalog(), this.optimizerId(), 'optimizers'))
  readonly optimizerId = computed(() => this.kerasEnabled() ? this.draft().keras.optimizer : this.draft().native.pytorch.optimizer)
  readonly optimizerParameters = computed(() => this.optimizer()?.parameters ?? [])
  readonly optimizerParameterValues = computed(() => this.kerasEnabled() ? this.draft().keras.optimizer_parameters : this.draft().native.pytorch.optimizer_parameters)
  readonly optimizerLabel = computed(() => this.optimizer()?.label ?? this.optimizerId())

  constructor() {
    effect(() => {
      const configured = structuredClone(this.configuration())
      const catalog = this.activeCatalog()
      const modelOutputs = this.outputs()
      this.draft.set(catalog ? withOutputSuggestions(configured, catalog, modelOutputs, this.kerasEnabled()) : configured)
      this.activeTab.set(modelOutputs.length && catalog ? 'outputs' : catalog?.optimizers.length ? 'optimizer' : this.nativeCapability() ? 'native' : 'advanced')
    })
  }

  closeFromBackdrop(event: MouseEvent): void { if (event.target === event.currentTarget) this.close.emit() }
  lossOption(catalog: CompilationCatalog, id: string): CompilationOption | null { return catalog.losses.find((item) => item.id === id) ?? null }
  compatibleOptions(options: CompilationOption[], modelOutput: CompilationOutputDescriptor): CompilationOption[] { const task = normalizedTask(modelOutput.task_hint); return task ? options.filter((item) => !item.supported_tasks?.length || item.supported_tasks.includes(task)) : options }
  taskLabel(value?: string | null): string | null { return normalizedTask(value)?.replaceAll('_', ' ') ?? null }
  selectedLoss(outputId: string): string { const mapping = this.lossMapping(); return mapping[this.outputKey(outputId)] ?? mapping['__default__'] ?? '' }
  selectedMetrics(outputId: string): string[] { const mapping = this.metricMapping(); return mapping[this.outputKey(outputId)] ?? mapping['__default__'] ?? [] }

  chooseOptimizer(event: Event, catalog: CompilationCatalog): void {
    const id = (event.target as HTMLSelectElement).value
    const option = catalog.optimizers.find((item) => item.id === id)
    const parameters = Object.fromEntries((option?.parameters ?? []).map((parameter) => [parameter.id, parameter.default]))
    this.draft.update((current) => this.kerasEnabled()
      ? { ...current, keras: { ...current.keras, optimizer: id, optimizer_parameters: parameters } }
      : { ...current, native: { ...current.native, pytorch: { ...current.native.pytorch, optimizer: id, optimizer_parameters: parameters } } })
  }

  setOptimizerParameter(parameter: CompilationParameter, event: Event): void {
    const element = event.target as HTMLInputElement
    const value: unknown = parameter.type === 'boolean' ? element.checked : parameter.type === 'integer' ? Number.parseInt(element.value, 10) : Number(element.value)
    this.draft.update((current) => this.kerasEnabled()
      ? { ...current, keras: { ...current.keras, optimizer_parameters: { ...current.keras.optimizer_parameters, [parameter.id]: value } } }
      : { ...current, native: { ...current.native, pytorch: { ...current.native.pytorch, optimizer_parameters: { ...current.native.pytorch.optimizer_parameters, [parameter.id]: value } } } })
  }

  setOutputLoss(outputId: string, event: Event): void {
    const value = (event.target as HTMLSelectElement).value
    const key = this.outputKey(outputId)
    this.draft.update((current) => {
      const mapping = { ...(this.kerasEnabled() ? current.keras.losses : current.native.pytorch.losses) }
      if (value) mapping[key] = value
      else delete mapping[key]
      return this.kerasEnabled()
        ? { ...current, keras: { ...current.keras, losses: mapping } }
        : { ...current, native: { ...current.native, pytorch: { ...current.native.pytorch, losses: mapping } } }
    })
  }

  toggleOutputMetric(outputId: string, metricId: string, event: Event): void {
    const checked = (event.target as HTMLInputElement).checked
    const key = this.outputKey(outputId)
    this.draft.update((current) => {
      const mapping = { ...(this.kerasEnabled() ? current.keras.metrics : current.native.pytorch.metrics) }
      const selected = new Set(mapping[key] ?? mapping['__default__'] ?? [])
      if (checked) selected.add(metricId)
      else selected.delete(metricId)
      mapping[key] = [...selected]
      return this.kerasEnabled()
        ? { ...current, keras: { ...current.keras, metrics: mapping } }
        : { ...current, native: { ...current.native, pytorch: { ...current.native.pytorch, metrics: mapping } } }
    })
  }

  setKerasText(key: 'default_initializer' | 'default_regularizer' | 'default_constraint', event: Event): void { const value = (event.target as HTMLSelectElement).value; this.draft.update((current) => ({ ...current, keras: { ...current.keras, [key]: value } })) }
  setKerasMixedPrecision(event: Event): void { const value = (event.target as HTMLInputElement).checked; this.draft.update((current) => ({ ...current, keras: { ...current.keras, mixed_precision: value } })) }
  parameterValue(parameter: CompilationParameter, values: Record<string, unknown>): unknown { return values[parameter.id] ?? parameter.default }
  booleanParameterValue(parameter: CompilationParameter, values: Record<string, unknown>): boolean { return Boolean(this.parameterValue(parameter, values)) }
  numericConstraint(parameter: CompilationParameter, key: 'min' | 'max'): number | null { const value = parameter.constraints[key]; return typeof value === 'number' ? value : null }

  nativeParameterValue(parameter: CompilationParameter): unknown {
    const project = this.draft().native
    const backend = this.nativeCapability()?.backend
    if (backend === 'tensorflow') return project.tensorflow[parameter.id as keyof typeof project.tensorflow] ?? parameter.default
    if (backend === 'jax') return project.jax[parameter.id as keyof typeof project.jax] ?? parameter.default
    return project.pytorch[parameter.id as keyof typeof project.pytorch] ?? parameter.default
  }

  setNativeParameter(parameter: CompilationParameter, event: Event): void {
    const element = event.target as HTMLInputElement | HTMLSelectElement
    const value: unknown = parameter.type === 'boolean' ? (element as HTMLInputElement).checked : parameter.type === 'integer' ? Number.parseInt(element.value, 10) : parameter.type === 'number' ? Number(element.value) : element.value
    const backend = this.nativeCapability()?.backend
    this.draft.update((current) => {
      if (backend === 'tensorflow') return { ...current, native: { ...current.native, tensorflow: { ...current.native.tensorflow, [parameter.id]: value } } }
      if (backend === 'jax') return { ...current, native: { ...current.native, jax: { ...current.native.jax, [parameter.id]: value } } }
      return { ...current, native: { ...current.native, pytorch: { ...current.native.pytorch, [parameter.id]: value } } }
    })
  }

  private outputKey(outputId: string): string { return this.outputs().length === 1 ? '__default__' : outputId }
  private lossMapping(): Record<string, string> { return this.kerasEnabled() ? this.draft().keras.losses : this.draft().native.pytorch.losses }
  private metricMapping(): Record<string, string[]> { return this.kerasEnabled() ? this.draft().keras.metrics : this.draft().native.pytorch.metrics }
  private findOption(catalog: CompilationCatalog | null, id: string | undefined, field: 'optimizers' | 'losses' | 'metrics'): CompilationOption | null { return catalog?.[field].find((item) => item.id === id) ?? null }
}

function withOutputSuggestions(configuration: ModelConfiguration, catalog: CompilationCatalog, outputs: CompilationOutputDescriptor[], keras: boolean): ModelConfiguration {
  if (!outputs.length) return configuration
  const losses = { ...(keras ? configuration.keras.losses : configuration.native.pytorch.losses) }
  const metrics = { ...(keras ? configuration.keras.metrics : configuration.native.pytorch.metrics) }
  for (const modelOutput of outputs) {
    const key = outputs.length === 1 ? '__default__' : modelOutput.id
    const task = normalizedTask(modelOutput.task_hint)
    if (!task) continue
    const currentLoss = losses[key] ?? losses['__default__']
    const currentLossOption = catalog.losses.find((item) => item.id === currentLoss)
    if (!currentLossOption || (currentLossOption.supported_tasks?.length && !currentLossOption.supported_tasks.includes(task))) {
      const suggested = catalog.losses.find((item) => !item.supported_tasks?.length || item.supported_tasks.includes(task))
      if (suggested) losses[key] = suggested.id
    }
    const currentMetrics = metrics[key] ?? metrics['__default__'] ?? []
    const compatibleCurrent = currentMetrics.filter((id) => {
      const option = catalog.metrics.find((item) => item.id === id)
      return Boolean(option && (!option.supported_tasks?.length || option.supported_tasks.includes(task)))
    })
    if (!compatibleCurrent.length) metrics[key] = catalog.metrics.filter((item) => !item.supported_tasks?.length || item.supported_tasks.includes(task)).slice(0, 2).map((item) => item.id)
  }
  return keras
    ? { ...configuration, keras: { ...configuration.keras, losses, metrics } }
    : { ...configuration, native: { ...configuration.native, pytorch: { ...configuration.native.pytorch, losses, metrics } } }
}

function normalizedTask(value?: string | null): 'regression' | 'binary_classification' | 'multiclass_classification' | null {
  const normalized = value?.toLowerCase() ?? ''
  if (normalized.includes('regress') || normalized.includes('continuous')) return 'regression'
  if (normalized.includes('binary')) return 'binary_classification'
  if (normalized.includes('multiclass') || normalized.includes('categor') || normalized === 'classification') return 'multiclass_classification'
  return null
}
