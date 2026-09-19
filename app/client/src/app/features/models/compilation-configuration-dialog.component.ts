import { Component, computed, effect, input, output, signal } from '@angular/core'
import type {
  CompilationObject,
  CompilationCatalog,
  CompilationOption,
  CompilationPlan,
  ExecutionBackend,
  ModelConfiguration,
  NativeCompilationCapability,
  ParameterDefinition,
} from '../../domain/types'
import { emptyModelConfiguration } from '../../domain/types'
import { MathFormulaComponent } from '../../shared/components/math-formula.component'
import { ModalGuardDirective } from '../../shared/components/modal-guard.directive'

export interface CompilationOutputDescriptor {
  id: string
  label: string
  selectors?: string[]
  task_hint?: string | null
}

type CompilationTab = 'optimizer' | 'outputs' | 'advanced' | 'native'

@Component({
  selector: 'nw-compilation-configuration-dialog',
  standalone: true,
  imports: [MathFormulaComponent, ModalGuardDirective],
  template: `
    <div class="dialog-backdrop compilation-modal-backdrop" nwModalGuard role="presentation" (mousedown)="closeFromBackdrop($event)">
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
              <div class="compilation-parameter-grid">@for (parameter of optimizerParameters(); track parameter.id) { <label class="field"><span>{{ parameter.display_name }}<small>{{ parameter.description }}</small></span>@if (parameter.type === 'boolean') { <input type="checkbox" [checked]="booleanParameterValue(parameter, optimizerParameterValues())" (change)="setOptimizerParameter(parameter, $event)" /> } @else if (parameter.type === 'enum') { <select [value]="parameterValue(parameter, optimizerParameterValues())" (change)="setOptimizerParameter(parameter, $event)">@for (option of parameter.options; track option) { <option [value]="option">{{ option }}</option> }</select> } @else { <input type="number" [value]="parameterValue(parameter, optimizerParameterValues())" [min]="numericConstraint(parameter, 'min')" [max]="numericConstraint(parameter, 'max')" step="any" (change)="setOptimizerParameter(parameter, $event)" /> }</label> }</div>
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
                        <label class="field"><span>Loss</span><select (change)="setOutputLoss(modelOutput.id, $event)"><option value="" [selected]="selectedLoss(modelOutput.id) === ''">Select loss</option>@for (item of compatibleOptions(catalog.losses, modelOutput); track item.id) { <option [value]="item.id" [selected]="item.id === selectedLoss(modelOutput.id)">{{ item.label }}</option> }</select></label>
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
              <div class="compilation-parameter-grid"><label class="field"><span>Initializer</span><select [value]="draft().keras.compilation.default_initializer.id" (change)="setKerasText('default_initializer', $event)">@for (item of catalog.initializers; track item) { <option [value]="item">{{ item }}</option> }</select></label><label class="field"><span>Regularizer</span><select [value]="draft().keras.compilation.default_regularizer?.id ?? 'none'" (change)="setKerasText('default_regularizer', $event)">@for (item of catalog.regularizers; track item) { <option [value]="item">{{ item }}</option> }</select></label><label class="field"><span>Constraint</span><select [value]="draft().keras.compilation.default_constraint?.id ?? 'none'" (change)="setKerasText('default_constraint', $event)">@for (item of catalog.constraints; track item) { <option [value]="item">{{ item }}</option> }</select></label><label class="toggle-field"><span>Mixed precision</span><input type="checkbox" [checked]="draft().keras.mixed_precision" (change)="setKerasMixedPrecision($event)" /></label></div>
            </section>
          }

          @if (activeTab() === 'native' && nativeCapability(); as capability) {
            <section class="compilation-section compilation-tab-panel" role="tabpanel">
              <div class="section-heading"><div><span class="eyebrow">{{ capability.compiler }}</span><h3>Backend-native execution</h3></div></div>
              @if (!capability.available) { <p class="compilation-unavailable">{{ capability.reason ?? 'This compiler is not operational in the current environment. Settings can still be saved for a later runtime.' }}</p> }
              <div class="compilation-parameter-grid">@for (parameter of capability.parameters; track parameter.id) { <label [class.toggle-field]="parameter.type === 'boolean'" [class.field]="parameter.type !== 'boolean'"><span>{{ parameter.display_name }}<small>{{ parameter.description }}</small></span>@if (parameter.type === 'boolean') { <input type="checkbox" [checked]="nativeParameterValue(parameter) === true" (change)="setNativeParameter(parameter, $event)" /> } @else if (parameter.type === 'enum') { <select [value]="nativeParameterValue(parameter)" (change)="setNativeParameter(parameter, $event)">@for (option of parameter.options; track option) { <option [value]="option">{{ option }}</option> }</select> } @else { <input type="number" [value]="nativeParameterValue(parameter)" (change)="setNativeParameter(parameter, $event)" /> }</label> }</div>
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

  readonly nativeCatalog = computed(() => this.nativeCapability()?.backend === 'pytorch' ? this.nativeCapability()?.catalog ?? null : null)
  readonly activeCatalog = computed(() => this.kerasEnabled() ? this.kerasCatalog() : this.nativeCatalog())
  readonly optimizer = computed(() => this.findOption(this.activeCatalog(), this.optimizerId(), 'optimizers'))
  readonly optimizerId = computed(() => this.kerasEnabled() ? this.draft().keras.compilation.optimizer.id : this.nativeCapability() ? this.nativeConfiguration(this.draft()).compilation.optimizer.id : '')
  readonly optimizerParameters = computed(() => this.optimizer()?.parameters ?? [])
  readonly optimizerParameterValues = computed(() => this.kerasEnabled() ? this.draft().keras.compilation.optimizer.parameters : this.nativeCapability() ? this.nativeConfiguration(this.draft()).compilation.optimizer.parameters : {})
  readonly optimizerLabel = computed(() => this.optimizer()?.label ?? this.optimizerId())

  constructor() {
    effect(() => {
      const configured = structuredClone(this.configuration())
      const catalog = this.activeCatalog()
      const modelOutputs = this.outputs()
      this.draft.set(catalog ? withOutputSuggestions(configured, catalog, modelOutputs, this.kerasEnabled(), this.nativeCapability()?.backend ?? 'pytorch') : configured)
      this.activeTab.set(modelOutputs.length && catalog ? 'outputs' : catalog?.optimizers.length ? 'optimizer' : this.nativeCapability() ? 'native' : 'advanced')
    })
  }

  closeFromBackdrop(event: MouseEvent): void { if (event.target === event.currentTarget) this.close.emit() }
  lossOption(catalog: CompilationCatalog, id: string): CompilationOption | null { return catalog.losses.find((item) => item.id === id) ?? null }
  compatibleOptions(options: CompilationOption[], modelOutput: CompilationOutputDescriptor): CompilationOption[] { const task = normalizedTask(modelOutput.task_hint); return task ? options.filter((item) => !item.supported_tasks?.length || item.supported_tasks.includes(task)) : options }
  taskLabel(value?: string | null): string | null { return normalizedTask(value)?.replaceAll('_', ' ') ?? null }
  selectedLoss(outputId: string): string { const mapping = this.lossMapping(); return this.mappingValue(mapping, outputId)?.id ?? '' }
  selectedMetrics(outputId: string): string[] { const mapping = this.metricMapping(); return (this.mappingValue(mapping, outputId) ?? []).map((item) => item.id) }

  chooseOptimizer(event: Event, catalog: CompilationCatalog): void {
    const id = (event.target as HTMLSelectElement).value
    const option = catalog.optimizers.find((item) => item.id === id)
    const parameters = Object.fromEntries((option?.parameters ?? []).map((parameter) => [parameter.id, parameter.default]))
    this.draft.update((current) => this.withCompilation(current, {
      optimizer: { id, parameters },
    }))
  }

  setOptimizerParameter(parameter: ParameterDefinition, event: Event): void {
    const element = event.target as HTMLInputElement | HTMLSelectElement
    this.draft.update((current) => {
      const compilation = this.currentCompilation(current)
      return this.withCompilation(current, {
        optimizer: {
          ...compilation.optimizer,
          parameters: { ...compilation.optimizer.parameters, [parameter.id]: parseCompilationParameter(parameter, element) },
        },
      })
    })
  }

  setOutputLoss(outputId: string, event: Event): void {
    const value = (event.target as HTMLSelectElement).value
    const key = this.outputKey(outputId)
    this.draft.update((current) => {
      const mapping = { ...this.currentCompilation(current).losses }
      if (value) mapping[key] = { id: value, parameters: {} }
      else delete mapping[key]
      return this.withCompilation(current, { losses: mapping })
    })
  }

  toggleOutputMetric(outputId: string, metricId: string, event: Event): void {
    const checked = (event.target as HTMLInputElement).checked
    const key = this.outputKey(outputId)
    this.draft.update((current) => {
      const mapping = { ...this.currentCompilation(current).metrics }
      const selected = new Set((mapping[key] ?? mapping['__default__'] ?? []).map((item) => item.id))
      if (checked) selected.add(metricId)
      else selected.delete(metricId)
      mapping[key] = [...selected].map((id) => ({ id, parameters: {} }))
      return this.withCompilation(current, { metrics: mapping })
    })
  }

  setKerasText(key: 'default_initializer' | 'default_regularizer' | 'default_constraint', event: Event): void {
    const value = (event.target as HTMLSelectElement).value
    this.draft.update((current) => ({
      ...current,
      keras: {
        ...current.keras,
        compilation: {
          ...current.keras.compilation,
          [key]: value === 'none' && key !== 'default_initializer' ? null : { id: value, parameters: {} },
        },
      },
    }))
  }
  setKerasMixedPrecision(event: Event): void { const value = (event.target as HTMLInputElement).checked; this.draft.update((current) => ({ ...current, keras: { ...current.keras, mixed_precision: value } })) }
  parameterValue(parameter: ParameterDefinition, values: Record<string, unknown>): unknown { return values[parameter.id] ?? parameter.default }
  booleanParameterValue(parameter: ParameterDefinition, values: Record<string, unknown>): boolean { return Boolean(this.parameterValue(parameter, values)) }
  numericConstraint(parameter: ParameterDefinition, key: 'min' | 'max'): number | null { const value = parameter.constraints[key]; return typeof value === 'number' ? value : null }

  nativeParameterValue(parameter: ParameterDefinition): unknown {
    return this.nativeConfiguration(this.draft()).compiler.parameters[parameter.id] ?? parameter.default
  }

  setNativeParameter(parameter: ParameterDefinition, event: Event): void {
    const element = event.target as HTMLInputElement | HTMLSelectElement
    const value = parseCompilationParameter(parameter, element)
    const backend = this.nativeCapability()?.backend
    this.draft.update((current) => {
      if (!backend) return current
      const profile = this.nativeConfiguration(current)
      return updateNativeConfiguration(current, backend, {
        compiler: {
          ...profile.compiler,
          parameters: { ...profile.compiler.parameters, [parameter.id]: value },
        },
      })
    })
  }

  private outputKey(outputId: string): string { return this.outputs().length === 1 ? '__default__' : outputId }
  private mappingValue<T>(mapping: Record<string, T>, outputId: string): T | undefined {
    for (const selector of [this.outputKey(outputId), ...(this.outputs().find((item) => item.id === outputId)?.selectors ?? [])]) {
      if (mapping[selector] !== undefined) return mapping[selector]
    }
    return mapping['__default__']
  }
  private lossMapping(): Record<string, CompilationObject> { return this.currentCompilation(this.draft()).losses }
  private metricMapping(): Record<string, CompilationObject[]> { return this.currentCompilation(this.draft()).metrics }
  private nativeBackend(): ExecutionBackend { return this.nativeCapability()?.backend ?? 'pytorch' }
  private nativeConfiguration(configuration: ModelConfiguration): ModelConfiguration['native'][ExecutionBackend] { return configuration.native[this.nativeBackend()] }
  private currentCompilation(configuration: ModelConfiguration): CompilationPlan { return this.kerasEnabled() ? configuration.keras.compilation : this.nativeConfiguration(configuration).compilation }
  private withCompilation(configuration: ModelConfiguration, patch: Partial<CompilationPlan>): ModelConfiguration {
    const compilation = { ...this.currentCompilation(configuration), ...patch }
    if (this.kerasEnabled()) return { ...configuration, keras: { ...configuration.keras, compilation } }
    return updateNativeConfiguration(configuration, this.nativeBackend(), { compilation })
  }
  private findOption(catalog: CompilationCatalog | null, id: string | undefined, field: 'optimizers' | 'losses' | 'metrics'): CompilationOption | null { return catalog?.[field].find((item) => item.id === id) ?? null }
}

function updateNativeConfiguration(
  configuration: ModelConfiguration,
  backend: ExecutionBackend,
  patch: { compilation?: CompilationPlan; compiler?: ModelConfiguration['native'][ExecutionBackend]['compiler'] },
): ModelConfiguration {
  if (backend === 'tensorflow') {
    return {
      ...configuration,
      native: {
        ...configuration.native,
        tensorflow: { ...configuration.native.tensorflow, ...patch },
      },
    }
  }
  if (backend === 'jax') {
    return {
      ...configuration,
      native: {
        ...configuration.native,
        jax: { ...configuration.native.jax, ...patch },
      },
    }
  }
  return {
    ...configuration,
    native: {
      ...configuration.native,
      pytorch: { ...configuration.native.pytorch, ...patch },
    },
  }
}

function withOutputSuggestions(
  configuration: ModelConfiguration,
  catalog: CompilationCatalog,
  outputs: CompilationOutputDescriptor[],
  keras: boolean,
  backend: ExecutionBackend,
): ModelConfiguration {
  if (!outputs.length) return configuration
  const current = keras ? configuration.keras.compilation : configuration.native[backend].compilation
  const losses = { ...current.losses }
  const metrics = { ...current.metrics }
  for (const modelOutput of outputs) {
    const key = outputs.length === 1 ? '__default__' : modelOutput.id
    const task = normalizedTask(modelOutput.task_hint)
    if (!task) continue
    const currentLoss = mappingValue(losses, key, modelOutput.selectors)?.id
    const currentLossOption = catalog.losses.find((item) => item.id === currentLoss)
    if (!currentLossOption || (currentLossOption.supported_tasks?.length && !currentLossOption.supported_tasks.includes(task))) {
      const suggested = catalog.losses.find((item) => !item.supported_tasks?.length || item.supported_tasks.includes(task))
      if (suggested) losses[key] = { id: suggested.id, parameters: {} }
    }
    const currentMetrics = mappingValue(metrics, key, modelOutput.selectors) ?? []
    const compatibleCurrent = currentMetrics.filter((item) => { const option = catalog.metrics.find((candidate) => candidate.id === item.id); return option && (!option.supported_tasks?.length || option.supported_tasks.includes(task)) })
    if (!compatibleCurrent.length) metrics[key] = catalog.metrics.filter((item) => !item.supported_tasks?.length || item.supported_tasks.includes(task)).slice(0, 2).map((item) => ({ id: item.id, parameters: {} }))
  }
  return keras
    ? { ...configuration, keras: { ...configuration.keras, compilation: { ...configuration.keras.compilation, losses, metrics } } }
    : updateNativeConfiguration(configuration, backend, { compilation: { ...configuration.native[backend].compilation, losses, metrics } })
}

function mappingValue<T>(mapping: Record<string, T>, key: string, selectors?: string[]): T | undefined {
  for (const selector of [key, ...(selectors ?? [])]) {
    if (mapping[selector] !== undefined) return mapping[selector]
  }
  return mapping['__default__']
}

function parseCompilationParameter(parameter: ParameterDefinition, element: HTMLInputElement | HTMLSelectElement): unknown {
  if (parameter.type === 'boolean') return (element as HTMLInputElement).checked
  if (parameter.type === 'integer') return Number.parseInt(element.value, 10)
  if (parameter.type === 'number') return Number(element.value)
  return element.value
}

function normalizedTask(value?: string | null): 'regression' | 'binary_classification' | 'multiclass_classification' | null {
  const normalized = value?.toLowerCase() ?? ''
  if (normalized.includes('regress') || normalized.includes('continuous')) return 'regression'
  if (normalized.includes('binary')) return 'binary_classification'
  if (normalized.includes('multiclass') || normalized.includes('categor') || normalized === 'classification') return 'multiclass_classification'
  return null
}
