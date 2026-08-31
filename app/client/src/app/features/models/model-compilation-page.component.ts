import { Component, computed, effect, inject, signal } from '@angular/core'
import { LucideCheckCircle2, LucideClipboardCheck, LucideFileCode2, LucideRefreshCw, LucideSettings2 } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import type { ModelConfiguration, NativeCompilationCapability } from '../../domain/types'
import { implementationLabel } from '../../domain/types'
import { CompilationStatusBadgeComponent } from '../../shared/components/model-status-badge.component'
import { CompilationConfigurationDialogComponent } from './compilation-configuration-dialog.component'
import { ModelWorkspaceState } from './model-workspace.state'

@Component({
  selector: 'nw-model-compilation-page',
  standalone: true,
  imports: [LucideCheckCircle2, LucideClipboardCheck, LucideFileCode2, LucideRefreshCw, LucideSettings2, CompilationStatusBadgeComponent, CompilationConfigurationDialogComponent],
  template: `
    @if (state.project(); as project) {
      <div class="workflow-page compilation-page">
        <div class="workflow-intro"><div><span class="eyebrow">COMPILATION</span><h2>Configure and generate source</h2><p>Compilation owns the saved execution profile and generated source for this revision. It does not train weights or download artifacts.</p></div><nw-compilation-status-badge [compiled]="!!project.generated_source" [target]="implementationLabel(project.execution_backend, project.keras_enabled)"></nw-compilation-status-badge></div>
        <div class="workflow-grid"><section class="workflow-card panel"><div class="section-heading"><div><span class="eyebrow">EXECUTION PROFILE</span><h3>{{ implementationLabel(project.execution_backend, project.keras_enabled) }}</h3></div><svg lucideSettings2 [size]="18"></svg></div><p class="muted">Revision {{ project.current_revision }} is the saved graph used for this source snapshot.</p><div class="workflow-stat-row"><div><span>Backend</span><strong>{{ project.execution_backend }}</strong></div><div><span>Configuration</span><strong>{{ configurationSummary(project) }}</strong></div><div><span>Source</span><strong>{{ project.generated_source ? 'Available' : 'Not generated' }}</strong></div></div><div class="workflow-card-actions"><button class="secondary" type="button" (click)="openConfiguration()" [disabled]="!resources.runtimeReady() || configurationBusy()"><svg lucideSettings2 [size]="16"></svg>{{ configurationBusy() ? 'Saving…' : 'Configure compilation' }}</button><button class="secondary" type="button" (click)="validate()" [disabled]="validationBusy()"><svg lucideClipboardCheck [size]="16"></svg>{{ validationBusy() ? 'Validating…' : 'Validate profile' }}</button></div></section><section class="workflow-card panel"><div class="section-heading"><div><span class="eyebrow">GENERATED SOURCE</span><h3>{{ project.generated_source ? 'Current source snapshot' : 'No source snapshot' }}</h3></div><svg lucideFileCode2 [size]="18"></svg></div><p class="muted">Generate inspectable source from the current saved revision. Copy is available here; artifact downloads belong to Export.</p><div class="workflow-stat-row"><div><span>Revision</span><strong>{{ project.current_revision }}</strong></div><div><span>Profile</span><strong>{{ project.keras_enabled ? 'Keras API' : 'Native API' }}</strong></div><div><span>Training state</span><strong>Unavailable</strong></div></div><button class="primary" type="button" (click)="compile()" [disabled]="busy()">@if (busy()) { <svg class="spin" lucideRefreshCw [size]="16"></svg> } @else { <svg lucideCheckCircle2 [size]="16"></svg> }{{ busy() ? 'Compiling…' : project.generated_source ? 'Recompile source' : 'Compile source' }}</button></section></div>
        @if (code(); as source) { <section class="generated-code panel"><div class="section-heading"><div><span class="eyebrow">SOURCE PREVIEW</span><h3>Generated Python</h3></div><span class="muted">Revision {{ project.current_revision }}</span></div><pre><code>{{ source }}</code></pre></section> } @else { <section class="generated-code-placeholder panel"><span class="placeholder-glyph"><svg lucideClipboardCheck [size]="22"></svg></span><strong>No source snapshot yet</strong><span>Compile this saved revision to inspect generated code here.</span></section> }
        @if (validation(); as result) { <section class="validation-panel panel" [class.valid]="result.valid"><div class="validation-summary"><strong>{{ result.valid ? 'Profile is valid' : 'Blocking findings' }}</strong><span>{{ result.diagnostics.length }} diagnostics · {{ result.total_parameters ?? 0 }} parameters</span><svg lucideCheckCircle2 [size]="16"></svg></div>@for (diagnostic of result.diagnostics; track $index) { <p class="diagnostic-row"><strong>{{ diagnostic.severity }}</strong>{{ diagnostic.message }}</p> }</section> }
        @if (notice()) { <button class="notice workflow-notice" type="button" (click)="notice.set(null)">{{ notice() }}</button> }
        @if (configurationOpen()) { <nw-compilation-configuration-dialog [configuration]="project.graph.model_configuration" [kerasEnabled]="project.keras_enabled" [kerasCatalog]="kerasCatalog()" [nativeCapability]="nativeCapability()" [outputIds]="outputIds()" (close)="closeConfiguration()" (apply)="saveConfiguration($event)" /> }
      </div>
    }
  `,
})
export class ModelCompilationPageComponent {
  readonly state = inject(ModelWorkspaceState)
  private readonly api = inject(ApiService)
  readonly resources = inject(AppResourcesService)
  readonly implementationLabel = implementationLabel
  readonly code = signal<string | null>(null)
  readonly validation = signal<Awaited<ReturnType<ApiService['validate']>> | null>(null)
  readonly notice = signal<string | null>(null)
  readonly busy = signal(false)
  readonly configurationOpen = signal(false)
  readonly configurationBusy = signal(false)
  readonly validationBusy = signal(false)
  readonly kerasCatalog = computed(() => this.resources.compilationCapabilities()?.keras ?? null)
  readonly nativeCapability = computed<NativeCompilationCapability | null>(() => {
    const project = this.state.project()
    return project ? this.resources.compilationCapabilities()?.native[project.execution_backend] ?? null : null
  })
  readonly outputIds = computed(() => this.state.project()?.graph.nodes.filter((node) => node.definition_id.startsWith('output.')).map((node) => node.id) ?? [])

  constructor() {
    effect(() => {
      const project = this.state.project()
      if (project) this.code.set(project.generated_source ?? null)
    })
  }

  async compile(): Promise<void> {
    const project = this.state.project()
    if (!project) return
    this.busy.set(true)
    try {
      const source = await this.api.sourceForProject(project.id)
      this.code.set(source.code)
      const refreshed = await this.state.refresh()
      this.notice.set(`Compiled ${implementationLabel(refreshed.execution_backend, refreshed.keras_enabled)} source for revision ${refreshed.current_revision}.`)
    } catch (reason) { this.notice.set(`Compilation failed: ${reason instanceof Error ? reason.message : String(reason)}`) } finally { this.busy.set(false) }
  }

  async validate(): Promise<void> {
    const project = this.state.project()
    if (!project) return
    this.validationBusy.set(true)
    try { const result = await this.api.validate(project.graph, project.execution_backend, project.keras_enabled); this.validation.set(result); this.notice.set(result.valid ? 'Compilation configuration is valid for this graph.' : 'Compilation configuration has blocking findings.') } catch (reason) { this.notice.set(`Validation failed: ${reason instanceof Error ? reason.message : String(reason)}`) } finally { this.validationBusy.set(false) }
  }

  openConfiguration(): void { if (this.resources.runtimeReady()) this.configurationOpen.set(true) }
  closeConfiguration(): void { if (!this.configurationBusy()) this.configurationOpen.set(false) }

  async saveConfiguration(configuration: ModelConfiguration): Promise<void> {
    const project = this.state.project()
    if (!project || this.configurationBusy()) return
    this.configurationBusy.set(true)
    try {
      const graph = { ...project.graph, model_configuration: structuredClone(configuration) }
      const result = await this.api.validate(graph, project.execution_backend, project.keras_enabled)
      this.validation.set(result)
      const saved = await this.api.updateProject(project.id, { name: project.name, description: project.description, execution_backend: project.execution_backend, keras_enabled: project.keras_enabled, graph, expected_revision: project.current_revision })
      await this.state.refresh()
      this.configurationOpen.set(false)
      this.code.set(null)
      this.notice.set(result.valid ? `Saved compilation settings in revision ${saved.current_revision}.` : `Saved revision ${saved.current_revision} with validation findings.`)
    } catch (reason) { this.notice.set(`Saving compilation settings failed: ${reason instanceof Error ? reason.message : String(reason)}`) } finally { this.configurationBusy.set(false) }
  }

  configurationSummary(project: { execution_backend: 'tensorflow' | 'pytorch' | 'jax'; keras_enabled: boolean; graph: { model_configuration: ModelConfiguration } }): string {
    if (project.keras_enabled) {
      const catalog = this.kerasCatalog()
      const optimizer = catalog?.optimizers.find((item) => item.id === project.graph.model_configuration.keras.optimizer)?.label ?? project.graph.model_configuration.keras.optimizer
      const lossId = project.graph.model_configuration.keras.losses['__default__']
      const loss = lossId ? catalog?.losses.find((item) => item.id === lossId)?.label ?? lossId : 'No loss'
      return `${optimizer} · ${loss}`
    }
    if (project.execution_backend === 'tensorflow') return `XLA ${project.graph.model_configuration.native.tensorflow.jit_compile ? 'on' : 'off'}`
    if (project.execution_backend === 'pytorch') return `torch.compile ${project.graph.model_configuration.native.pytorch.compile_enabled ? 'on' : 'off'}`
    return `${this.nativeCapability()?.compiler ?? 'jax.jit'} ${project.graph.model_configuration.native.jax.jit_enabled ? 'on' : 'off'}`
  }
}
