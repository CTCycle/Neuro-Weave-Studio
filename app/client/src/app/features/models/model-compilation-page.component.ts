import { Component, computed, effect, inject, signal } from '@angular/core'
import { LucideCheckCircle2, LucideClipboardCheck, LucideFileCode2, LucideRefreshCw, LucideSettings2 } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { outputSelectors, summarizeGraphIo } from '../../domain/graph-io'
import type { ModelConfiguration, NativeCompilationCapability } from '../../domain/types'
import { implementationLabel } from '../../domain/types'
import { CompilationStatusBadgeComponent } from '../../shared/components/model-status-badge.component'
import { CompilationConfigurationDialogComponent, type CompilationOutputDescriptor } from './compilation-configuration-dialog.component'
import { ModelWorkspaceState } from './model-workspace.state'

@Component({
  selector: 'nw-model-compilation-page',
  standalone: true,
  imports: [LucideCheckCircle2, LucideClipboardCheck, LucideFileCode2, LucideRefreshCw, LucideSettings2, CompilationStatusBadgeComponent, CompilationConfigurationDialogComponent],
  template: `
    @if (state.project(); as project) {
      <div class="workflow-page compilation-page">
        <div class="workflow-intro"><div><span class="eyebrow">COMPILATION</span><h2>Compile revision {{ project.current_revision }}</h2><p>Configure the saved execution profile, validate every output objective, then generate inspectable source for this exact revision.</p></div><nw-compilation-status-badge [compiled]="!!project.generated_source" [target]="implementationLabel(project.execution_backend, project.keras_enabled)"></nw-compilation-status-badge></div>

        <section class="compilation-context panel" aria-label="Compilation context">
          <div><span>Saved revision</span><strong>r{{ project.current_revision }}</strong></div><div><span>Backend</span><strong>{{ implementationLabel(project.execution_backend, project.keras_enabled) }}</strong></div><div><span>Outputs</span><strong>{{ ioSummary().outputCount }}</strong></div><div><span>Validation</span><strong [class.status-good]="effectiveValid() === true" [class.status-bad]="effectiveValid() === false">{{ validationLabel(project.validation_status) }}</strong></div><div><span>Generated source</span><strong>{{ project.generated_source ? 'Current revision available' : 'Not generated' }}</strong></div>
        </section>

        <div class="compilation-layout">
          <section class="workflow-card panel configuration-card">
            <div class="section-heading"><div><span class="eyebrow">CONFIGURATION</span><h3>{{ configurationSummary(project) }}</h3></div><button class="secondary compact-action" type="button" (click)="openConfiguration()" [disabled]="!resources.runtimeReady() || configurationBusy()"><svg lucideSettings2 [size]="15"></svg>{{ configurationBusy() ? 'Saving…' : 'Configure' }}</button></div>
            <div class="configuration-line"><span>Optimizer</span><strong>{{ optimizerLabel(project) }}</strong></div>
            <div class="output-summary-list">
              @for (modelOutput of ioSummary().outputs; track modelOutput.node_id + ':' + modelOutput.port_id) {
                <article class="output-summary"><div class="output-summary-heading"><strong>{{ modelOutput.name }}</strong>@if (modelOutput.task_hint) { <span>{{ modelOutput.task_hint }}</span> }</div><dl><div><dt>Loss</dt><dd>{{ lossForOutput(project.graph.model_configuration, modelOutput) }}</dd></div><div><dt>Metrics</dt><dd>{{ metricsForOutput(project.graph.model_configuration, modelOutput) }}</dd></div></dl></article>
              }
              @if (!ioSummary().outputs.length) { <div class="empty-output-summary"><strong>No effective outputs</strong><span>Complete the graph before configuring objectives.</span></div> }
            </div>
          </section>

          <section class="workflow-card panel readiness-card">
            <div class="section-heading"><div><span class="eyebrow">READINESS</span><h3>{{ nextAction(project) }}</h3></div><svg lucideClipboardCheck [size]="18"></svg></div>
            <div class="readiness-steps"><div [class.complete]="effectiveValid() === true"><span>1</span><div><strong>Validate profile</strong><small>{{ effectiveValid() === true ? 'Current configuration is valid.' : effectiveValid() === false ? 'Blocking diagnostics need attention.' : 'Run validation against the saved graph.' }}</small></div></div><div [class.complete]="!!project.generated_source"><span>2</span><div><strong>Generate source</strong><small>{{ project.generated_source ? 'Source snapshot exists for the current saved revision.' : 'Generate after validation succeeds.' }}</small></div></div></div>
            <div class="workflow-card-actions"><button class="secondary" type="button" (click)="validate()" [disabled]="validationBusy()"><svg lucideClipboardCheck [size]="16"></svg>{{ validationBusy() ? 'Validating…' : 'Validate profile' }}</button><button class="primary" type="button" (click)="compile()" [disabled]="busy() || effectiveValid() === false">@if (busy()) { <svg class="spin" lucideRefreshCw [size]="16"></svg> } @else { <svg lucideCheckCircle2 [size]="16"></svg> }{{ busy() ? 'Compiling…' : project.generated_source ? 'Recompile source' : 'Compile source' }}</button></div>
          </section>
        </div>

        @if (validation(); as result) { <section class="validation-panel panel" [class.valid]="result.valid"><div class="validation-summary"><strong>{{ result.valid ? 'Profile is valid' : 'Blocking findings' }}</strong><span>{{ result.diagnostics.length }} diagnostics · {{ result.total_parameters ?? 0 }} parameters</span><svg lucideCheckCircle2 [size]="16"></svg></div>@for (diagnostic of result.diagnostics; track $index) { <p class="diagnostic-row"><strong>{{ diagnostic.severity }}</strong>{{ diagnostic.node_id ? diagnostic.node_id + ': ' : '' }}{{ diagnostic.message }}</p> }</section> }

        @if (code(); as source) { <section class="generated-code panel"><div class="section-heading"><div><span class="eyebrow">GENERATED SOURCE</span><h3>Python · revision {{ project.current_revision }}</h3></div><span class="source-state"><svg lucideFileCode2 [size]="14"></svg>Current snapshot</span></div><pre><code>{{ source }}</code></pre></section> } @else { <section class="generated-code-placeholder panel"><span class="placeholder-glyph"><svg lucideFileCode2 [size]="22"></svg></span><strong>No generated source</strong><span>Validate the profile, then compile this saved revision.</span></section> }
        @if (notice()) { <button class="notice workflow-notice" type="button" (click)="notice.set(null)">{{ notice() }}</button> }
        @if (configurationOpen()) { <nw-compilation-configuration-dialog [configuration]="project.graph.model_configuration" [kerasEnabled]="project.keras_enabled" [kerasCatalog]="kerasCatalog()" [nativeCapability]="nativeCapability()" [outputs]="compilationOutputs()" (close)="closeConfiguration()" (apply)="saveConfiguration($event)" /> }
      </div>
    }
  `,
  styles: [`
    .compilation-context { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); padding:10px 14px; gap:0; }
    .compilation-context > div { min-width:0; display:grid; gap:4px; padding:2px 12px; border-left:1px solid var(--border); }
    .compilation-context > div:first-child { border-left:0; padding-left:0; }
    .compilation-context span { color:var(--muted); font:600 9px/1 'IBM Plex Mono', Consolas, monospace; text-transform:uppercase; letter-spacing:.06em; }
    .compilation-context strong { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:11px; }
    .status-good { color:var(--cyan); } .status-bad { color:var(--accent-bright); }
    .compilation-layout { display:grid; grid-template-columns:minmax(0,1.45fr) minmax(310px,.75fr); gap:14px; }
    .configuration-card, .readiness-card { min-width:0; }
    .compact-action { min-height:31px; padding:5px 9px; }
    .configuration-line { display:flex; justify-content:space-between; gap:14px; padding:9px 0; border-top:1px solid var(--border); border-bottom:1px solid var(--border); font-size:11px; }
    .configuration-line span { color:var(--muted); }
    .output-summary-list { display:grid; gap:0; margin-top:5px; }
    .output-summary { padding:11px 0; border-bottom:1px solid color-mix(in srgb, var(--border) 72%, transparent); }
    .output-summary:last-child { border-bottom:0; }
    .output-summary-heading { display:flex; justify-content:space-between; gap:10px; align-items:center; }
    .output-summary-heading > span { color:var(--cyan); font:600 9px/1 'IBM Plex Mono', Consolas, monospace; text-transform:uppercase; }
    .output-summary dl { display:grid; grid-template-columns:1fr 1.4fr; gap:12px; margin:8px 0 0; }
    .output-summary dl > div { min-width:0; }
    .output-summary dt { color:var(--muted); font:600 9px/1 'IBM Plex Mono', Consolas, monospace; text-transform:uppercase; }
    .output-summary dd { margin:4px 0 0; color:var(--text); font-size:11px; overflow-wrap:anywhere; }
    .empty-output-summary { display:grid; gap:4px; padding:14px 0; color:var(--muted); }
    .readiness-steps { display:grid; gap:12px; margin:8px 0 16px; }
    .readiness-steps > div { display:grid; grid-template-columns:26px 1fr; gap:9px; align-items:start; }
    .readiness-steps > div > span { display:grid; place-items:center; width:24px; height:24px; border:1px solid var(--border); border-radius:50%; color:var(--muted); font:600 10px/1 'IBM Plex Mono', Consolas, monospace; }
    .readiness-steps > div.complete > span { border-color:var(--cyan); color:var(--cyan); }
    .readiness-steps strong, .readiness-steps small { display:block; }
    .readiness-steps small { margin-top:3px; color:var(--muted); line-height:1.4; }
    .source-state { display:inline-flex; align-items:center; gap:5px; color:var(--cyan); font:600 10px/1 'IBM Plex Mono', Consolas, monospace; }
    @media (max-width: 1100px) { .compilation-context { grid-template-columns:repeat(3,minmax(0,1fr)); row-gap:10px; } .compilation-context > div:nth-child(4) { border-left:0; padding-left:0; } .compilation-layout { grid-template-columns:1fr; } }
    @media (max-width: 700px) { .compilation-context { grid-template-columns:1fr 1fr; } .compilation-context > div:nth-child(odd) { border-left:0; padding-left:0; } .output-summary dl { grid-template-columns:1fr; } }
  `],
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
  readonly nativeCapability = computed<NativeCompilationCapability | null>(() => { const project = this.state.project(); return project ? this.resources.compilationCapabilities()?.native[project.execution_backend] ?? null : null })
  readonly ioSummary = computed(() => { const project = this.state.project(); return project ? summarizeGraphIo(project.graph, this.resources.manifests()) : { inputCount: 0, outputCount: 0, inputs: [], outputs: [] } })
  readonly compilationOutputs = computed<CompilationOutputDescriptor[]>(() => this.ioSummary().outputs.map((item) => ({ id: `${item.node_id}:${item.port_id}`, label: item.name, task_hint: item.task_hint })))

  constructor() {
    void this.resources.loadRuntimeMetadata()
    void this.resources.loadCatalog()
    effect(() => { const project = this.state.project(); if (project) { this.code.set(project.generated_source ?? null); this.validation.set(null) } })
  }

  async compile(): Promise<void> {
    const project = this.state.project(); if (!project) return
    this.busy.set(true)
    try { const source = await this.api.sourceForProject(project.id); this.code.set(source.code); const refreshed = await this.state.refresh(); this.notice.set(`Compiled ${implementationLabel(refreshed.execution_backend, refreshed.keras_enabled)} source for revision ${refreshed.current_revision}.`) } catch (reason) { this.notice.set(`Compilation failed: ${reason instanceof Error ? reason.message : String(reason)}`) } finally { this.busy.set(false) }
  }

  async validate(): Promise<void> {
    const project = this.state.project(); if (!project) return
    this.validationBusy.set(true)
    try { const result = await this.api.validate(project.graph, project.execution_backend, project.keras_enabled); this.validation.set(result); this.notice.set(result.valid ? 'Compilation configuration is valid for this graph.' : 'Compilation configuration has blocking findings.') } catch (reason) { this.notice.set(`Validation failed: ${reason instanceof Error ? reason.message : String(reason)}`) } finally { this.validationBusy.set(false) }
  }

  openConfiguration(): void { if (this.resources.runtimeReady()) this.configurationOpen.set(true) }
  closeConfiguration(): void { if (!this.configurationBusy()) this.configurationOpen.set(false) }

  async saveConfiguration(configuration: ModelConfiguration): Promise<void> {
    const project = this.state.project(); if (!project || this.configurationBusy()) return
    this.configurationBusy.set(true)
    try {
      const graph = { ...project.graph, model_configuration: structuredClone(configuration) }
      const result = await this.api.validate(graph, project.execution_backend, project.keras_enabled)
      this.validation.set(result)
      if (!result.valid) { this.notice.set('Configuration was not saved. Resolve the output-specific validation findings first.'); return }
      const saved = await this.api.updateProject(project.id, { name: project.name, description: project.description, execution_backend: project.execution_backend, keras_enabled: project.keras_enabled, graph, expected_revision: project.current_revision })
      await this.state.refresh(); this.configurationOpen.set(false); this.code.set(null); this.notice.set(`Saved validated compilation settings in revision ${saved.current_revision}.`)
    } catch (reason) { this.notice.set(`Saving compilation settings failed: ${reason instanceof Error ? reason.message : String(reason)}`) } finally { this.configurationBusy.set(false) }
  }

  effectiveValid(): boolean | null { const validation = this.validation(); if (validation) return validation.valid; const project = this.state.project(); return project?.validation_status === 'valid' ? true : project?.validation_status === 'invalid' ? false : null }
  validationLabel(status: string): string { const value = this.effectiveValid(); return value === true ? 'Valid' : value === false ? 'Blocking findings' : status || 'Not checked' }
  nextAction(project: { generated_source?: string | null }): string { return this.effectiveValid() === false ? 'Resolve blocking findings' : project.generated_source ? 'Review or recompile source' : this.effectiveValid() === true ? 'Generate source' : 'Validate the profile' }

  optimizerLabel(project: { keras_enabled: boolean; graph: { model_configuration: ModelConfiguration } }): string { const id = project.keras_enabled ? project.graph.model_configuration.keras.optimizer : project.graph.model_configuration.native.pytorch.optimizer; const catalog = project.keras_enabled ? this.kerasCatalog() : this.nativeCapability()?.catalog; return catalog?.optimizers.find((item) => item.id === id)?.label ?? id }
  lossForOutput(configuration: ModelConfiguration, modelOutput: ReturnType<ModelCompilationPageComponent['ioSummary']>['outputs'][number]): string { const project = this.state.project(); if (!project) return 'Not configured'; const mapping = project.keras_enabled ? configuration.keras.losses : configuration.native.pytorch.losses; const id = this.resolveMapping(mapping, modelOutput) ?? 'Not configured'; const catalog = project.keras_enabled ? this.kerasCatalog() : this.nativeCapability()?.catalog; return catalog?.losses.find((item) => item.id === id)?.label ?? id }
  metricsForOutput(configuration: ModelConfiguration, modelOutput: ReturnType<ModelCompilationPageComponent['ioSummary']>['outputs'][number]): string { const project = this.state.project(); if (!project) return 'None'; const mapping = project.keras_enabled ? configuration.keras.metrics : configuration.native.pytorch.metrics; const ids = this.resolveMapping(mapping, modelOutput) ?? []; const catalog = project.keras_enabled ? this.kerasCatalog() : this.nativeCapability()?.catalog; return ids.length ? ids.map((id) => catalog?.metrics.find((item) => item.id === id)?.label ?? id).join(', ') : 'None' }
  configurationSummary(project: { execution_backend: 'tensorflow' | 'pytorch' | 'jax'; keras_enabled: boolean }): string { return project.keras_enabled ? 'Keras compile profile' : project.execution_backend === 'pytorch' ? 'Native PyTorch profile' : `Native ${project.execution_backend.toUpperCase()} profile` }

  private resolveMapping<T>(mapping: Record<string, T>, modelOutput: ReturnType<ModelCompilationPageComponent['ioSummary']>['outputs'][number]): T | undefined { const project = this.state.project(); if (!project) return mapping['__default__']; for (const selector of outputSelectors(modelOutput, project.graph)) { if (mapping[selector] !== undefined) return mapping[selector] } return mapping['__default__'] }
}
