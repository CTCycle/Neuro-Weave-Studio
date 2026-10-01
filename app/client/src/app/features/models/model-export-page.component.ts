// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, computed, effect, inject, signal } from '@angular/core'
import { LucideAlertTriangle, LucideCheckCircle2, LucideDownload, LucideRefreshCw, LucideSquare } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import type { ExecutionBackendStatus, ExportCapability, ExportCategory, ExportFormat, ExportOptions, JobResponse, ProjectDocument, RuntimeDevice, WorkerConfiguration } from '../../domain/types'
import { pollJobUntilTerminal, terminalJobStates } from './job-polling'
import { ModelWorkspaceState } from './model-workspace.state'

const initialOptions: ExportOptions = { opset_version: 18, dynamic_batch: true, external_data: 'auto', max_shard_size_gb: null }
const defaultOnnxOpsets = [17, 18]
const categoryOrder: ExportCategory[] = ['project', 'source', 'native', 'interchange', 'checkpoint']
const categoryLabels: Record<ExportCategory, string> = { project: 'Project', source: 'Source', native: 'Native', interchange: 'Interchange', checkpoint: 'Weights / Checkpoint' }
@Component({
  selector: 'nw-model-export-page',
  standalone: true,
  imports: [LucideAlertTriangle, LucideCheckCircle2, LucideDownload, LucideRefreshCw, LucideSquare],
  template: `
    @if (state.project(); as project) {
      <div class="workflow-page export-page"><section class="model-export-page panel" aria-labelledby="model-export-title"><header class="model-export-header"><div><span class="eyebrow">MODEL EXPORT CENTER</span><h2 id="model-export-title">Export {{ project.name }}</h2><p>Revision {{ capabilitiesRevision() ?? project.current_revision }} · {{ project.execution_backend }} · {{ project.keras_enabled ? 'Keras profile' : 'native profile' }}</p></div></header>
        @if (loading()) { <div class="model-export-state"><svg class="spin" lucideRefreshCw [size]="20"></svg><span>Probing export capabilities…</span></div> } @else if (error(); as message) { <div class="model-export-state export-error" role="alert"><svg lucideAlertTriangle [size]="18"></svg><div><strong>Capabilities unavailable</strong><span>{{ message }}</span><button class="secondary" type="button" (click)="load()"><svg lucideRefreshCw [size]="14"></svg>Retry</button></div></div> } @else {
          <div class="model-export-body"><div class="model-export-catalog">@for (group of groupedCapabilities(); track group.category) { <section class="export-category" [attr.aria-labelledby]="'export-category-' + group.category"><div class="export-category-heading"><div><span class="eyebrow">EXPORT METHODS</span><h3 [id]="'export-category-' + group.category">{{ categoryLabels[group.category] }}</h3></div><span class="muted">{{ availableCount(group.formats) }}/{{ group.formats.length }} available</span></div><div class="export-format-list">@for (capability of group.formats; track capability.format) { <button type="button" class="export-format-option" [class.selected]="selectedFormat() === capability.format" [class.unavailable]="!capability.available" [disabled]="!capability.available || busy()" [title]="capability.available ? capability.compatibility : capability.reason || 'Unavailable'" (click)="selectFormat(capability)"><span class="export-format-top"><strong>{{ capability.label }}</strong>@if (capability.available) { <svg lucideCheckCircle2 [size]="14"></svg> } @else { <svg lucideAlertTriangle [size]="14"></svg> }</span><span class="export-badge-row">@for (badge of badges(capability); track badge) { <span class="export-badge">{{ badge }}</span> }</span><span class="export-format-description">{{ capability.available ? capability.compatibility : capability.reason || 'Unavailable for this profile.' }}</span></button> }</div></section> } @if (!capabilities().length) { <div class="model-export-state"><svg lucideAlertTriangle [size]="18"></svg><span>No export formats were returned for this project.</span></div> }</div>
            <aside class="model-export-config panel"><div class="section-heading"><div><span class="eyebrow">EXPORT SETTINGS</span><h3>{{ selectedCapability()?.label || 'Select a format' }}</h3></div><svg lucideDownload [size]="17"></svg></div>@if (selectedCapability(); as capability) { <label class="export-field">Filename stem<input [value]="filenameStem()" maxlength="160" (input)="filenameStem.set($any($event.target).value)" /></label><span class="export-help">The server rejects paths and supplied extensions, then derives the final extension.</span><div class="export-preview-name"><span>Download name</span><strong>{{ displayedFilename(capability, project) }}</strong></div>@if (capability.format === 'onnx') { <div class="export-option-stack">@if (hasOption(capability, 'opset_version')) { <label class="export-field">{{ optionLabel(capability, 'opset_version', 'ONNX opset') }}<select [value]="options().opset_version" (change)="updateOptions({ opset_version: +$any($event.target).value })">@for (choice of optionChoices(capability, 'opset_version', [17, 18, 19, 20]); track $index) { <option [value]="choice">{{ choice }}</option> }</select></label> }@if (hasOption(capability, 'dynamic_batch')) { <label class="export-checkbox"><input type="checkbox" [checked]="options().dynamic_batch" (change)="updateOptions({ dynamic_batch: $any($event.target).checked })" /><span>Dynamic batch dimensions</span></label> }@if (hasOption(capability, 'external_data')) { <label class="export-field">{{ optionLabel(capability, 'external_data', 'External data') }}<select [value]="options().external_data" (change)="updateOptions({ external_data: $any($event.target).value })">@for (choice of optionChoices(capability, 'external_data', ['auto', 'always', 'never']); track $index) { <option [value]="choice">{{ choice }}</option> }</select></label> }<span class="export-help">Named inputs and outputs are preserved. External data is packaged as a deterministic ZIP when forced.</span></div> }@if (hasOption(capability, 'max_shard_size_gb')) { <label class="export-field">Maximum shard size (GB)<input type="number" min="0.01" max="100" step="0.01" [value]="options().max_shard_size_gb ?? ''" (input)="updateOptions({ max_shard_size_gb: $any($event.target).value ? +$any($event.target).value : null })" /></label> }<div class="export-detail-list"><div><span>Payload</span><strong>{{ payloadLabel(capability.payload) }}</strong></div><div><span>Conversion</span><strong>{{ conversionLabel(capability.conversion) }}</strong></div><div><span>Verification</span><strong>Structural · reload · forward pass</strong></div></div>@if (!isDirect(capability)) { <div class="export-worker-summary"><span>Worker</span><strong>{{ workerForProject(project) ? workerForProject(project)?.api + ' · ' + workerForProject(project)?.backend + ' · ' + workerForProject(project)?.device_id : 'No operational worker' }}</strong></div> } } @else { <p class="muted">Select an available capability to configure an export.</p> }</aside>
          </div>
          @if (job(); as current) { <div class="model-export-job" [class.failed]="current.state === 'failed'" [class.succeeded]="current.state === 'succeeded'" role="status"><div><span class="eyebrow">EXPORT JOB · {{ current.state }}</span><strong>{{ current.progress.message }}</strong><small>{{ current.progress.phase }}{{ current.progress.percent === null || current.progress.percent === undefined ? '' : ' · ' + current.progress.percent + '%' }}{{ current.artifact ? ' · ' + current.artifact.filename : '' }}</small></div>@if (exportActive()) { <button class="secondary" type="button" (click)="stopExport()" [disabled]="cancelling()"><svg lucideSquare [size]="14"></svg>{{ cancelling() ? 'Cancelling…' : 'Cancel export' }}</button> }</div> } @if (notice()) { <div class="model-export-notice" role="alert">{{ notice() }}</div> }
        }
        <footer class="model-export-footer dialog-actions"><span class="model-export-footer-note">{{ selectedCapability()?.limitations?.[0] || 'Unsupported formats remain visible with their reason.' }}</span><button class="primary" type="button" (click)="exportSelected(project)" [disabled]="!canExport(project)"><svg [class.spin]="busy()" [attr.data-icon]="busy() ? 'loading' : 'download'" lucideDownload [size]="15"></svg>{{ busy() ? 'Exporting…' : job()?.state === 'failed' ? 'Retry export' : 'Export artifact' }}</button></footer>
      </section></div>
    }
  `,
})
export class ModelExportPageComponent {
  readonly state = inject(ModelWorkspaceState)
  private readonly api = inject(ApiService)
  private readonly resources = inject(AppResourcesService)
  private readonly destroyRef = inject(DestroyRef)
  readonly capabilities = signal<ExportCapability[]>([])
  readonly capabilitiesRevision = signal<number | null>(null)
  readonly selectedFormat = signal<ExportFormat | null>(null)
  readonly filenameStem = signal('model')
  readonly options = signal<ExportOptions>(initialOptions)
  readonly loading = signal(true)
  readonly error = signal<string | null>(null)
  readonly notice = signal<string | null>(null)
  readonly job = signal<JobResponse | null>(null)
  readonly busy = signal(false)
  readonly cancelling = signal(false)
  private readonly discoveryAbort = new AbortController()
  readonly groupedCapabilities = computed(() => categoryOrder.map((category) => ({ category, formats: this.capabilities().filter((item) => item.category === category) })).filter((group) => group.formats.length))
  readonly categoryLabels = categoryLabels
  readonly selectedCapability = (): ExportCapability | null => this.capabilities().find((item) => item.format === this.selectedFormat()) ?? null
  readonly exportActive = computed(() => { const current = this.job(); return Boolean(current && !terminalJobStates.has(current.state)) })

  constructor() {
    void this.resources.loadRuntimeMetadata()
    this.destroyRef.onDestroy(() => this.discoveryAbort.abort())
    effect(() => {
      const project = this.state.project()
      if (!project) return
      const api = project.keras_enabled ? 'keras' : 'native'
      void this.resources.discoverRuntime(project.execution_backend, api, false, this.discoveryAbort.signal).catch((reason) => {
        if (!this.destroyRef.destroyed) this.notice.set(`Runtime discovery failed: ${reason instanceof Error ? reason.message : String(reason)}`)
      })
    })
    effect(() => { const project = this.state.project(); if (project) { this.filenameStem.set(project.name); void this.load() } })
  }

  async load(): Promise<void> {
    const project = this.state.project(); if (!project) return
    this.loading.set(true); this.error.set(null)
    try {
      const response = await this.api.exportCapabilities(project.id)
      if (this.destroyRef.destroyed) return
      this.capabilities.set(response.formats); this.capabilitiesRevision.set(response.project_revision)
      const current = this.selectedFormat(); this.selectedFormat.set(current && response.formats.some((item) => item.format === current && item.available) ? current : response.formats.find((item) => item.available)?.format ?? null)
      const capability = response.formats.find((item) => item.format === this.selectedFormat())
      if (capability) this.options.set(optionsFor(capability))
    } catch (reason) { if (!this.destroyRef.destroyed) this.error.set(reason instanceof Error ? reason.message : String(reason)) } finally { if (!this.destroyRef.destroyed) this.loading.set(false) }
  }

  selectFormat(capability: ExportCapability): void { this.selectedFormat.set(capability.format); this.options.set(optionsFor(capability)); this.notice.set(null); this.job.set(null) }
  availableCount(formats: ExportCapability[]): number { return formats.filter((item) => item.available).length }
  badges(capability: ExportCapability): string[] { return [capability.conversion === 'native' ? 'Native' : capability.conversion === 'bridge' ? 'Converted' : null, capability.payload === 'full_model' ? 'Full model' : capability.payload === 'weights_only' ? 'Weights-only' : null, capability.inference_only ? 'No trained weights' : null, capability.training_state ? 'Training state unavailable' : null].filter((item): item is string => Boolean(item)) }
  hasOption(capability: ExportCapability, id: string): boolean { return capability.options.some((item) => item.id === id) }
  optionLabel(capability: ExportCapability, id: string, fallback: string): string { return capability.options.find((item) => item.id === id)?.label || fallback }
  optionChoices(capability: ExportCapability, id: string, fallback: unknown[]): unknown[] { const choices = capability.options.find((item) => item.id === id)?.choices; return choices?.length ? choices : id === 'opset_version' ? defaultOnnxOpsets : fallback }
  updateOptions(patch: Partial<ExportOptions>): void { this.options.update((current) => ({ ...current, ...patch })) }
  isDirect(capability: ExportCapability): boolean { return capability.format === 'project_bundle' || capability.format === 'source' }
  displayedFilename(capability: ExportCapability, project: ProjectDocument): string { return `${safeFilename(this.filenameStem() || project.name)}${capability.format === 'onnx' && this.options().external_data === 'always' ? '.onnx.zip' : extensionFor(capability)}` }
  payloadLabel(value: ExportCapability['payload']): string { return value === 'full_model' ? 'Full model' : value === 'weights_only' ? 'Weights only' : value === 'source' ? 'Generated source' : 'Project graph' }
  conversionLabel(value: ExportCapability['conversion']): string { return value === 'native' ? 'Native' : value === 'bridge' ? 'Verified bridge' : value === 'keras' ? 'Keras adapter' : 'Canonical' }

  workerForProject(project: ProjectDocument): WorkerConfiguration | null { return workerForProject(project, this.resources.backendStatuses(), this.resources.runtimePreferences(), this.resources) }
  canExport(_project: ProjectDocument): boolean { const capability = this.selectedCapability(); return Boolean(capability?.available && !this.busy()) }

  async exportSelected(project: ProjectDocument): Promise<void> {
    const capability = this.selectedCapability(); const format = this.selectedFormat(); if (!capability?.available || !format || this.busy()) return
    this.busy.set(true); this.notice.set(null); this.job.set(null)
    try {
      if (format === 'project_bundle') { const bundle = await this.api.exportProject(project.id); this.download(new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' }), this.displayedFilename(capability, project)); return }
      if (format === 'source') { const source = await this.api.sourceForProject(project.id); this.download(new Blob([source.code], { type: 'text/x-python' }), this.displayedFilename(capability, project)); return }
      let worker = this.workerForProject(project)
      if (!worker) {
        const api = project.keras_enabled ? 'keras' : 'native'
        await this.resources.discoverRuntime(project.execution_backend, api, false, this.discoveryAbort.signal)
        worker = this.workerForProject(project)
      }
      if (!worker) throw new Error('No operational worker is available for this project profile. Check the backend in Environment and retry.')
      let current = await this.api.submitModelExport(project.id, format, this.filenameStem().trim() || project.name, this.options(), worker); this.job.set(current)
       if (!terminalJobStates.has(current.state)) current = await pollJobUntilTerminal(current, (jobId) => this.api.getJob(jobId), (next) => this.job.set(next), { intervalMs: 400, timeoutMs: 60_000, isDestroyed: () => this.destroyRef.destroyed })
       if (this.destroyRef.destroyed) return
       if (!terminalJobStates.has(current.state)) { this.notice.set(`${capability.label} did not reach a terminal state within 60 seconds. The worker may still be completing; refresh the page to inspect it.`); return }
       if (current.state !== 'succeeded') { this.notice.set(`${capability.label} ended with ${current.state}: ${current.error?.message || 'the worker did not publish an artifact'}`); return }
      const blob = await this.api.jobArtifact(current.id); this.download(blob, current.artifact?.filename || this.displayedFilename(capability, project)); this.notice.set(`Downloaded verified ${current.artifact?.filename || this.displayedFilename(capability, project)}.`)
    } catch (reason) { if (!this.destroyRef.destroyed) this.notice.set(`${capability.label} failed: ${reason instanceof Error ? reason.message : String(reason)}`) } finally { if (!this.destroyRef.destroyed) this.busy.set(false) }
  }

  async stopExport(): Promise<void> { const current = this.job(); if (!current || !this.exportActive() || this.cancelling()) return; this.cancelling.set(true); try { this.job.set(await this.api.stopJob(current.id)); this.notice.set('Cancellation requested. The worker will finish cleanup before the lease is released.') } catch (reason) { this.notice.set(`Cancellation failed: ${reason instanceof Error ? reason.message : String(reason)}`) } finally { this.cancelling.set(false) } }
  private download(blob: Blob, filename: string): void { const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url) }
}

function optionsFor(capability: ExportCapability): ExportOptions { const next = { ...initialOptions }; const value = (id: string) => capability.options.find((item) => item.id === id)?.default; if (typeof value('opset_version') === 'number') next.opset_version = value('opset_version') as number; if (typeof value('dynamic_batch') === 'boolean') next.dynamic_batch = value('dynamic_batch') as boolean; const external = value('external_data'); if (external === 'auto' || external === 'always' || external === 'never') next.external_data = external; return next }
function extensionFor(capability: ExportCapability): string { return capability.extension.startsWith('.') ? capability.extension : `.${capability.extension}` }
function safeFilename(value: string): string { return value.replaceAll(/[^a-zA-Z0-9_-]/g, '_') || 'model' }
function workerForProject(project: ProjectDocument, statuses: ExecutionBackendStatus[], preferences: Partial<Record<ProjectDocument['execution_backend'], WorkerConfiguration>>, resources: AppResourcesService): WorkerConfiguration | null {
  const status = statuses.find((item) => item.id === project.execution_backend); if (!status?.available) return null
  const api: WorkerConfiguration['api'] = project.keras_enabled ? 'keras' : 'native'; if (!resources.runtimeCapabilityReady(project.execution_backend, api)) return null
  const preference = preferences[project.execution_backend]; const devices = status.devices?.length ? status.devices : [fallbackDevice()]; const preferred = preference ? devices.find((item) => item.device_id === preference.device_id && item.available) : null; const device = preferred ?? devices.find((item) => item.available); if (!device) return null
  const preferredPolicy = preference?.memory_policy; const memoryPolicy = preferredPolicy && device.supported_memory_policies.includes(preferredPolicy.kind) ? preferredPolicy : policyForDevice(device)
  return { api, backend: project.execution_backend, device_id: device.device_id, memory_policy: memoryPolicy, precision: preference?.precision ?? 'float32' }
}
function fallbackDevice(): RuntimeDevice { return { device_id: 'cpu', lease_key: 'cpu', kind: 'cpu', name: 'CPU', accelerator: 'cpu', available: true, supported_memory_policies: ['default'] } }
function policyForDevice(device: RuntimeDevice): WorkerConfiguration['memory_policy'] { const kind = device.supported_memory_policies[0] ?? 'default'; if (kind === 'tensorflow_limit_mb') return { kind, limit_mb: 1024 }; if (kind === 'jax_preallocate_fraction' || kind === 'pytorch_fraction') return { kind, fraction: 0.5 }; return { kind } }
