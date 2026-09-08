import { Component, DestroyRef, computed, inject, signal } from '@angular/core'
import { takeUntilDestroyed } from '@angular/core/rxjs-interop'
import { LucideAlertTriangle, LucideCheckCircle2, LucideCpu, LucidePackageX, LucideSquare, LucideTerminal } from '@lucide/angular'
import { from, timer } from 'rxjs'
import { switchMap } from 'rxjs/operators'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { backendSupportsApi, preferredRuntimeApi } from '../../core/state/runtime-selection'
import type { ExecutionBackendStatus, JobResponse, MemoryPolicyKind, RuntimeApi, RuntimeApiStatus, RuntimeDevice, WorkerConfiguration } from '../../domain/types'

const backendLabels: Record<ExecutionBackendStatus['id'], string> = { tensorflow: 'TensorFlow', pytorch: 'PyTorch', jax: 'JAX' }
const apiLabels: Record<RuntimeApi, string> = { keras: 'Keras', native: 'Native framework' }
const fallbackRuntimeApis: RuntimeApiStatus[] = [{ id: 'keras', available: true }, { id: 'native', available: false, reason_code: 'runtime_unavailable' }]

@Component({
  selector: 'nw-backend-status-panel',
  standalone: true,
  imports: [LucideAlertTriangle, LucideCheckCircle2, LucideCpu, LucidePackageX, LucideSquare, LucideTerminal],
  template: `
    @if (resources.runtimeReady() && resources.backendStatuses().length) {
      <section class="backend-status-panel" aria-label="Compute environment status">
        <div class="backend-status-heading"><div><span class="eyebrow">COMPUTE ENVIRONMENT</span><h2>Execution runtime</h2><p>{{ operationalCount() ? operationalCount() + ' operational backend' + (operationalCount() === 1 ? '' : 's') : 'No operational backend' }} · isolated runtime discovery · one lease per physical device</p></div><span class="backend-status-command"><svg lucideTerminal [size]="13"></svg>isolated worker controller</span></div>
        <div class="runtime-api-controls" aria-label="Runtime API">@for (status of runtimeApiStatuses(); track status.id) { <button type="button" class="runtime-api-option" [class.selected]="runtimeSelection()?.api === status.id" [disabled]="!status.available" (click)="selectRuntimeApi(status.id)"><strong>{{ apiLabels[status.id] }}</strong><span>{{ status.available ? 'Operational' : status.state === 'checking' ? 'Checking' : status.state === 'unchecked' ? 'Not checked' : status.reason_code === 'not_implemented' ? 'Deferred' : 'Unavailable' }}</span></button> }</div>
        <div class="backend-status-grid">@for (status of resources.backendStatuses(); track status.id) { <div class="backend-status-card-container" [class.selected]="selectedBackend() === status.id"><button type="button" class="backend-status-card" [class.usable]="status.available" [class.unavailable]="!status.available && status.installed" [class.missing]="!status.available && !status.installed" [class.selected]="selectedBackend() === status.id" [disabled]="!status.available" [attr.aria-pressed]="selectedBackend() === status.id" (click)="selectBackend(status)"><div class="backend-status-card-top"><span class="backend-status-name"><svg lucideCpu [size]="14"></svg>{{ backendLabels[status.id] }}</span><span class="backend-status-state">@if (status.available) { <svg lucideCheckCircle2 [size]="13"></svg>Usable } @else if (status.discovery_state === 'checking') { <svg lucideAlertTriangle [size]="13"></svg>Checking } @else if (status.discovery_state === 'unchecked') { <svg lucideAlertTriangle [size]="13"></svg>Not checked } @else if (status.installed) { <svg lucideAlertTriangle [size]="13"></svg>Installed } @else { <svg lucidePackageX [size]="13"></svg>Missing }</span></div><strong>{{ status.version ? 'v' + status.version : 'Optional framework' }}</strong><span>{{ backendDetail(status) }}</span></button>@if (!status.available && status.discovery_state !== 'checking') { <button class="secondary runtime-check" type="button" (click)="check(status, $event)">{{ status.discovery_state === 'unchecked' ? 'Check runtime' : 'Refresh check' }}</button> } @if (status.available) { <div class="runtime-fields"><label>Device<select [value]="selectedDevice(status).device_id" (change)="selectDevice(status, $event)">@for (device of devicesFor(status); track device.device_id) { <option [value]="device.device_id">{{ device.device_id }} · {{ device.name }}</option> }</select></label><label>Memory policy<select [value]="selectedPolicy(status).kind" (change)="selectPolicy(status, $event)">@for (policy of selectedDevice(status).supported_memory_policies; track policy) { <option [value]="policy">{{ policy }}</option> }</select></label>@if (selectedPolicy(status).kind === 'tensorflow_limit_mb') { <label>Limit (MB)<input type="number" min="1" [value]="selectedPolicy(status).limit_mb ?? 1024" (change)="updatePolicyValue(status, $event)" /></label> } @if (selectedPolicy(status).kind === 'jax_preallocate_fraction' || selectedPolicy(status).kind === 'pytorch_fraction') { <label>Fraction<input type="number" min="0.01" max="1" step="0.01" [value]="selectedPolicy(status).fraction ?? 0.5" (change)="updatePolicyValue(status, $event)" /></label> }</div> }</div> }</div>
        @if (pendingSelection()) { <div class="selection-prompt panel" role="alert"><strong>This runtime is used by an active worker.</strong><span>Choose whether to stop and reap it before switching, or keep it unchanged and use the new selection only for subsequent jobs.</span><div class="dialog-actions"><button class="secondary" type="button" (click)="keepRunning()" [disabled]="stopping()">Keep running</button><button class="danger-action" type="button" (click)="stopAndSwitch()" [disabled]="stopping()">{{ stopping() ? 'Reaping…' : 'Stop and switch' }}</button></div></div> }
        @if (jobsError()) { <div class="selection-prompt panel" role="status"><strong>Worker status unavailable.</strong><span>{{ jobsError() }} Last confirmed worker state is preserved below.</span></div> }
        @if (!activeJobs().length && !jobsError()) { <div class="active-jobs panel"><div><span class="eyebrow">ACTIVE WORKERS</span><strong>No active workers</strong></div><span class="muted">Jobs continue across navigation and are reaped before their device lease is released.</span></div> } @else if (activeJobs().length) { <div class="active-jobs panel"><div class="section-heading"><div><span class="eyebrow">ACTIVE WORKERS</span><h3>{{ activeJobs().length }} active job{{ activeJobs().length === 1 ? '' : 's' }}</h3></div><svg lucideTerminal [size]="16"></svg></div>@for (job of activeJobs(); track job.id) { <div class="active-job-row"><div><strong>{{ job.operation }}</strong><span>{{ job.worker.backend }} · {{ job.worker.device_id }} · PID {{ job.pid ?? job.device_lease.pid ?? 'starting' }}</span></div><div><span>{{ job.state }}</span><small>{{ job.progress.phase }} · {{ memoryText(job) }}</small></div><button type="button" class="icon-button" title="Stop worker" (click)="stopJob(job)"><svg lucideSquare [size]="14"></svg></button></div> }</div> }
      </section>
    }
  `,
})
export class BackendStatusPanelComponent {
  readonly resources = inject(AppResourcesService)
  readonly backendLabels = backendLabels
  readonly apiLabels = apiLabels
  private readonly api = inject(ApiService)
  private readonly destroyRef = inject(DestroyRef)
  readonly activeJobs = signal<JobResponse[]>([])
  readonly jobsError = signal<string | null>(null)
  readonly pendingSelection = signal<WorkerConfiguration | null>(null)
  readonly stopping = signal(false)
  readonly checking = signal<string | null>(null)
  private readonly discoveryAbort = new AbortController()
  readonly runtimeSelection = computed(() => {
    const backend = this.resources.runtimePreferenceBackend()
    return backend ? this.resources.runtimePreferences()[backend] ?? null : null
  })
  readonly selectedBackend = computed(() => this.resources.runtimePreferenceBackend())
  readonly runtimeApiStatuses = computed(() => this.resources.runtimeApis().length ? this.resources.runtimeApis() : fallbackRuntimeApis)
  readonly currentJobUsesSelection = computed(() => this.activeJobs().some((job) => sameWorker(job.worker, this.runtimeSelection())))

  constructor() {
    this.destroyRef.onDestroy(() => this.discoveryAbort.abort())
    timer(0, 1000).pipe(
      switchMap(() => from(this.refreshJobs())),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe()
  }

  operationalCount(): number { return this.resources.backendStatuses().filter((status) => status.available).length }
  backendDetail(status: ExecutionBackendStatus): string { return status.available ? `${status.devices?.length || 1} device${status.devices?.length === 1 ? '' : 's'}${status.device_name ? ` · ${status.device_name}` : ''}` : status.discovery_state === 'unchecked' ? 'Capability check has not run' : status.discovery_state === 'checking' ? 'Checking in an isolated worker' : status.installed ? formatReason(status.reason_code) : 'Not installed' }
  devicesFor(status: ExecutionBackendStatus): RuntimeDevice[] { return status.devices?.length ? status.devices : [fallbackDevice()] }
  selectedDevice(status: ExecutionBackendStatus): RuntimeDevice { const current = this.runtimeSelection(); return this.devicesFor(status).find((device) => device.device_id === current?.device_id) ?? this.devicesFor(status)[0] }
  selectedPolicy(status: ExecutionBackendStatus): WorkerConfiguration['memory_policy'] { const current = this.runtimeSelection(); const device = this.selectedDevice(status); const policy = current?.backend === status.id ? current.memory_policy : { kind: 'default' as MemoryPolicyKind }; return device.supported_memory_policies.includes(policy.kind) ? policy : policyFromKind(device.supported_memory_policies[0] ?? 'default') }

  async check(status: ExecutionBackendStatus, event: Event): Promise<void> {
    event.stopPropagation()
    if (this.checking()) return
    const api = status.runtime_apis?.find((item) => item.id === this.runtimeSelection()?.api)?.id ?? preferredRuntimeApi(status) ?? 'keras'
    this.checking.set(`${status.id}:${api}`)
    try {
      await this.resources.discoverRuntime(status.id, api, status.discovery_state !== 'unchecked', this.discoveryAbort.signal)
    } catch (reason) {
      if (!this.destroyRef.destroyed && !(reason instanceof Error && reason.name === 'AbortError')) {
        this.resources.errors.update((current) => ({ ...current, runtime: reason instanceof Error ? reason.message : String(reason) }))
      }
    } finally {
      if (!this.destroyRef.destroyed) this.checking.set(null)
    }
  }

  selectRuntimeApi(apiId: RuntimeApi): void {
    const statuses = this.resources.backendStatuses()
    const currentBackend = this.resources.runtimePreferenceBackend()
    const current = currentBackend ? statuses.find((item) => item.id === currentBackend) : undefined
    const targetId = current && backendSupportsApi(current, apiId)
      ? current.id
      : this.resources.preferredBackendForApi(apiId)
    const target = statuses.find((item) => item.id === targetId)
    if (!target) return
    const existing = this.resources.runtimePreferences()[target.id]
    const devices = this.devicesFor(target)
    const device = existing ? devices.find((item) => item.device_id === existing.device_id && item.available) : undefined
    const selectedDevice = device ?? devices.find((item) => item.available)
    if (!selectedDevice) return
    const memoryPolicy = existing && selectedDevice.supported_memory_policies.includes(existing.memory_policy.kind) ? existing.memory_policy : policyFromKind(selectedDevice.supported_memory_policies[0] ?? 'default')
    this.requestSelection({ api: apiId, backend: target.id, device_id: selectedDevice.device_id, memory_policy: memoryPolicy, precision: existing?.precision ?? 'float32' })
  }

  selectBackend(status: ExecutionBackendStatus): void { if (status.available) this.requestSelection(this.selectionFor(status, this.selectedDevice(status), this.selectedPolicy(status))) }
  selectDevice(status: ExecutionBackendStatus, event: Event): void { const device = this.devicesFor(status).find((item) => item.device_id === this.value(event)); if (device) this.requestSelection(this.selectionFor(status, device, policyFromKind(device.supported_memory_policies[0] ?? 'default'))) }
  selectPolicy(status: ExecutionBackendStatus, event: Event): void { this.requestSelection(this.selectionFor(status, this.selectedDevice(status), policyFromKind(this.value(event) as MemoryPolicyKind))) }
  updatePolicyValue(status: ExecutionBackendStatus, event: Event): void { const selected = this.selectedPolicy(status); const numeric = Number(this.value(event)); const policy = selected.kind === 'tensorflow_limit_mb' ? { kind: selected.kind, limit_mb: numeric } : { kind: selected.kind, fraction: numeric }; this.requestSelection(this.selectionFor(status, this.selectedDevice(status), policy)) }

  keepRunning(): void { const pending = this.pendingSelection(); if (pending) this.applySelection(pending) }
  async stopAndSwitch(): Promise<void> {
    const pending = this.pendingSelection()
    const current = this.runtimeSelection()
    if (!pending || !current) return
    this.stopping.set(true)
    try {
      await Promise.all(this.activeJobs().filter((job) => sameWorker(job.worker, current)).map((job) => this.api.stopJob(job.id)))
      let remaining = this.activeJobs().filter((job) => sameWorker(job.worker, current))
      for (let attempt = 0; attempt < 30 && remaining.length; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 250))
        remaining = (await this.api.listJobs(true)).filter((job) => sameWorker(job.worker, current))
      }
      if (remaining.length) throw new Error('The active worker did not stop before the switch timeout.')
      this.applySelection(pending)
      await this.refreshJobs()
    } catch (reason) {
      this.jobsError.set(`Unable to stop the active worker: ${message(reason)}`)
    } finally {
      this.stopping.set(false)
    }
  }

  async stopJob(job: JobResponse): Promise<void> {
    try {
      await this.api.stopJob(job.id)
      await this.refreshJobs()
    } catch (reason) {
      this.jobsError.set(`Unable to stop worker ${job.id}: ${message(reason)}`)
    }
  }

  memoryText(job: JobResponse): string { return job.diagnostics.process_rss_bytes ? `${Math.round(job.diagnostics.process_rss_bytes / 1024 / 1024)} MB RSS` : 'memory pending' }

  private async refreshJobs(): Promise<void> {
    try {
      this.activeJobs.set(await this.api.listJobs(true))
      this.jobsError.set(null)
    } catch (reason) {
      this.jobsError.set(`Unable to refresh active workers: ${message(reason)}`)
    }
  }

  private requestSelection(next: WorkerConfiguration): void { if (this.runtimeSelection() && this.currentJobUsesSelection() && !sameWorker(this.runtimeSelection(), next)) this.pendingSelection.set(next); else this.applySelection(next) }
  private applySelection(next: WorkerConfiguration): void { this.resources.setRuntimePreference(next); this.pendingSelection.set(null) }
  private selectionFor(status: ExecutionBackendStatus, device: RuntimeDevice, memoryPolicy: WorkerConfiguration['memory_policy']): WorkerConfiguration { return { api: this.selectedApi(status), backend: status.id, device_id: device.device_id, memory_policy: memoryPolicy, precision: this.runtimeSelection()?.precision ?? 'float32' } }
  private selectedApi(status: ExecutionBackendStatus): RuntimeApi { const current = this.runtimeSelection(); return current?.api && backendSupportsApi(status, current.api) ? current.api : preferredRuntimeApi(status) ?? 'keras' }
  private value(event: Event): string { return (event.target as HTMLInputElement | HTMLSelectElement).value }
}

function sameWorker(left: WorkerConfiguration | null, right: WorkerConfiguration | null): boolean { return Boolean(left && right && left.api === right.api && left.backend === right.backend && left.device_id === right.device_id && left.memory_policy.kind === right.memory_policy.kind && left.precision === right.precision) }
function policyFromKind(kind: MemoryPolicyKind): WorkerConfiguration['memory_policy'] { if (kind === 'tensorflow_limit_mb') return { kind, limit_mb: 1024 }; if (kind === 'jax_preallocate_fraction' || kind === 'pytorch_fraction') return { kind, fraction: 0.5 }; return { kind } }
function fallbackDevice(): RuntimeDevice { return { device_id: 'cpu', lease_key: 'cpu', kind: 'cpu', name: 'CPU', accelerator: 'cpu', available: true, supported_memory_policies: ['default'] } }
function formatReason(reason: string | null | undefined): string { if (reason === 'dependency_missing') return 'Installed package could not be imported'; if (reason === 'worker_failed') return 'Worker probe failed'; if (reason === 'backend_import_failed') return 'Backend import failed'; return reason ? reason.replaceAll('_', ' ') : 'Capability probe unavailable' }
function message(reason: unknown): string { return reason instanceof Error ? reason.message : String(reason) }
