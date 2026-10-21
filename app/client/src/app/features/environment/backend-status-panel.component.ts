// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, computed, inject, signal } from '@angular/core'
import { takeUntilDestroyed } from '@angular/core/rxjs-interop'
import { LucideAlertTriangle, LucideCheckCircle2, LucideCpu, LucidePackageX, LucideSquare, LucideTerminal } from '@lucide/angular'
import { from, timer } from 'rxjs'
import { switchMap } from 'rxjs/operators'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { backendSupportsApi, defaultMemoryPolicy, isMemoryPolicyValid, preferredRuntimeApi } from '../../core/state/runtime-selection'
import type { ExecutionBackendStatus, JobResponse, MemoryPolicyKind, RuntimeApi, RuntimeDevice, WorkerConfiguration } from '../../domain/types'

const backendLabels: Record<ExecutionBackendStatus['id'], string> = { tensorflow: 'TensorFlow', pytorch: 'PyTorch', jax: 'JAX' }
const apiLabels: Record<RuntimeApi, string> = { keras: 'Keras', native: 'Native framework' }

@Component({
  selector: 'nw-backend-status-panel',
  standalone: true,
  imports: [LucideAlertTriangle, LucideCheckCircle2, LucideCpu, LucidePackageX, LucideSquare, LucideTerminal],
  template: `
    @if (resources.runtimeReady() && resources.backendStatuses().length) {
      <section class="backend-status-panel" aria-label="Compute environment status">
        <div class="backend-status-heading"><div><span class="eyebrow">COMPUTE ENVIRONMENT</span><h2>Execution runtime</h2><p>{{ operationalCount() ? operationalCount() + ' operational backend' + (operationalCount() === 1 ? '' : 's') : 'No operational backend' }} · isolated runtime discovery · one lease per physical device</p></div><span class="backend-status-command"><svg lucideTerminal [size]="13"></svg>isolated worker controller</span></div>
        <div class="runtime-api-controls" aria-label="Runtime API">@for (status of runtimeApiStatuses(); track status.id) { <button type="button" class="runtime-api-option" [class.selected]="runtimeSelection()?.api === status.id" [disabled]="!status.available" (click)="selectRuntimeApi(status.id)"><strong>{{ apiLabels[status.id] }}</strong><span>{{ status.available ? 'Operational' : status.state === 'checking' ? 'Checking' : status.state === 'unchecked' ? 'Not checked' : status.reason_code === 'not_implemented' ? 'Deferred' : 'Unavailable' }}</span></button> }</div>
        <div class="backend-status-grid">
          @for (status of resources.backendStatuses(); track status.id) {
            <div class="backend-status-card-container" [class.selected]="selectedBackend() === status.id">
              <button type="button" class="backend-status-card" [class.usable]="status.available && backendSupportsAnyApi(status)" [class.unavailable]="!status.available && status.installed" [class.missing]="!status.available && !status.installed" [class.selected]="selectedBackend() === status.id" [disabled]="!status.available || !backendSupportsAnyApi(status)" [attr.aria-pressed]="selectedBackend() === status.id" (click)="selectBackend(status)">
                <div class="backend-status-card-top"><span class="backend-status-name"><svg lucideCpu [size]="14"></svg>{{ backendLabels[status.id] }}</span><span class="backend-status-state">@if (status.available && backendSupportsAnyApi(status)) { <svg lucideCheckCircle2 [size]="13"></svg>Usable } @else if (status.discovery_state === 'checking') { <svg lucideAlertTriangle [size]="13"></svg>Checking } @else if (status.discovery_state === 'unchecked') { <svg lucideAlertTriangle [size]="13"></svg>Not checked } @else if (status.installed) { <svg lucideAlertTriangle [size]="13"></svg>Installed } @else { <svg lucidePackageX [size]="13"></svg>Missing }</span></div>
                <strong>{{ status.version ? 'v' + status.version : 'Optional framework' }}</strong><span>{{ backendDetail(status) }}</span>
              </button>
              @if (status.runtime_apis?.length && status.discovery_state !== 'checking') {
                <div class="runtime-checks" [attr.aria-label]="backendLabels[status.id] + ' runtime checks'">
                  @for (apiStatus of status.runtime_apis; track apiStatus.id) {
                    @if (!apiStatus.available) {
                      <button class="secondary runtime-check" type="button" [disabled]="checking() === status.id + ':' + apiStatus.id" (click)="check(status, $event, apiStatus.id)">{{ checking() === status.id + ':' + apiStatus.id ? 'Checking…' : apiStatus.state === 'unchecked' ? 'Check ' + apiLabels[apiStatus.id] : 'Refresh ' + apiLabels[apiStatus.id] }}</button>
                    }
                  }
                </div>
              }
              @if (status.available && backendSupportsAnyApi(status)) {
                @if (selectedDevice(status); as device) {
                  @if (selectedPolicy(status); as policy) {
                    <div class="runtime-fields">
                      <label>Device<select [value]="device.device_id" (change)="selectDevice(status, $event)">@for (availableDevice of devicesFor(status); track availableDevice.device_id) { <option [value]="availableDevice.device_id" [selected]="availableDevice.device_id === device.device_id">{{ availableDevice.device_id }} · {{ availableDevice.name }}</option> }</select></label>
                      <label>Memory policy<select [value]="policy.kind" (change)="selectPolicy(status, $event)">@for (policyKind of device.supported_memory_policies; track $index) { <option [value]="policyKind" [selected]="policyKind === policy.kind">{{ policyKind }}</option> }</select></label>
                      @if (policy.kind === 'tensorflow_limit_mb') { <label>Limit (MB)<input type="number" min="1" [value]="policy.limit_mb ?? 1024" (input)="updatePolicyValue(status, $event)" /></label> }
                      @if (policy.kind === 'jax_preallocate_fraction' || policy.kind === 'pytorch_fraction') { <label>Fraction<input type="number" min="0.01" max="1" step="0.01" [value]="policy.fraction ?? 0.5" (input)="updatePolicyValue(status, $event)" /></label> }
                    </div>
                  }
                }
              }
            </div>
          }
        </div>
        @if (resources.runtimePreferenceSaveError()) { <div class="selection-prompt panel" role="alert"><strong>Runtime preferences could not be saved.</strong><span>{{ resources.runtimePreferenceSaveError() }}</span></div> }
        @else if (resources.runtimePreferenceSaveStatus() === 'saving') { <p class="runtime-save-status" role="status">Saving runtime preferences…</p> }
        @else if (resources.runtimePreferenceSaveStatus() === 'saved') { <p class="runtime-save-status" role="status">Runtime preferences saved automatically.</p> }
        @if (activeJobs().length) { <p class="runtime-save-status">New runtime selections apply to future jobs. Active workers keep their original configuration.</p> }
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
  readonly checking = signal<string | null>(null)
  private readonly discoveryAbort = new AbortController()
  readonly runtimeSelection = computed(() => {
    const backend = this.resources.runtimePreferenceBackend()
    return backend ? this.resources.runtimePreferences()[backend] ?? null : null
  })
  readonly selectedBackend = computed(() => this.resources.runtimePreferenceBackend())
  readonly runtimeApiStatuses = computed(() => this.resources.runtimeApis())

  constructor() {
    this.destroyRef.onDestroy(() => this.discoveryAbort.abort())
    timer(0, 1000).pipe(
      switchMap(() => from(this.refreshJobs())),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe()
  }

  operationalCount(): number { return this.resources.backendStatuses().filter((status) => status.available && this.backendSupportsAnyApi(status)).length }
  backendSupportsAnyApi(status: ExecutionBackendStatus): boolean { return Boolean(status.runtime_apis?.some((item) => item.available)) }
  preferredRuntimeApi(status: ExecutionBackendStatus): RuntimeApi | null { return preferredRuntimeApi(status) }
  backendDetail(status: ExecutionBackendStatus): string { return status.available ? `${status.devices?.length || 1} device${status.devices?.length === 1 ? '' : 's'}${status.device_name ? ` · ${status.device_name}` : ''}` : status.discovery_state === 'unchecked' ? 'Capability check has not run' : status.discovery_state === 'checking' ? 'Checking in an isolated worker' : status.installed ? formatReason(status.reason_code) : 'Not installed' }
  devicesFor(status: ExecutionBackendStatus): RuntimeDevice[] { return status.devices?.filter((device) => device.available) ?? [] }
  selectedDevice(status: ExecutionBackendStatus): RuntimeDevice | undefined {
    const current = this.resources.runtimePreferences()[status.id]
    return this.devicesFor(status).find((device) => device.device_id === current?.device_id) ?? this.devicesFor(status)[0]
  }
  selectedPolicy(status: ExecutionBackendStatus): WorkerConfiguration['memory_policy'] | null {
    const current = this.resources.runtimePreferences()[status.id]
    const device = this.selectedDevice(status)
    if (!device) return null
    const policy = current?.memory_policy
    if (policy && device.supported_memory_policies.includes(policy.kind) && isMemoryPolicyValid(policy)) return policy
    const kind = device.supported_memory_policies[0]
    return kind ? defaultMemoryPolicy(kind) : null
  }

  async check(status: ExecutionBackendStatus, event: Event, requestedApi?: RuntimeApi): Promise<void> {
    event.stopPropagation()
    if (this.checking()) return
    const api = requestedApi ?? status.runtime_apis?.find((item) => item.id === this.runtimeSelection()?.api && !item.available)?.id ?? preferredRuntimeApi(status)
    if (!api) return
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
    const device = existing ? devices.find((item) => item.device_id === existing.device_id) : undefined
    const selectedDevice = device ?? devices[0]
    if (!selectedDevice) return
    const memoryPolicy = existing && selectedDevice.supported_memory_policies.includes(existing.memory_policy.kind) && isMemoryPolicyValid(existing.memory_policy)
      ? existing.memory_policy
      : selectedDevice.supported_memory_policies[0] ? defaultMemoryPolicy(selectedDevice.supported_memory_policies[0]) : null
    if (!memoryPolicy) return
    this.requestSelection({ api: apiId, backend: target.id, device_id: selectedDevice.device_id, memory_policy: memoryPolicy, precision: existing?.precision ?? 'float32' })
  }

  selectBackend(status: ExecutionBackendStatus): void {
    const device = this.selectedDevice(status)
    const policy = this.selectedPolicy(status)
    const selection = device && policy ? this.selectionFor(status, device, policy) : null
    if (selection) this.requestSelection(selection)
  }
  selectDevice(status: ExecutionBackendStatus, event: Event): void {
    const device = this.devicesFor(status).find((item) => item.device_id === this.value(event))
    const policy = device?.supported_memory_policies[0] ? defaultMemoryPolicy(device.supported_memory_policies[0]) : null
    const selection = device && policy ? this.selectionFor(status, device, policy) : null
    if (selection) this.requestSelection(selection)
  }
  selectPolicy(status: ExecutionBackendStatus, event: Event): void {
    const device = this.selectedDevice(status)
    const policy = defaultMemoryPolicy(this.value(event) as MemoryPolicyKind)
    const selection = device ? this.selectionFor(status, device, policy) : null
    if (selection) this.requestSelection(selection)
  }
  updatePolicyValue(status: ExecutionBackendStatus, event: Event): void {
    const selected = this.selectedPolicy(status)
    const device = this.selectedDevice(status)
    if (!selected || !device) return
    const numeric = Number(this.value(event))
    const policy = selected.kind === 'tensorflow_limit_mb' ? { kind: selected.kind, limit_mb: numeric } : { kind: selected.kind, fraction: numeric }
    const selection = this.selectionFor(status, device, policy)
    if (selection) this.requestSelection(selection, 350)
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

  private requestSelection(next: WorkerConfiguration, debounceMs = 0): void { this.applySelection(next, debounceMs) }
  private applySelection(next: WorkerConfiguration, debounceMs = 0): void { this.resources.setRuntimePreference(next, debounceMs) }
  private selectionFor(status: ExecutionBackendStatus, device: RuntimeDevice, memoryPolicy: WorkerConfiguration['memory_policy']): WorkerConfiguration | null {
    const api = this.selectedApi(status)
    if (!device.available || !api || !device.supported_memory_policies.includes(memoryPolicy.kind) || !isMemoryPolicyValid(memoryPolicy)) return null
    return { api, backend: status.id, device_id: device.device_id, memory_policy: memoryPolicy, precision: this.runtimeSelection()?.precision ?? 'float32' }
  }
  private selectedApi(status: ExecutionBackendStatus): RuntimeApi | null { const current = this.runtimeSelection(); return current?.api && backendSupportsApi(status, current.api) ? current.api : preferredRuntimeApi(status) }
  private value(event: Event): string { return (event.target as HTMLInputElement | HTMLSelectElement).value }
}

function formatReason(reason: string | null | undefined): string { if (reason === 'dependency_missing') return 'Installed package could not be imported'; if (reason === 'worker_failed') return 'Worker probe failed'; if (reason === 'backend_import_failed') return 'Backend import failed'; return reason ? reason.replaceAll('_', ' ') : 'Capability probe unavailable' }
function message(reason: unknown): string { return reason instanceof Error ? reason.message : String(reason) }
