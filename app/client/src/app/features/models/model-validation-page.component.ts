// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, effect, inject, signal } from '@angular/core'
import { RouterLink } from '@angular/router'
import { LucideFlaskConical, LucideSquare } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import { ApiError } from '../../core/api/api-error'
import type { JobResponse, RuntimeDevice, WorkerConfiguration } from '../../domain/types'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { pollJobUntilTerminal, terminalJobStates } from './job-polling'
import { ModelWorkspaceState } from './model-workspace.state'

interface SavedValidationJobReference {
  jobId: string
  projectId: string
  revision: number
  backend: string
  api: 'keras' | 'native'
}

@Component({
  selector: 'nw-model-validation-page',
  standalone: true,
  imports: [RouterLink, LucideFlaskConical, LucideSquare],
  template: `
    @if (state.project(); as project) {
      <div class="workflow-page validation-page"><div class="workflow-intro"><div><span class="eyebrow">FUNCTIONAL VALIDATION</span><h2>Run a safe model check</h2><p>The worker creates compatible dummy inputs and performs one forward pass. It may run a temporary optimizer step in worker memory to verify the optimizer; it does not train or change the saved model, or serve user predictions.</p></div><div class="backend-selection-summary workflow-backend-summary"><span>Model backend preference</span>@if (worker(); as selection) { <strong>{{ selection.api }} · {{ selection.backend }} · {{ selection.device_id }}</strong> } @else { <a routerLink="/environment">Configure {{ project.execution_backend }} in Environment</a> }</div></div><section class="validation-controls panel"><div class="validation-control-note"><svg lucideFlaskConical [size]="18"></svg><span>Functional validation uses the model's persisted backend and its matching runtime preference. It never changes saved model data.</span></div><a class="secondary environment-link" routerLink="/environment">Configure runtime</a></section><section class="smoke-test-drawer panel"><div class="section-heading"><div><span class="eyebrow">FUNCTIONAL CHECK</span><h3>One forward pass</h3></div>@if (job(); as current) { <span class="muted">{{ current.state }}</span> }</div><p class="muted">The worker returns input/output tensor shapes, device information, and parameter count.</p><div class="workflow-card-actions"><button class="primary" type="button" (click)="run()" [disabled]="busy()">{{ busy() ? 'Running…' : 'Run functional validation' }}</button>@if (busy()) { <button class="secondary" type="button" (click)="stop()"><svg lucideSquare [size]="14"></svg>Stop</button> }</div>@if (result(); as value) { <div class="smoke-result"><strong>Completed on {{ value.device_name }}</strong><span>{{ value.inputs.length }} inputs · {{ value.outputs.length }} outputs · {{ value.parameter_count }} parameters · {{ value.duration_ms }} ms</span></div> }@if (error()) { <div class="model-export-notice" role="alert">{{ error() }}</div> }</section></div>
    }
  `,
})
export class ModelValidationPageComponent {
  readonly state = inject(ModelWorkspaceState)
  private readonly api = inject(ApiService)
  private readonly resources = inject(AppResourcesService)
  private readonly destroyRef = inject(DestroyRef)
  readonly busy = signal(false)
  readonly job = signal<JobResponse | null>(null)
  readonly result = signal<{ device_name: string; inputs: unknown[]; outputs: unknown[]; parameter_count: number; duration_ms: number } | null>(null)
  readonly error = signal<string | null>(null)
  readonly worker = () => this.workerForProject()
  private readonly discoveryAbort = new AbortController()
  private currentTargetKey: string | null = null

  constructor() {
    void this.resources.loadRuntimeMetadata()
    this.destroyRef.onDestroy(() => this.discoveryAbort.abort())
    effect(() => {
      const project = this.state.project()
      if (!project) return
      const api = project.keras_enabled ? 'keras' : 'native'
      const targetKey = `${project.id}:${project.current_revision}:${project.execution_backend}:${api}`
      if (this.currentTargetKey !== targetKey) {
        this.currentTargetKey = targetKey
        this.job.set(null)
        this.result.set(null)
        this.error.set(null)
        this.busy.set(false)
        void this.restoreSavedJob(project.id, project.current_revision, project.execution_backend, api)
      }
      void this.resources.discoverRuntime(project.execution_backend, api, false, this.discoveryAbort.signal).catch((reason) => {
        if (!this.destroyRef.destroyed) this.error.set(reason instanceof Error ? reason.message : String(reason))
      })
    })
  }

  private workerForProject(): WorkerConfiguration | null {
    const project = this.state.project()
    if (!project) return null
    const status = this.resources.backendStatuses().find((item) => item.id === project.execution_backend && item.available)
    const preference = this.resources.runtimePreferences()[project.execution_backend]
    const devices = status?.devices?.length ? status.devices : status ? [fallbackDevice()] : []
    const device = devices.find((item) => item.device_id === preference?.device_id && item.available) ?? devices.find((item) => item.available)
    const api = project.keras_enabled ? 'keras' : 'native'
    if (!status || !preference || !device || !this.resources.runtimeCapabilityReady(project.execution_backend, api)) return null
    const memoryPolicy = device.supported_memory_policies.includes(preference.memory_policy.kind) ? preference.memory_policy : { kind: 'default' as const }
    return { ...preference, api, device_id: device.device_id, memory_policy: memoryPolicy }
  }

  async run(): Promise<void> {
    const project = this.state.project()
    if (!project) return
    const api = project.keras_enabled ? 'keras' : 'native'
    this.busy.set(true); this.error.set(null); this.result.set(null)
    try {
      let worker = this.worker()
      if (!worker) {
        await this.resources.discoverRuntime(project.execution_backend, api, false, this.discoveryAbort.signal)
        worker = this.worker()
      }
      if (!worker) throw new Error('The selected runtime is unavailable. Check it in Environment and retry.')
      let current = await this.api.submitFunctionalValidation(project.id, project.graph, worker)
      this.saveJobReference(current.id, project.id, project.current_revision, project.execution_backend, api)
      this.job.set(current)
      if (!terminalJobStates.has(current.state)) current = await pollJobUntilTerminal(current, (jobId) => this.api.getJob(jobId), (next) => { if (this.isCurrentTarget(project.id, project.current_revision, project.execution_backend, api)) this.job.set(next) }, { intervalMs: 250, timeoutMs: 60_000, isDestroyed: () => this.destroyRef.destroyed || !this.isCurrentTarget(project.id, project.current_revision, project.execution_backend, api) })
      if (this.destroyRef.destroyed || !this.isCurrentTarget(project.id, project.current_revision, project.execution_backend, api)) return
      if (!terminalJobStates.has(current.state)) throw new Error('The validation worker did not reach a terminal state within 60 seconds. Refresh the page to inspect it.')
      this.applyJobOutcome(current)
    } catch (reason) { if (!this.destroyRef.destroyed && this.isCurrentTarget(project.id, project.current_revision, project.execution_backend, api)) this.error.set(reason instanceof Error ? reason.message : String(reason)) } finally { if (!this.destroyRef.destroyed && this.isCurrentTarget(project.id, project.current_revision, project.execution_backend, api)) this.busy.set(false) }
  }

  async stop(): Promise<void> { const current = this.job(); if (!current) return; try { this.job.set(await this.api.stopJob(current.id)) } catch (reason) { this.error.set(reason instanceof Error ? reason.message : String(reason)) } }

  private async restoreSavedJob(projectId: string, revision: number, backend: string, api: 'keras' | 'native'): Promise<void> {
    const saved = this.readJobReference(projectId)
    if (!saved) return
    if (saved.projectId !== projectId || saved.revision !== revision || saved.backend !== backend || saved.api !== api) {
      this.clearJobReference(projectId)
      return
    }

    this.busy.set(true)
    try {
      let current = await this.api.getJob(saved.jobId)
      if (!this.isCurrentTarget(projectId, revision, backend, api)) return
      if (current.operation !== 'smoke_inference' || current.worker.api !== api || current.worker.backend !== backend) {
        this.clearJobReference(projectId)
        return
      }
      this.job.set(current)
      if (!terminalJobStates.has(current.state)) current = await pollJobUntilTerminal(current, (jobId) => this.api.getJob(jobId), (next) => { if (this.isCurrentTarget(projectId, revision, backend, api)) this.job.set(next) }, { intervalMs: 250, timeoutMs: 60_000, isDestroyed: () => this.destroyRef.destroyed || !this.isCurrentTarget(projectId, revision, backend, api) })
      if (this.destroyRef.destroyed || !this.isCurrentTarget(projectId, revision, backend, api)) return
      this.job.set(current)
      this.applyJobOutcome(current)
    } catch (reason) {
      if (this.destroyRef.destroyed || !this.isCurrentTarget(projectId, revision, backend, api)) return
      if (reason instanceof ApiError && reason.status === 404) {
        this.clearJobReference(projectId)
        this.job.set(null)
        return
      }
      this.error.set(reason instanceof Error ? reason.message : String(reason))
    } finally {
      if (!this.destroyRef.destroyed && this.isCurrentTarget(projectId, revision, backend, api)) this.busy.set(false)
    }
  }

  private applyJobOutcome(current: JobResponse): void {
    if (current.state !== 'succeeded') {
      this.error.set(current.error?.message || current.progress.message || 'The validation worker did not complete.')
      return
    }
    const data = current.result || {}
    this.result.set({ device_name: String(data['device_name'] || current.worker.device_id), inputs: Array.isArray(data['inputs']) ? data['inputs'] : [], outputs: Array.isArray(data['outputs']) ? data['outputs'] : [], parameter_count: Number(data['parameter_count'] || 0), duration_ms: Number(data['duration_ms'] || 0) })
  }

  private isCurrentTarget(projectId: string, revision: number, backend: string, api: 'keras' | 'native'): boolean {
    const current = this.state.project()
    return current?.id === projectId && current.current_revision === revision && current.execution_backend === backend && (current.keras_enabled ? 'keras' : 'native') === api
  }

  private saveJobReference(jobId: string, projectId: string, revision: number, backend: string, api: 'keras' | 'native'): void {
    if (typeof sessionStorage === 'undefined') return
    const reference: SavedValidationJobReference = { jobId, projectId, revision, backend, api }
    try { sessionStorage.setItem(this.jobStorageKey(projectId), JSON.stringify(reference)) } catch { /* Storage is only a reload convenience. */ }
  }

  private readJobReference(projectId: string): SavedValidationJobReference | null {
    if (typeof sessionStorage === 'undefined') return null
    const key = this.jobStorageKey(projectId)
    try {
      const raw = sessionStorage.getItem(key)
      if (!raw) return null
      const value = JSON.parse(raw) as Partial<SavedValidationJobReference>
      if (typeof value.jobId === 'string' && typeof value.projectId === 'string' && typeof value.revision === 'number' && typeof value.backend === 'string' && (value.api === 'keras' || value.api === 'native')) return value as SavedValidationJobReference
      this.clearJobReference(projectId)
      return null
    } catch {
      this.clearJobReference(projectId)
      return null
    }
  }

  private clearJobReference(projectId: string): void {
    if (typeof sessionStorage !== 'undefined') {
      try { sessionStorage.removeItem(this.jobStorageKey(projectId)) } catch { /* Storage is only a reload convenience. */ }
    }
  }

  private jobStorageKey(projectId: string): string { return `neuro-weave:validation-job:${projectId}` }
}

function fallbackDevice(): RuntimeDevice { return { device_id: 'cpu', lease_key: 'cpu', kind: 'cpu', name: 'CPU', accelerator: 'cpu', available: true, supported_memory_policies: ['default'] } }
