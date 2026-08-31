import { Component, inject, signal } from '@angular/core'
import { RouterLink } from '@angular/router'
import { LucideFlaskConical, LucideSquare } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import type { JobResponse, WorkerConfiguration } from '../../domain/types'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { ModelWorkspaceState } from './model-workspace.state'

@Component({
  selector: 'nw-model-inference-page',
  standalone: true,
  imports: [RouterLink, LucideFlaskConical, LucideSquare],
  template: `
    @if (state.project(); as project) {
      <div class="workflow-page inference-page"><div class="workflow-intro"><div><span class="eyebrow">INFERENCE</span><h2>Run a safe model check</h2><p>This workspace creates compatible dummy inputs and performs one forward pass. It never trains or updates weights.</p></div><div class="backend-selection-summary workflow-backend-summary"><span>Model backend preference</span>@if (worker(); as selection) { <strong>{{ selection.api }} · {{ selection.backend }} · {{ selection.device_id }}</strong> } @else { <a routerLink="/environment">Configure {{ project.execution_backend }} in Environment</a> }</div></div><section class="inference-controls panel"><div class="inference-control-note"><svg lucideFlaskConical [size]="18"></svg><span>Inference uses the model's persisted backend and its matching runtime preference. It never changes model data.</span></div><a class="secondary environment-link" routerLink="/environment">Configure runtime</a></section><section class="smoke-test-drawer panel"><div class="section-heading"><div><span class="eyebrow">SMOKE TEST</span><h3>One forward pass</h3></div>@if (job(); as current) { <span class="muted">{{ current.state }}</span> }</div><p class="muted">The worker returns input/output tensor shapes, device information, and parameter count.</p><div class="workflow-card-actions"><button class="primary" type="button" (click)="run()" [disabled]="busy() || !worker()">{{ busy() ? 'Running…' : 'Run smoke test' }}</button>@if (busy()) { <button class="secondary" type="button" (click)="stop()"><svg lucideSquare [size]="14"></svg>Stop</button> }</div>@if (result(); as value) { <div class="smoke-result"><strong>Completed on {{ value.device_name }}</strong><span>{{ value.inputs.length }} inputs · {{ value.outputs.length }} outputs · {{ value.parameter_count }} parameters · {{ value.duration_ms }} ms</span></div> }@if (error()) { <div class="model-export-notice" role="alert">{{ error() }}</div> }</section></div>
    }
  `,
})
export class ModelInferencePageComponent {
  readonly state = inject(ModelWorkspaceState)
  private readonly api = inject(ApiService)
  private readonly resources = inject(AppResourcesService)
  readonly busy = signal(false)
  readonly job = signal<JobResponse | null>(null)
  readonly result = signal<{ device_name: string; inputs: unknown[]; outputs: unknown[]; parameter_count: number; duration_ms: number } | null>(null)
  readonly error = signal<string | null>(null)
  readonly worker = () => this.workerForProject()

  private workerForProject(): WorkerConfiguration | null {
    const project = this.state.project()
    if (!project) return null
    const status = this.resources.backendStatuses().find((item) => item.id === project.execution_backend && item.available)
    const preference = this.resources.runtimePreferences()[project.execution_backend]
    const device = status?.devices?.find((item) => item.available)
    if (!status || !preference || !device) return null
    return { ...preference, api: project.keras_enabled ? 'keras' : 'native', device_id: device.device_id }
  }

  async run(): Promise<void> {
    const project = this.state.project(); const worker = this.worker()
    if (!project || !worker) return
    this.busy.set(true); this.error.set(null); this.result.set(null)
    try { let current = await this.api.submitSmokeInference(project.id, project.graph, worker); this.job.set(current); while (!['succeeded', 'terminated', 'failed'].includes(current.state)) { await new Promise((resolve) => window.setTimeout(resolve, 300)); current = await this.api.getJob(current.id); this.job.set(current) } if (current.state !== 'succeeded') throw new Error(current.error?.message || 'The inference worker did not complete.'); const data = current.result || {}; this.result.set({ device_name: String(data['device_name'] || worker.device_id), inputs: Array.isArray(data['inputs']) ? data['inputs'] : [], outputs: Array.isArray(data['outputs']) ? data['outputs'] : [], parameter_count: Number(data['parameter_count'] || 0), duration_ms: Number(data['duration_ms'] || 0) }) } catch (reason) { this.error.set(reason instanceof Error ? reason.message : String(reason)) } finally { this.busy.set(false) }
  }

  async stop(): Promise<void> { const current = this.job(); if (!current) return; try { this.job.set(await this.api.stopJob(current.id)) } catch (reason) { this.error.set(reason instanceof Error ? reason.message : String(reason)) } }
}
