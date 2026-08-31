import { Component, effect, inject, signal } from '@angular/core'
import { LucideAlertTriangle, LucideCheckCircle2, LucideDownload, LucideRefreshCw } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import type { ExportCapability, ExportFormat } from '../../domain/types'
import { ModelWorkspaceState } from './model-workspace.state'

@Component({
  selector: 'nw-model-export-page',
  standalone: true,
  imports: [LucideAlertTriangle, LucideCheckCircle2, LucideDownload, LucideRefreshCw],
  template: `
    @if (state.project(); as project) {
      <div class="workflow-page export-page"><section class="model-export-page panel" aria-labelledby="model-export-title"><header class="model-export-header"><div><span class="eyebrow">MODEL EXPORT CENTER</span><h2 id="model-export-title">Export {{ project.name }}</h2><p>Revision {{ capabilitiesRevision() ?? project.current_revision }} · {{ project.execution_backend }} · {{ project.keras_enabled ? 'Keras profile' : 'native profile' }}</p></div></header>
        @if (loading()) { <div class="model-export-state"><svg class="spin" lucideRefreshCw [size]="20"></svg><span>Probing export capabilities…</span></div> } @else if (error()) { <div class="model-export-state export-error" role="alert"><svg lucideAlertTriangle [size]="18"></svg><div><strong>Capabilities unavailable</strong><span>{{ error() }}</span><button class="secondary" type="button" (click)="load()"><svg lucideRefreshCw [size]="14"></svg>Retry</button></div></div> } @else {
          <div class="model-export-body"><div class="model-export-catalog">@for (capability of capabilities(); track capability.format) { <button type="button" class="export-format-option" [class.selected]="selectedFormat() === capability.format" [class.unavailable]="!capability.available" [disabled]="!capability.available" (click)="selectedFormat.set(capability.format)"><span class="export-format-top"><strong>{{ capability.label }}</strong>@if (capability.available) { <svg lucideCheckCircle2 [size]="14"></svg> } @else { <svg lucideAlertTriangle [size]="14"></svg> }</span><span class="export-format-description">{{ capability.available ? capability.compatibility : capability.reason || 'Unavailable for this profile.' }}</span></button> } @if (!capabilities().length) { <div class="model-export-state"><svg lucideAlertTriangle [size]="18"></svg><span>No export formats were returned for this project.</span></div> }</div><aside class="model-export-config panel"><div class="section-heading"><div><span class="eyebrow">EXPORT SETTINGS</span><h3>{{ selectedCapability()?.label || 'Select a format' }}</h3></div><svg lucideDownload [size]="17"></svg></div>@if (selectedCapability(); as capability) { <label class="export-field">Filename stem<input [value]="filenameStem()" (input)="filenameStem.set($any($event.target).value)" /></label><div class="export-preview-name"><span>Download name</span><strong>{{ filenameStem() || project.name }}{{ extension(capability) }}</strong></div><div class="export-detail-list"><div><span>Payload</span><strong>{{ capability.payload }}</strong></div><div><span>Conversion</span><strong>{{ capability.conversion }}</strong></div><div><span>Verification</span><strong>Structural · reload · inference</strong></div></div> } @else { <p class="muted">Select an available capability to configure an export.</p> }</aside></div>
          <footer class="model-export-footer dialog-actions"><span class="model-export-footer-note">Unsupported formats remain visible with their reason.</span><button class="primary" type="button" (click)="exportSelected()" [disabled]="!selectedCapability()"><svg lucideDownload [size]="15"></svg>Export artifact</button></footer>
        }
        @if (notice()) { <div class="model-export-notice" role="alert">{{ notice() }}</div> }
      </section></div>
    }
  `,
})
export class ModelExportPageComponent {
  readonly state = inject(ModelWorkspaceState)
  private readonly api = inject(ApiService)
  readonly capabilities = signal<ExportCapability[]>([])
  readonly capabilitiesRevision = signal<number | null>(null)
  readonly selectedFormat = signal<ExportFormat | null>(null)
  readonly filenameStem = signal('model')
  readonly loading = signal(true)
  readonly error = signal<string | null>(null)
  readonly notice = signal<string | null>(null)
  readonly selectedCapability = () => this.capabilities().find((item) => item.format === this.selectedFormat()) ?? null

  constructor() { effect(() => { const project = this.state.project(); if (project) { this.filenameStem.set(project.name); void this.load() } }) }

  async load(): Promise<void> {
    const project = this.state.project()
    if (!project) return
    this.loading.set(true); this.error.set(null)
    try { const response = await this.api.exportCapabilities(project.id); this.capabilities.set(response.formats); this.capabilitiesRevision.set(response.project_revision); this.selectedFormat.set(response.formats.find((item) => item.available)?.format ?? null) } catch (reason) { this.error.set(reason instanceof Error ? reason.message : String(reason)) } finally { this.loading.set(false) }
  }

  extension(capability: ExportCapability): string { return capability.extension.startsWith('.') ? capability.extension : `.${capability.extension}` }
  async exportSelected(): Promise<void> { const capability = this.selectedCapability(); const project = this.state.project(); if (!capability || !project) return; try { const bundle = capability.format === 'project_bundle' ? await this.api.exportProject(project.id) : null; if (bundle) this.download(new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' }), `${this.filenameStem() || project.name}.json`); else this.notice.set('This export is available from the full worker flow; configure the runtime before running it.') } catch (reason) { this.notice.set(`Export failed: ${reason instanceof Error ? reason.message : String(reason)}`) } }
  private download(blob: Blob, filename: string): void { const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click(); URL.revokeObjectURL(url); this.notice.set(`Downloaded ${filename}.`) }
}
