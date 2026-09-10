import { CommonModule } from '@angular/common'
import { Component, inject } from '@angular/core'
import { LucideAlertTriangle, LucideInfo } from '@lucide/angular'
import type { DatasetBinding } from '../../domain/dataset.types'
import { DatasetWorkspaceState } from './dataset-workspace.state'

@Component({
  selector: 'nw-dataset-overview-page',
  standalone: true,
  imports: [CommonModule, LucideAlertTriangle, LucideInfo],
  template: `
    @if (state.dataset(); as dataset) {
      <div class="dataset-overview workspace-section">
        @if (dataset.availability.status === 'unavailable') { <section class="dataset-alert panel"><svg lucideAlertTriangle [size]="17"></svg><div><strong>External source unavailable</strong><p>{{ dataset.availability.unavailable_source_ids.length }} source path{{ dataset.availability.unavailable_source_ids.length === 1 ? '' : 's' }} cannot currently be read. The registration and stored metadata are preserved; use Edit / relink to point to a new source.</p></div></section> }
        <section class="workspace-card-grid"><div class="workspace-card panel"><span class="eyebrow">SOURCE</span><code>{{ dataset.sources[0]?.path }}</code><small>{{ dataset.sources.length }} registered source{{ dataset.sources.length === 1 ? '' : 's' }}</small></div><div class="workspace-card panel"><span class="eyebrow">SAMPLES</span><strong>{{ dataset.selected_sample_count | number }} / {{ dataset.sample_count | number }}</strong><small>{{ subsetLabel(dataset) }}</small></div><div class="workspace-card panel"><span class="eyebrow">FORMAT</span><strong>{{ dataset.format }}</strong><small>Registration snapshot {{ formatDate(dataset.validation.checked_at) }}</small></div></section>
        <div class="overview-columns"><section class="detail-panel panel"><header><span class="eyebrow">MODEL INPUT</span><h2>{{ bindingSummary(dataset.input_definition.bindings) }}</h2></header><div class="binding-list">@for (binding of dataset.input_definition.bindings; track $index) { <div class="binding-row"><span>{{ binding.type }}</span><strong>{{ bindingDetail(binding) }}</strong></div> }@empty { <p class="muted">No input binding stored.</p> }</div></section><section class="detail-panel panel"><header><span class="eyebrow">MODEL OUTPUT</span><h2>{{ bindingSummary(dataset.output_definition.bindings, 'None (input only)') }}</h2></header><div class="binding-list">@for (binding of dataset.output_definition.bindings; track $index) { <div class="binding-row"><span>{{ binding.type }}</span><strong>{{ bindingDetail(binding) }}</strong></div> }@empty { <p class="muted">This is an input-only definition.</p> }</div></section></div>
        <section class="detail-panel panel"><header><span class="eyebrow">REGISTERED DEFINITION</span><h2>Sources and selection</h2></header><div class="source-list">@for (source of dataset.sources; track source.id) { <div class="source-row"><div><strong>{{ source.id }}</strong><span>{{ source.type }}</span></div><code>{{ source.path }}</code></div> }</div><div class="definition-meta"><div><span>Subset</span><strong>{{ subsetLabel(dataset) }}</strong></div><div><span>Created</span><strong>{{ formatDate(dataset.created_at) }}</strong></div><div><span>Updated</span><strong>{{ formatDate(dataset.updated_at) }}</strong></div></div></section>
        <section class="copy-boundary panel"><svg lucideInfo [size]="16"></svg><span>NeuroWeave stores paths, roles, counts, schema metadata, validation findings, and subset configuration only. It never stores full table contents, images, tensors, arrays, or binary previews.</span></section>
        @if (dataset.validation.warnings.length) { <section class="detail-panel panel"><header><span class="eyebrow">WARNINGS</span><h2>Registration notes</h2></header>@for (warning of dataset.validation.warnings; track warning.code + warning.message) { <p class="finding warning">{{ warning.message }}</p> }</section> }
      </div>
    }
  `,
  styles: [`
    .workspace-section{display:grid;gap:14px}.dataset-alert{display:flex;align-items:flex-start;gap:10px;padding:13px 15px;border-color:rgba(229,183,95,.4);color:var(--warning)}.dataset-alert p{margin:5px 0 0;color:var(--muted);font-size:10px;line-height:1.45}.workspace-card-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.workspace-card{display:grid;gap:6px;min-width:0;padding:14px}.workspace-card code{overflow-wrap:anywhere;font-size:10px}.workspace-card strong{font-size:18px}.workspace-card small,.muted{color:var(--muted);font-size:9px}.overview-columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.detail-panel{min-width:0;padding:15px}.detail-panel header{display:grid;gap:5px;margin-bottom:12px}.detail-panel h2{margin:0;font-size:14px}.binding-list,.source-list{display:grid;gap:5px}.binding-row{display:grid;grid-template-columns:100px minmax(0,1fr);gap:8px;padding:8px;border-bottom:1px solid var(--border);font-size:9px}.binding-row span,.source-row span,.definition-meta span{color:var(--muted)}.binding-row strong,.source-row code{overflow-wrap:anywhere}.source-row{display:grid;grid-template-columns:140px minmax(0,1fr);gap:10px;padding:9px;border:1px solid var(--border);border-radius:5px;background:#0c141b}.source-row div{display:grid;gap:3px}.source-row code{font-size:9px}.definition-meta{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px}.definition-meta div{display:grid;gap:4px;padding-top:9px;border-top:1px solid var(--border);font-size:9px}.copy-boundary{display:flex;gap:8px;align-items:flex-start;padding:12px;color:var(--muted);font-size:9px;line-height:1.45}.copy-boundary svg{flex:0 0 auto;color:var(--cyan)}.finding{margin:0;padding:8px;border-bottom:1px solid var(--border);font-size:9px}.finding.warning{color:var(--warning)}@media(max-width:900px){.workspace-card-grid,.overview-columns{grid-template-columns:1fr}.definition-meta{grid-template-columns:1fr}.source-row{grid-template-columns:1fr}}
  `],
})
export class DatasetOverviewPageComponent {
  readonly state = inject(DatasetWorkspaceState)

  bindingSummary(bindings: DatasetBinding[], empty = 'None'): string { return bindings.length ? bindings.map((binding) => this.bindingDetail(binding)).join(' + ') : empty }

  bindingDetail(binding: DatasetBinding): string {
    if (binding.type === 'columns') return binding.columns.join(', ')
    if (binding.type === 'array') return binding.keys.join(', ')
    if (binding.type === 'image') return 'Image samples'
    if (binding.type === 'text') return 'Text samples'
    return 'Folder labels'
  }

  subsetLabel(dataset: { subset_definition: { type: string; count?: number; start?: number; end?: number } }): string {
    if (dataset.subset_definition.type === 'all') return 'All samples'
    if (dataset.subset_definition.type === 'first_n') return `First ${dataset.subset_definition.count ?? 0}`
    return `Range ${dataset.subset_definition.start ?? 0}–${dataset.subset_definition.end ?? 0}`
  }

  formatDate(value: string): string { const date = new Date(value); return Number.isNaN(date.valueOf()) ? 'Unknown' : new Intl.DateTimeFormat(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(date) }
}
