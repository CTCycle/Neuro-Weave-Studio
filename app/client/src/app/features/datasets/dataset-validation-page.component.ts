import { CommonModule } from '@angular/common'
import { Component, inject, signal } from '@angular/core'
import { LucideAlertTriangle, LucideCheck, LucideRefreshCw } from '@lucide/angular'
import { DatasetWorkspaceState } from './dataset-workspace.state'

@Component({
  selector: 'nw-dataset-validation-page',
  standalone: true,
  imports: [CommonModule, LucideAlertTriangle, LucideCheck, LucideRefreshCw],
  template: `
    @if (state.dataset(); as dataset) {
      <div class="workspace-section validation-page">
        <section class="validation-header panel"><div><span class="eyebrow">CURRENT VALIDATION</span><h2>{{ dataset.validation.status === 'valid' ? 'Source definition is valid' : 'Source definition needs attention' }}</h2><p>Validation inspects the external source again and persists the result with this dataset registration.</p></div><button class="primary" type="button" [disabled]="busy()" (click)="revalidate()"><svg lucideRefreshCw [size]="14"></svg>{{ busy() ? 'Revalidating…' : 'Revalidate' }}</button></section>
        @if (error()) { <p class="page-error" role="alert">{{ error() }}</p> }
        <section class="validation-stat-grid"><div class="panel"><span class="eyebrow">STATUS</span><strong class="status" [class.valid]="dataset.validation.status === 'valid'">@if (dataset.validation.status === 'valid') { <svg lucideCheck [size]="14"></svg> } @else { <svg lucideAlertTriangle [size]="14"></svg> }{{ dataset.validation.status === 'valid' ? 'Valid' : 'Invalid' }}</strong><small>Checked {{ formatDate(dataset.validation.checked_at) }}</small></div><div class="panel"><span class="eyebrow">SOURCE</span><strong>{{ dataset.availability.status === 'available' ? 'Available' : 'Unavailable' }}</strong><small>{{ dataset.sources.length }} registered source{{ dataset.sources.length === 1 ? '' : 's' }}</small></div><div class="panel"><span class="eyebrow">POPULATION</span><strong>{{ dataset.selected_sample_count | number }} / {{ dataset.sample_count | number }}</strong><small>Selected / inspected samples</small></div></section>
        @if (dataset.availability.status === 'unavailable') { <section class="finding-panel panel unavailable"><svg lucideAlertTriangle [size]="16"></svg><div><strong>Source availability changed</strong><p>{{ dataset.availability.unavailable_source_ids.join(', ') }} cannot currently be read. The previous inspection snapshot is retained until a source can be inspected again.</p></div></section> }
        <section class="finding-panel panel"><header><span class="eyebrow">ERRORS</span><strong>{{ dataset.validation.errors.length ? dataset.validation.errors.length + ' blocking finding' + (dataset.validation.errors.length === 1 ? '' : 's') : 'No blocking findings' }}</strong></header>@for (finding of dataset.validation.errors; track finding.code + finding.message) { <div class="finding error"><strong>{{ finding.code }}</strong><span>{{ finding.message }}</span></div> }@empty { <p class="muted">The stored definition passed the latest validation.</p> }</section>
        <section class="finding-panel panel"><header><span class="eyebrow">WARNINGS</span><strong>{{ dataset.validation.warnings.length ? dataset.validation.warnings.length + ' note' + (dataset.validation.warnings.length === 1 ? '' : 's') : 'No warnings' }}</strong></header>@for (finding of dataset.validation.warnings; track finding.code + finding.message) { <div class="finding warning"><strong>{{ finding.code }}</strong><span>{{ finding.message }}</span></div> }@empty { <p class="muted">No warnings were stored.</p> }</section>
      </div>
    }
  `,
  styles: [`
    .workspace-section{display:grid;gap:14px}.validation-header{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;padding:16px}.validation-header h2{margin:5px 0}.validation-header p{margin:0;color:var(--muted);font-size:10px;line-height:1.45}.validation-header button{display:inline-flex;align-items:center;gap:7px;flex:0 0 auto}.validation-stat-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.validation-stat-grid>div{display:grid;gap:6px;padding:14px}.validation-stat-grid strong{display:flex;align-items:center;gap:6px;font-size:16px}.validation-stat-grid small,.muted{color:var(--muted);font-size:9px}.status{color:var(--warning)}.status.valid{color:var(--success,#63ca8f)}.finding-panel{padding:15px}.finding-panel header{display:flex;justify-content:space-between;gap:12px;margin-bottom:10px}.finding-panel header strong{font-size:10px}.finding{display:grid;grid-template-columns:170px minmax(0,1fr);gap:10px;padding:9px 0;border-top:1px solid var(--border);font-size:9px}.finding span{color:var(--muted);line-height:1.4}.finding.error strong{color:var(--error)}.finding.warning strong{color:var(--warning)}.finding-panel>p{margin:0}.finding-panel.unavailable{display:flex;gap:9px;align-items:flex-start;border-color:rgba(229,183,95,.4);color:var(--warning);font-size:9px;line-height:1.45}.finding-panel.unavailable p{margin:5px 0 0;color:var(--muted)}.page-error{margin:0;padding:9px;border:1px solid rgba(239,106,114,.45);border-radius:5px;background:rgba(239,106,114,.08);color:var(--error);font-size:10px}@media(max-width:700px){.validation-header{display:grid}.validation-header button{justify-self:start}.validation-stat-grid{grid-template-columns:1fr}.finding{grid-template-columns:1fr;gap:4px}}
  `],
})
export class DatasetValidationPageComponent {
  readonly state = inject(DatasetWorkspaceState)
  readonly busy = signal(false)
  readonly error = signal<string | null>(null)

  async revalidate(): Promise<void> {
    if (this.busy()) return
    this.busy.set(true)
    this.error.set(null)
    try { await this.state.revalidate() } catch (reason) { this.error.set(reason instanceof Error ? reason.message : String(reason)) } finally { this.busy.set(false) }
  }

  formatDate(value: string): string { const date = new Date(value); return Number.isNaN(date.valueOf()) ? 'Unknown' : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date) }
}
