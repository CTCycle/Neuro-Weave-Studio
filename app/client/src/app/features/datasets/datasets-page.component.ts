import { CommonModule } from '@angular/common'
import { Component, computed, inject, signal } from '@angular/core'
import { Router, RouterLink } from '@angular/router'
import { LucideDatabase, LucideFilePlus2, LucideRefreshCw, LucideSearch, LucideTrash2 } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import type { DatasetDocument, DatasetSummary, DatasetType } from '../../domain/dataset.types'
import { datasetTypeLabel } from '../../domain/dataset-capabilities'
import { DatasetWizardComponent } from './dataset-wizard.component'
import { ModalGuardDirective } from '../../shared/components/modal-guard.directive'

type AvailabilityFilter = 'all' | 'available' | 'unavailable'

@Component({
  selector: 'nw-datasets-page',
  standalone: true,
  imports: [CommonModule, RouterLink, LucideDatabase, LucideFilePlus2, LucideRefreshCw, LucideSearch, LucideTrash2, DatasetWizardComponent, ModalGuardDirective],
  template: `
    <div class="datasets-page">
      <header class="catalog-page-header">
        <div>
          <span class="eyebrow page-eyebrow">DATA WORKSPACE</span>
          <h1>Datasets</h1>
          <p>Register external data sources by reference. NeuroWeave stores lightweight metadata and never copies source contents.</p>
        </div>
        <button class="primary" type="button" (click)="openWizard()"><svg lucideFilePlus2 [size]="16"></svg>Register dataset</button>
      </header>

      <section class="catalog-toolbar panel" aria-label="Dataset filters">
        <label class="search-field"><svg lucideSearch [size]="15"></svg><span class="sr-only">Search datasets</span><input [value]="query()" placeholder="Search by name, type, or path" (input)="query.set($any($event.target).value)" /></label>
        <div class="filter-group" role="group" aria-label="Availability"><button type="button" [class.active]="availability() === 'all'" (click)="availability.set('all')">All <span>{{ resources.datasets().length }}</span></button><button type="button" [class.active]="availability() === 'available'" (click)="availability.set('available')">Available <span>{{ availableCount() }}</span></button><button type="button" [class.active]="availability() === 'unavailable'" (click)="availability.set('unavailable')">Unavailable <span>{{ unavailableCount() }}</span></button></div>
      </section>

      @if (!resources.datasetsReady() && !resources.datasets().length) {
        <section class="catalog-state panel"><span class="spinner"></span><p>Loading registered datasets…</p></section>
      } @else if (resources.hasError('datasets')) {
        <section class="catalog-state panel"><svg lucideDatabase [size]="28"></svg><h2>Dataset catalog unavailable</h2><p>{{ resources.errors()['datasets'] }}</p><button class="secondary" type="button" (click)="resources.retryDatasets()"><svg lucideRefreshCw [size]="14"></svg>Retry</button></section>
      } @else if (!resources.datasets().length) {
        <section class="catalog-state panel"><svg lucideDatabase [size]="30"></svg><h2>No registered datasets</h2><p>Register a table, image folder, text source, array, or composed dataset to create a durable metadata reference.</p></section>
      } @else if (!filteredDatasets().length) {
        <section class="catalog-state panel"><svg lucideSearch [size]="28"></svg><h2>No matching datasets</h2><p>Adjust the search or availability filter.</p></section>
      } @else {
        <div class="dataset-grid">
          @for (dataset of filteredDatasets(); track dataset.id) {
            <article class="dataset-card panel" [class.unavailable]="dataset.availability.status === 'unavailable'">
              <div class="dataset-card-top"><span class="eyebrow">{{ typeLabel(dataset.dataset_type) }}</span><span class="availability" [class.bad]="dataset.availability.status === 'unavailable'">{{ dataset.availability.status === 'available' ? 'AVAILABLE' : 'SOURCE UNAVAILABLE' }}</span></div>
              <a class="dataset-card-link" [routerLink]="['/datasets', dataset.id]">
                <div class="dataset-card-title"><div class="dataset-icon"><svg lucideDatabase [size]="18"></svg></div><div><h2>{{ dataset.name }}</h2><code [title]="dataset.primary_path">{{ dataset.primary_path }}</code></div></div>
                <div class="dataset-stats"><div><span>SELECTED</span><strong>{{ dataset.selected_sample_count | number }}</strong><small>of {{ dataset.sample_count | number }} available</small></div><div><span>FORMAT</span><strong>{{ dataset.format }}</strong><small>{{ dataset.validation_status === 'valid' ? 'Validated' : 'Needs review' }}</small></div></div>
                <div class="dataset-roles"><span><b>Input</b> {{ dataset.input_summary || '—' }}</span><span><b>Output</b> {{ dataset.output_summary || 'None · input only' }}</span></div>
              </a>
              <footer class="dataset-card-footer"><span>Updated {{ formatDate(dataset.updated_at) }}</span><button class="icon-button danger-button" type="button" title="Delete registration" [attr.aria-label]="'Delete registration for ' + dataset.name" (click)="askDelete(dataset, $event)"><svg lucideTrash2 [size]="15"></svg></button></footer>
            </article>
          }
        </div>
      }

      @if (notice()) { <p class="notice dataset-notice" role="status">{{ notice() }}</p> }
      @if (wizardOpen()) { <nw-dataset-wizard [open]="true" (closed)="closeWizard()" (saved)="onSaved($event)" /> }
      @if (deleteTarget(); as target) {
        <div class="dialog-backdrop" nwModalGuard (click)="cancelDelete()"><section class="confirm-dialog panel" role="dialog" aria-modal="true" aria-labelledby="delete-title" (click)="$event.stopPropagation()"><h2 id="delete-title">Delete registration?</h2><p><strong>{{ target.name }}</strong> will be removed from NeuroWeave. The registered source files and folders remain untouched.</p><div class="dialog-actions"><button class="secondary" type="button" (click)="cancelDelete()">Cancel</button><button class="danger" type="button" [disabled]="deleting()" (click)="confirmDelete()">{{ deleting() ? 'Deleting…' : 'Delete registration' }}</button></div></section></div>
      }
    </div>
  `,
  styles: [`
    :host{display:block}.datasets-page{display:grid;gap:16px}.catalog-page-header{display:flex;align-items:flex-end;justify-content:space-between;gap:18px}.catalog-page-header h1{margin:6px 0}.catalog-page-header p{max-width:620px;margin:0;color:var(--muted);font-size:11px;line-height:1.5}.catalog-page-header button,.catalog-state button,.catalog-toolbar button{display:inline-flex;align-items:center;gap:7px}.catalog-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 12px}.search-field{display:flex;align-items:center;gap:8px;min-width:min(420px,100%);color:var(--muted)}.search-field input{width:100%;border:0;outline:0;background:transparent;color:var(--text);font:inherit}.filter-group{display:flex;gap:4px;flex-wrap:wrap}.filter-group button{min-height:28px;border:1px solid transparent;border-radius:5px;background:transparent;color:var(--muted);padding:0 9px;font-size:9px}.filter-group button.active{border-color:var(--cyan);color:var(--text);background:rgba(101,201,212,.08)}.filter-group span{color:var(--accent-bright);font-family:'IBM Plex Mono',Consolas,monospace}.catalog-state{display:grid;justify-items:center;gap:9px;padding:56px 24px;text-align:center}.catalog-state svg{color:var(--cyan)}.catalog-state h2,.catalog-state p{margin:0}.catalog-state p{max-width:480px;color:var(--muted);font-size:10px;line-height:1.5}.spinner{width:20px;height:20px;border:2px solid var(--border);border-top-color:var(--cyan);border-radius:50%;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}.dataset-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px}.dataset-card{display:grid;gap:12px;padding:14px;min-width:0}.dataset-card.unavailable{border-color:rgba(239,106,114,.5)}.dataset-card-top,.dataset-card-footer{display:flex;align-items:center;justify-content:space-between;gap:10px}.availability{color:var(--success,#63ca8f);font:500 8px 'IBM Plex Mono',Consolas,monospace}.availability.bad{color:var(--error)}.dataset-card-link{display:grid;gap:13px;color:inherit;text-decoration:none}.dataset-card-title{display:flex;align-items:flex-start;gap:10px;min-width:0}.dataset-icon{display:grid;place-items:center;flex:0 0 34px;height:34px;border:1px solid rgba(101,201,212,.35);border-radius:6px;color:var(--cyan)}.dataset-card-title h2{margin:1px 0 5px;font-size:14px;overflow-wrap:anywhere}.dataset-card-title code{display:block;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-size:9px}.dataset-stats{display:grid;grid-template-columns:1fr 1fr;gap:8px}.dataset-stats>div{display:grid;gap:4px;padding:9px;border:1px solid var(--border);background:#0c141b;border-radius:5px}.dataset-stats span,.dataset-roles b{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace;letter-spacing:.04em}.dataset-stats strong{font-size:14px}.dataset-stats small{color:var(--muted);font-size:9px}.dataset-roles{display:grid;gap:5px;color:var(--text);font-size:10px}.dataset-roles span{overflow-wrap:anywhere}.dataset-card-footer{padding-top:8px;border-top:1px solid var(--border);color:var(--muted);font-size:9px}.danger-button{color:var(--error)}.dataset-notice{justify-self:end}.confirm-dialog{width:min(440px,calc(100vw - 32px));padding:20px}.confirm-dialog h2{margin:0 0 8px}.confirm-dialog p{margin:0;color:var(--muted);font-size:10px;line-height:1.5}.dialog-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:18px}.danger{border:1px solid rgba(239,106,114,.6);border-radius:5px;background:rgba(239,106,114,.14);color:var(--text);padding:8px 12px;font-size:10px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@media(max-width:760px){.catalog-page-header,.catalog-toolbar{align-items:stretch;flex-direction:column}.search-field{min-width:0}.dataset-grid{grid-template-columns:1fr}}
  `],
})
export class DatasetsPageComponent {
  readonly resources = inject(AppResourcesService)
  private readonly api = inject(ApiService)
  private readonly router = inject(Router)
  readonly query = signal('')
  readonly availability = signal<AvailabilityFilter>('all')
  readonly wizardOpen = signal(false)
  readonly deleteTarget = signal<DatasetSummary | null>(null)
  readonly deleting = signal(false)
  readonly notice = signal<string | null>(null)
  readonly availableCount = computed(() => this.resources.datasets().filter((dataset) => dataset.availability.status === 'available').length)
  readonly unavailableCount = computed(() => this.resources.datasets().filter((dataset) => dataset.availability.status === 'unavailable').length)
  readonly filteredDatasets = computed(() => {
    const query = this.query().trim().toLowerCase()
    return this.resources.datasets().filter((dataset) => {
      if (this.availability() !== 'all' && dataset.availability.status !== this.availability()) return false
      if (!query) return true
      return [dataset.name, dataset.dataset_type, dataset.primary_path, dataset.format, dataset.input_summary, dataset.output_summary].some((value) => value.toLowerCase().includes(query))
    })
  })

  constructor() { void this.resources.loadDatasets() }

  openWizard(): void { this.notice.set(null); this.wizardOpen.set(true) }
  closeWizard(): void { this.wizardOpen.set(false) }
  typeLabel(type: DatasetType): string { return datasetTypeLabel(type) }
  formatDate(value: string): string { return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date(value)) }

  onSaved(_dataset: DatasetDocument): void {
    this.wizardOpen.set(false)
    this.notice.set('Dataset registration saved. Source files remain in place.')
    void this.resources.refreshDatasets()
  }

  askDelete(dataset: DatasetSummary, event: Event): void { event.stopPropagation(); this.deleteTarget.set(dataset) }
  cancelDelete(): void { if (!this.deleting()) this.deleteTarget.set(null) }
  async confirmDelete(): Promise<void> {
    const target = this.deleteTarget()
    if (!target || this.deleting()) return
    this.deleting.set(true)
    try {
      await this.api.deleteDataset(target.id)
      this.deleteTarget.set(null)
      this.notice.set('Registration deleted. Source files were not changed.')
      await this.resources.refreshDatasets()
      if (this.router.url.includes(`/datasets/${target.id}`)) await this.router.navigate(['/datasets'])
    } catch (reason) {
      this.notice.set(reason instanceof Error ? reason.message : String(reason))
    } finally { this.deleting.set(false) }
  }
}
