import { Component, computed, inject, signal, ViewChild } from '@angular/core'
import type { ElementRef } from '@angular/core'
import { Router, RouterLink } from '@angular/router'
import { LucideBox, LucideFilePlus2, LucideFilter, LucideFolderOpen, LucideImport, LucideSearch, LucideTrash2 } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import type { ExecutionBackend, ProjectExportBundle, ProjectSummary } from '../../domain/types'
import { executionBackendShortLabels } from '../../domain/types'
import { CompilationStatusBadgeComponent, ValidationStatusBadgeComponent } from '../../shared/components/model-status-badge.component'

type CatalogFilter = 'all' | 'compiled' | 'uncompiled' | 'valid' | 'attention'

@Component({
  selector: 'nw-model-catalog-page',
  standalone: true,
  imports: [RouterLink, LucideBox, LucideFilePlus2, LucideFilter, LucideFolderOpen, LucideImport, LucideSearch, LucideTrash2, ValidationStatusBadgeComponent, CompilationStatusBadgeComponent],
  template: `
    @if (!resources.projectsReady()) {
      <section class="page-state panel"><span class="eyebrow">MODEL CATALOG</span><h1>Loading saved models…</h1></section>
    } @else {
      <div class="catalog-page">
        <header class="catalog-page-header">
          <div><span class="eyebrow">MODEL LIBRARY</span><h1>Saved models</h1><p>Browse architectures, review readiness, and open a dedicated model workflow.</p></div>
          <div class="catalog-header-actions">
            <a class="primary inline-action" routerLink="/builder"><svg lucideFilePlus2 [size]="16"></svg>New model</a>
            <button class="secondary" type="button" (click)="openImport()"><svg lucideImport [size]="16"></svg>Import model</button>
            <input #importInput type="file" accept="application/json,.json,.keras,application/vnd.keras.zip" hidden (change)="importModel(importInput.files?.[0] ?? null)" />
          </div>
        </header>
        <section class="catalog-controls panel">
          <label class="search-field"><svg lucideSearch [size]="16"></svg><input aria-label="Search models" placeholder="Search saved models" [value]="query()" (input)="query.set($any($event.target).value)" /></label>
          <div class="catalog-filter-label"><svg lucideFilter [size]="14"></svg><span>Filter</span></div>
          <div class="catalog-filter-tabs" role="tablist" aria-label="Model filters">
            @for (item of filters; track item) { <button type="button" role="tab" [attr.aria-selected]="filter() === item" [class.active]="filter() === item" (click)="filter.set(item)">{{ filterLabel(item) }}</button> }
          </div>
          <span class="catalog-count-text" aria-live="polite">{{ filteredProjects().length }} / {{ resources.projects().length }} models</span>
        </section>
        @if (resources.hasError('projects')) { <section class="catalog-inline-error panel"><strong>Model catalog unavailable.</strong><span>Saved models could not be read. Retry the request without losing the rest of the application.</span><button class="secondary" type="button" (click)="resources.retryProjects()">Retry catalog</button></section> }
        @if (filteredProjects().length === 0) {
          <section class="catalog-empty panel"><svg lucideFolderOpen [size]="28"></svg><h2>{{ resources.projects().length ? 'No models match this view' : resources.hasError('projects') ? 'Model library unavailable' : 'Your model library is empty' }}</h2><p>{{ resources.projects().length ? 'Try another search or filter.' : resources.hasError('projects') ? 'The saved model request failed. Use the retry action above to try again.' : 'Create an architecture in Model Builder, then it will appear here.' }}</p>@if (!resources.projects().length && !resources.hasError('projects')) { <a class="primary inline-action" routerLink="/builder"><svg lucideFilePlus2 [size]="16"></svg>Create your first model</a> }</section>
        } @else {
          <div class="model-card-grid">@for (project of filteredProjects(); track project.id) { <article class="model-card panel"><a class="model-card-link" [routerLink]="['/models', project.id]"><div class="model-card-top"><span class="eyebrow">MODEL</span><span class="revision-label">r{{ project.current_revision }}</span></div><div class="model-card-title"><h2>{{ project.name }}</h2><span class="model-card-arrow">→</span></div><p>{{ project.description || 'No description added.' }}</p><div class="model-card-meta"><nw-validation-status-badge [status]="project.validation_status"></nw-validation-status-badge><nw-compilation-status-badge [compiled]="project.has_generated_source"></nw-compilation-status-badge><div class="model-profile-meta"><span>Backend</span><strong>{{ backendLabel(project.execution_backend) }}</strong></div><div class="model-profile-meta"><span>API</span><strong>{{ project.keras_enabled ? 'Keras layer' : 'Native' }}</strong></div></div><div class="model-card-stats"><span><strong>{{ project.node_count }}</strong> nodes</span><span><strong>{{ project.edge_count }}</strong> connections</span></div><footer><span>Updated {{ formatUpdated(project.updated_at) }}</span><svg lucideBox [size]="13"></svg></footer></a><button class="icon-button danger-button model-card-delete" type="button" [disabled]="deleteBusy()" [attr.aria-label]="'Delete model ' + project.name" title="Delete model" (click)="openDelete(project)"><svg lucideTrash2 [size]="16"></svg></button></article> }</div>
        }
        @if (notice()) { <button class="notice catalog-notice" type="button" aria-label="Dismiss catalog notice" aria-live="polite" (click)="notice.set(null)">{{ notice() }}</button> }
        @if (deleteTarget(); as project) { <div class="dialog-backdrop"><section class="confirm-dialog panel" role="alertdialog" aria-modal="true" aria-labelledby="catalog-delete-model-title" [attr.aria-busy]="deleteBusy()"><span class="eyebrow">DELETE MODEL</span><h2 id="catalog-delete-model-title">Delete “{{ project.name }}”?</h2><p>This permanently removes the saved architecture, revisions, and generated snapshots. This action cannot be undone.</p>@if (deleteError()) { <p class="delete-dialog-error" role="alert">{{ deleteError() }}</p> }<div class="dialog-actions"><button class="secondary" type="button" (click)="cancelDelete()" [disabled]="deleteBusy()">Cancel</button><button class="primary danger-action" type="button" (click)="confirmDelete()" [disabled]="deleteBusy()">{{ deleteBusy() ? 'Deleting…' : 'Delete permanently' }}</button></div></section></div> }
      </div>
    }
  `,
})
export class ModelCatalogPageComponent {
  @ViewChild('importInput') private importInput?: ElementRef<HTMLInputElement>
  readonly resources = inject(AppResourcesService)
  private readonly api = inject(ApiService)
  private readonly router = inject(Router)
  readonly query = signal('')
  readonly filter = signal<CatalogFilter>('all')
  readonly notice = signal<string | null>(null)
  readonly deleteTarget = signal<ProjectSummary | null>(null)
  readonly deleteBusy = signal(false)
  readonly deleteError = signal<string | null>(null)
  readonly filters: CatalogFilter[] = ['all', 'compiled', 'uncompiled', 'valid', 'attention']

  constructor() {
    void this.resources.loadProjects().then(() => performance.mark('nw:initial-route:ready'))
  }
  readonly filteredProjects = computed(() => {
    const search = this.query().trim().toLowerCase()
    const selected = this.filter()
    return this.resources.projects().filter((project) => {
      const text = `${project.name} ${project.description} ${project.execution_backend} ${project.keras_enabled ? 'keras' : 'native'}`.toLowerCase()
      const compiled = project.has_generated_source
      const matchesFilter = selected === 'all' || (selected === 'compiled' && compiled) || (selected === 'uncompiled' && !compiled) || (selected === 'valid' && project.validation_status === 'valid') || (selected === 'attention' && project.validation_status !== 'valid')
      return text.includes(search) && matchesFilter
    })
  })

  openImport(): void { this.importInput?.nativeElement.click() }

  async importModel(file: File | null): Promise<void> {
    if (!file) return
    try {
      let projectId: string
      const isKeras = file.name.toLowerCase().endsWith('.keras')
      if (isKeras) {
        const preferred = this.resources.backendStatuses().find((status) => status.id === 'pytorch' && status.available)?.id ?? this.resources.backendStatuses().find((status) => status.available)?.id ?? 'pytorch'
        const imported = await this.api.importKerasArchitecture(file, preferred, this.importName(file.name))
        projectId = imported.project.id
        if (imported.warnings.length) this.notice.set(imported.warnings.join(' '))
      } else {
        const bundle = JSON.parse(await file.text()) as ProjectExportBundle
        projectId = (await this.api.importProject(bundle)).id
      }
      await this.resources.refreshProjects()
      await this.router.navigate(['/models', projectId])
    } catch (reason) {
      this.notice.set(`Import failed: ${reason instanceof Error ? reason.message : String(reason)}`)
    } finally {
      if (this.importInput) this.importInput.nativeElement.value = ''
    }
  }

  openDelete(project: ProjectSummary): void {
    this.deleteTarget.set(project)
    this.deleteError.set(null)
  }

  cancelDelete(): void {
    if (this.deleteBusy()) return
    this.deleteTarget.set(null)
    this.deleteError.set(null)
  }

  async confirmDelete(): Promise<void> {
    const project = this.deleteTarget()
    if (!project || this.deleteBusy()) return
    this.deleteBusy.set(true)
    this.deleteError.set(null)
    try {
      await this.api.deleteProject(project.id)
      await this.resources.refreshProjects()
      this.deleteTarget.set(null)
      this.notice.set(this.resources.hasError('projects') ? `Deleted “${project.name}”, but the catalog could not be refreshed. Use Retry catalog to reconcile the list.` : `Deleted “${project.name}”.`)
    } catch (reason) {
      this.deleteError.set(`Delete failed: ${reason instanceof Error ? reason.message : String(reason)}`)
    } finally {
      this.deleteBusy.set(false)
    }
  }

  filterLabel(value: CatalogFilter): string { return value === 'all' ? 'All models' : value === 'attention' ? 'Needs attention' : value[0].toUpperCase() + value.slice(1) }
  backendLabel(value: ExecutionBackend): string { return executionBackendShortLabels[value] }
  formatUpdated(value: string): string { const date = new Date(value); return Number.isNaN(date.valueOf()) ? 'recently' : new Intl.DateTimeFormat(undefined, { year: 'numeric', month: 'short', day: 'numeric' }).format(date) }
  private importName(value: string): string { return value.replace(/\.(keras|json)$/i, '').replaceAll(/[_-]+/g, ' ').trim() || 'Imported Keras architecture' }
}
