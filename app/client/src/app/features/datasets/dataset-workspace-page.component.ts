import { CommonModule } from '@angular/common'
import { Component, effect, inject, input, signal } from '@angular/core'
import { Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router'
import { LucideAlertTriangle, LucideArrowLeft, LucideBox, LucideEdit3, LucideTrash2 } from '@lucide/angular'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { datasetTypeLabel } from '../../domain/dataset-capabilities'
import type { DatasetDocument } from '../../domain/dataset.types'
import { DatasetWizardComponent } from './dataset-wizard.component'
import { DatasetWorkspaceState } from './dataset-workspace.state'
import { ModalGuardDirective } from '../../shared/components/modal-guard.directive'

@Component({
  selector: 'nw-dataset-workspace-page',
  standalone: true,
  providers: [DatasetWorkspaceState],
  imports: [CommonModule, RouterLink, RouterLinkActive, RouterOutlet, LucideAlertTriangle, LucideArrowLeft, LucideBox, LucideEdit3, LucideTrash2, DatasetWizardComponent, ModalGuardDirective],
  template: `
    @if (state.loading()) {
      <section class="page-state panel"><span class="eyebrow">DATASET WORKSPACE</span><h1>Loading dataset…</h1></section>
    } @else if (state.error() || !state.dataset()) {
      <section class="page-state panel"><span class="eyebrow">DATASET WORKSPACE</span><h1>Dataset unavailable</h1><p>{{ state.error() || 'The requested dataset could not be found.' }}</p><a class="primary inline-action" routerLink="/datasets">Back to datasets</a></section>
    } @else {
      @let dataset = state.dataset()!;
      <div class="dataset-workspace-page">
        <header class="workspace-header">
          <div class="workspace-header-top"><a class="back-link" routerLink="/datasets"><svg lucideArrowLeft [size]="15"></svg>All datasets</a><span class="workspace-kicker">DATASET WORKSPACE</span></div>
          <div class="workspace-title-row"><div><div class="workspace-type"><span class="eyebrow">{{ typeLabel(dataset.dataset_type) }}</span>@if (dataset.availability.status === 'unavailable') { <span class="dataset-unavailable"><svg lucideAlertTriangle [size]="12"></svg>Source unavailable</span> }</div><h1>{{ dataset.name }}</h1><p>Registered metadata reference. NeuroWeave does not own or copy the underlying source.</p></div><div class="workspace-header-actions"><button class="secondary" type="button" (click)="editOpen.set(true)"><svg lucideEdit3 [size]="15"></svg>Edit / relink</button><button class="icon-button danger-button" type="button" title="Delete dataset registration" aria-label="Delete dataset registration" (click)="deleteOpen.set(true)"><svg lucideTrash2 [size]="16"></svg></button></div></div>
          <div class="workspace-meta summary-strip"><span><svg lucideBox [size]="13"></svg>{{ dataset.selected_sample_count | number }} selected / {{ dataset.sample_count | number }} available</span><span>{{ dataset.format }}</span><span>{{ dataset.input_definition.bindings.length }} input source{{ dataset.input_definition.bindings.length === 1 ? '' : 's' }}</span><span>{{ dataset.output_definition.bindings.length }} output source{{ dataset.output_definition.bindings.length === 1 ? '' : 's' }}</span><span class="dataset-status" [class.ok]="dataset.validation.status === 'valid'">{{ dataset.validation.status === 'valid' ? 'Valid registration' : 'Invalid' }}</span></div>
          <nav class="workspace-tabs" aria-label="Dataset workspace sections"><a [routerLink]="['/datasets', dataset.id]" routerLinkActive="active" [routerLinkActiveOptions]="{ exact: true }">Overview</a><a [routerLink]="['/datasets', dataset.id, 'structure']" routerLinkActive="active">Structure</a><a [routerLink]="['/datasets', dataset.id, 'processing']" routerLinkActive="active">Processing <small>Later</small></a><a [routerLink]="['/datasets', dataset.id, 'split']" routerLinkActive="active">Split <small>Later</small></a></nav>
        </header>
        <div class="workspace-body"><router-outlet /></div>
        @if (editOpen()) { <nw-dataset-wizard [open]="true" [editing]="dataset" (closed)="editOpen.set(false)" (saved)="onSaved($event)" /> }
        @if (deleteOpen()) { <div class="dialog-backdrop" nwModalGuard><section class="confirm-dialog panel" role="alertdialog" aria-modal="true" aria-labelledby="delete-dataset-title"><span class="eyebrow">DELETE REGISTRATION</span><h2 id="delete-dataset-title">Delete “{{ dataset.name }}”?</h2><p>This removes only NeuroWeave’s metadata registration. The external files and folders will not be modified or deleted.</p>@if (deleteError()) { <p class="delete-dialog-error" role="alert">{{ deleteError() }}</p> }<div class="dialog-actions"><button class="secondary" type="button" (click)="deleteOpen.set(false)" [disabled]="deleteBusy()">Cancel</button><button class="primary danger-action" type="button" (click)="remove()" [disabled]="deleteBusy()">{{ deleteBusy() ? 'Deleting…' : 'Delete registration' }}</button></div></section></div> }
      </div>
    }
  `,
  styles: [`
    .summary-strip{row-gap:7px}.summary-strip>span{white-space:nowrap}.workspace-type{display:flex;align-items:center;gap:8px}.dataset-unavailable{display:inline-flex;align-items:center;gap:4px;color:var(--warning);font:500 8px 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase}.dataset-status{color:var(--warning)}.dataset-status.ok{color:var(--success,#63ca8f)}.workspace-tabs small{margin-left:4px;color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase}@media(max-width:1180px){.summary-strip{flex-wrap:wrap}}
  `],
})
export class DatasetWorkspacePageComponent {
  readonly datasetId = input.required<string>()
  readonly state = inject(DatasetWorkspaceState)
  private readonly resources = inject(AppResourcesService)
  private readonly router = inject(Router)
  readonly editOpen = signal(false)
  readonly deleteOpen = signal(false)
  readonly deleteBusy = signal(false)
  readonly deleteError = signal<string | null>(null)
  constructor() { void this.resources.loadDatasets(); effect(() => { void this.state.load(this.datasetId()) }) }

  typeLabel(value: Parameters<typeof datasetTypeLabel>[0]): string { return datasetTypeLabel(value) }

  async onSaved(_dataset: DatasetDocument): Promise<void> { this.editOpen.set(false); await this.state.refresh() }

  async remove(): Promise<void> {
    if (this.deleteBusy()) return
    this.deleteBusy.set(true)
    this.deleteError.set(null)
    try { await this.state.remove(); await this.router.navigate(['/datasets']) } catch (reason) { this.deleteError.set(reason instanceof Error ? reason.message : String(reason)); this.deleteBusy.set(false) }
  }
}
