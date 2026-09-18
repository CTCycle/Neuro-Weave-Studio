import { CommonModule } from '@angular/common'
import { Component, effect, inject, input, signal } from '@angular/core'
import { Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router'
import { LucideAlertTriangle, LucideArrowLeft, LucideBox, LucideTrash2 } from '@lucide/angular'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { datasetTypeLabel } from '../../domain/dataset-capabilities'
import { DatasetWorkspaceState } from './dataset-workspace.state'
import { ModalGuardDirective } from '../../shared/components/modal-guard.directive'

@Component({
  selector: 'nw-dataset-workspace-page',
  standalone: true,
  providers: [DatasetWorkspaceState],
  imports: [CommonModule, RouterLink, RouterLinkActive, RouterOutlet, LucideAlertTriangle, LucideArrowLeft, LucideBox, LucideTrash2, ModalGuardDirective],
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
          <div class="workspace-title-row"><div><div class="workspace-type"><span class="eyebrow">{{ typeLabel(dataset.dataset_type) }}</span>@if (dataset.availability.status === 'unavailable') { <span class="dataset-unavailable"><svg lucideAlertTriangle [size]="12"></svg>Source unavailable</span> }</div><h1>{{ dataset.name }}</h1><p>Registered metadata reference. NeuroWeave does not own or copy the underlying source.</p></div><div class="workspace-header-actions"><button class="icon-button danger-button" type="button" title="Delete dataset registration" aria-label="Delete dataset registration" (click)="deleteOpen.set(true)"><svg lucideTrash2 [size]="16"></svg></button></div></div>
          <div class="workspace-meta summary-strip"><span><svg lucideBox [size]="13"></svg>{{ dataset.selected_sample_count | number }} selected / {{ dataset.sample_count | number }} available</span><span>{{ dataset.format }}</span><span>{{ dataset.input_definition.bindings.length }} input binding{{ dataset.input_definition.bindings.length === 1 ? '' : 's' }}</span><span>{{ dataset.output_definition.bindings.length }} output binding{{ dataset.output_definition.bindings.length === 1 ? '' : 's' }}</span><span class="dataset-status" [class.ok]="dataset.validation.status === 'valid'">{{ dataset.validation.status === 'valid' ? 'Valid registration' : 'Needs review' }}</span></div>
          <nav class="workspace-tabs" aria-label="Dataset workspace sections"><a [routerLink]="['/datasets', dataset.id]" routerLinkActive="active" #overviewTab="routerLinkActive" [routerLinkActiveOptions]="{ exact: true }" [attr.aria-current]="overviewTab.isActive ? 'page' : null">Overview</a><a [routerLink]="['/datasets', dataset.id, 'validation']" routerLinkActive="active" #validationTab="routerLinkActive" [attr.aria-current]="validationTab.isActive ? 'page' : null">Validation</a><a [routerLink]="['/datasets', dataset.id, 'io']" routerLinkActive="active" #ioTab="routerLinkActive" [attr.aria-current]="ioTab.isActive ? 'page' : null">I/O</a><a [routerLink]="['/datasets', dataset.id, 'processing']" routerLinkActive="active" #processingTab="routerLinkActive" [attr.aria-current]="processingTab.isActive ? 'page' : null">Processing</a><a [routerLink]="['/datasets', dataset.id, 'split']" routerLinkActive="active" #splitTab="routerLinkActive" [attr.aria-current]="splitTab.isActive ? 'page' : null">Split</a></nav>
        </header>
        <div class="workspace-body"><router-outlet /></div>
        @if (deleteOpen()) { <div class="dialog-backdrop" nwModalGuard><section class="confirm-dialog panel" role="alertdialog" aria-modal="true" aria-labelledby="delete-dataset-title" (keydown.escape)="deleteOpen.set(false)"><span class="eyebrow">DELETE REGISTRATION</span><h2 id="delete-dataset-title">Delete “{{ dataset.name }}”?</h2><p>This removes only NeuroWeave’s metadata registration. The external files and folders will not be modified or deleted.</p>@if (deleteError()) { <p class="delete-dialog-error" role="alert">{{ deleteError() }}</p> }<div class="dialog-actions"><button class="secondary" type="button" (click)="deleteOpen.set(false)" [disabled]="deleteBusy()">Cancel</button><button class="primary danger-action" type="button" (click)="remove()" [disabled]="deleteBusy()">{{ deleteBusy() ? 'Deleting…' : 'Delete registration' }}</button></div></section></div> }
      </div>
    }
  `,
  styles: [`
    .summary-strip{row-gap:7px}.summary-strip>span{white-space:nowrap}.workspace-type{display:flex;align-items:center;gap:8px}.dataset-unavailable{display:inline-flex;align-items:center;gap:4px;color:var(--warning);font:500 8px 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase}.dataset-status{color:var(--warning)}.dataset-status.ok{color:var(--success,#63ca8f)}.danger-button{color:var(--error)}.delete-dialog-error{margin-top:10px!important;color:var(--error)!important}.danger-action{border-color:rgba(239,106,114,.6)!important;background:rgba(239,106,114,.14)!important}@media(max-width:1180px){.summary-strip{flex-wrap:wrap}}@media(max-width:700px){.workspace-tabs{overflow-x:auto;scrollbar-width:none}.workspace-tabs a{white-space:nowrap}}
  `],
})
export class DatasetWorkspacePageComponent {
  readonly datasetId = input.required<string>()
  readonly state = inject(DatasetWorkspaceState)
  private readonly resources = inject(AppResourcesService)
  private readonly router = inject(Router)
  readonly deleteOpen = signal(false)
  readonly deleteBusy = signal(false)
  readonly deleteError = signal<string | null>(null)

  constructor() { void this.resources.loadDatasets(); effect(() => { void this.state.load(this.datasetId()) }) }

  typeLabel(value: Parameters<typeof datasetTypeLabel>[0]): string { return datasetTypeLabel(value) }

  async remove(): Promise<void> {
    if (this.deleteBusy()) return
    this.deleteBusy.set(true)
    this.deleteError.set(null)
    try { await this.state.remove(); await this.router.navigate(['/datasets']) } catch (reason) { this.deleteError.set(reason instanceof Error ? reason.message : String(reason)); this.deleteBusy.set(false) }
  }
}
