import { Component, computed, effect, inject, input, signal } from '@angular/core'
import { Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router'
import { LucideArrowLeft, LucideBox, LucideCopy, LucideFileCode2, LucideGitBranch, LucideSettings2, LucideTrash2 } from '@lucide/angular'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { summarizeGraphIo } from '../../domain/graph-io'
import { implementationLabel } from '../../domain/types'
import { CompilationStatusBadgeComponent, ValidationStatusBadgeComponent } from '../../shared/components/model-status-badge.component'
import { ModelWorkspaceState } from './model-workspace.state'
import { ModalGuardDirective } from '../../shared/components/modal-guard.directive'

@Component({
  selector: 'nw-model-workspace-page',
  standalone: true,
  providers: [ModelWorkspaceState],
  imports: [RouterLink, RouterLinkActive, RouterOutlet, LucideArrowLeft, LucideBox, LucideCopy, LucideFileCode2, LucideGitBranch, LucideSettings2, LucideTrash2, ValidationStatusBadgeComponent, CompilationStatusBadgeComponent, ModalGuardDirective],
  template: `
    @if (state.loading()) {
      <section class="page-state panel"><span class="eyebrow">MODEL WORKSPACE</span><h1>Loading model…</h1></section>
    } @else if (state.error() || !state.project()) {
      <section class="page-state panel"><span class="eyebrow">MODEL WORKSPACE</span><h1>Model unavailable</h1><p>{{ state.error() || 'The requested model could not be found.' }}</p><a class="primary inline-action" routerLink="/models">Back to models</a></section>
    } @else {
      @let project = state.project()!;
      @let io = ioSummary();
      <div class="model-workspace-page">
        <header class="workspace-header">
          <div class="workspace-header-top"><a class="back-link" routerLink="/models"><svg lucideArrowLeft [size]="15"></svg>All models</a><span class="workspace-kicker">MODEL WORKSPACE</span></div>
          <div class="workspace-title-row"><div><h1>{{ project.name }}</h1><p>{{ project.description || 'No description added.' }}</p></div><div class="workspace-header-actions"><a class="secondary inline-action" [routerLink]="['/builder', project.id]"><svg lucideSettings2 [size]="15"></svg>Edit architecture</a><button class="secondary" type="button" (click)="duplicate()"><svg lucideCopy [size]="15"></svg>Duplicate</button><button class="icon-button danger-button" type="button" title="Delete model" aria-label="Delete model" (click)="deleteOpen.set(true)"><svg lucideTrash2 [size]="16"></svg></button></div></div>
          <div class="workspace-meta summary-strip">
            <nw-validation-status-badge [status]="project.validation_status"></nw-validation-status-badge>
            <nw-compilation-status-badge [compiled]="!!project.generated_source" [target]="targetLabel(project)"></nw-compilation-status-badge>
            <span><svg lucideGitBranch [size]="13"></svg>Revision {{ project.current_revision }}</span>
            <span><svg lucideBox [size]="13"></svg>{{ io.inputCount }} input{{ io.inputCount === 1 ? '' : 's' }} · {{ io.outputCount }} output{{ io.outputCount === 1 ? '' : 's' }}</span>
            <span><svg lucideBox [size]="13"></svg>{{ project.graph.nodes.length }} nodes · {{ project.graph.edges.length }} connections</span>
            <span><svg lucideFileCode2 [size]="13"></svg>{{ targetLabel(project) }}</span>
          </div>
          <nav class="workspace-tabs" aria-label="Model workspace sections"><a [routerLink]="['/models', project.id]" routerLinkActive="active" ariaCurrentWhenActive="page" [routerLinkActiveOptions]="{ exact: true }">Overview</a><a [routerLink]="['/models', project.id, 'compilation']" routerLinkActive="active" ariaCurrentWhenActive="page">Compilation</a><a [routerLink]="['/models', project.id, 'validation']" routerLinkActive="active" ariaCurrentWhenActive="page">Validation</a><a [routerLink]="['/models', project.id, 'export']" routerLinkActive="active" ariaCurrentWhenActive="page">Export</a></nav>
        </header>
        <div class="workspace-body"><router-outlet /></div>
        @if (deleteOpen()) { <div class="dialog-backdrop" nwModalGuard><section class="confirm-dialog panel" role="alertdialog" aria-modal="true" aria-labelledby="delete-model-title"><span class="eyebrow">DELETE MODEL</span><h2 id="delete-model-title">Delete “{{ project.name }}”?</h2><p>This permanently removes the saved architecture, revisions, and generated snapshots. This action cannot be undone.</p><div class="dialog-actions"><button class="secondary" type="button" (click)="deleteOpen.set(false)" [disabled]="deleteBusy()">Cancel</button><button class="primary danger-action" type="button" (click)="remove()" [disabled]="deleteBusy()">{{ deleteBusy() ? 'Deleting…' : 'Delete permanently' }}</button></div></section></div> }
      </div>
    }
  `,
  styles: [`
    .summary-strip { row-gap:7px; }
    .summary-strip > span { white-space:nowrap; }
    @media (max-width: 1180px) { .summary-strip { flex-wrap:wrap; } }
  `],
})
export class ModelWorkspacePageComponent {
  readonly modelId = input.required<string>()
  readonly state = inject(ModelWorkspaceState)
  private readonly resources = inject(AppResourcesService)
  private readonly router = inject(Router)
  readonly deleteOpen = signal(false)
  readonly deleteBusy = signal(false)
  readonly ioSummary = computed(() => {
    const project = this.state.project()
    return project ? summarizeGraphIo(project.graph, this.resources.manifests()) : { inputCount: 0, outputCount: 0, inputs: [], outputs: [] }
  })

  constructor() {
    void this.resources.loadCatalog()
    effect(() => { void this.state.load(this.modelId()) })
  }

  targetLabel(project: { execution_backend: 'tensorflow' | 'pytorch' | 'jax'; keras_enabled: boolean }): string { return implementationLabel(project.execution_backend, project.keras_enabled) }

  async duplicate(): Promise<void> {
    try { await this.router.navigate(['/models', (await this.state.duplicate()).id]) } catch (reason) { this.state.error.set(`Duplicate failed: ${reason instanceof Error ? reason.message : String(reason)}`) }
  }

  async remove(): Promise<void> {
    this.deleteBusy.set(true)
    try { await this.state.remove(); await this.router.navigate(['/models']) } catch (reason) { this.state.error.set(`Delete failed: ${reason instanceof Error ? reason.message : String(reason)}`); this.deleteBusy.set(false); this.deleteOpen.set(false) }
  }
}
