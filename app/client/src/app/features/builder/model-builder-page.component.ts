import { Component, effect, inject, input, signal } from '@angular/core'
import { Router, RouterLink } from '@angular/router'
import { LucideCheck, LucideCopy, LucideLayoutGrid, LucideRedo2, LucideSave, LucideUndo2 } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { implementationLabel } from '../../domain/types'
import { GraphEditorStore } from './state/graph-editor.store'
import { GraphCanvasComponent } from './components/graph-canvas.component'
import { NodeCatalogComponent } from './components/node-catalog.component'
import { PropertiesInspectorComponent } from './components/properties-inspector.component'

@Component({
  selector: 'nw-model-builder-page',
  standalone: true,
  imports: [RouterLink, LucideCheck, LucideCopy, LucideLayoutGrid, LucideRedo2, LucideSave, LucideUndo2, GraphCanvasComponent, NodeCatalogComponent, PropertiesInspectorComponent],
  template: `
    <div class="builder-page">
      <header class="builder-toolbar panel"><div class="builder-toolbar-heading"><span class="eyebrow">MODEL BUILDER</span><span class="builder-badge"><svg lucideCheck [size]="11"></svg><span>{{ dirty() ? 'UNSAVED' : 'SAVED' }}</span></span></div><div class="builder-identity"><input class="project-title" aria-label="Model name" [value]="name()" (input)="name.set($any($event.target).value)" placeholder="Untitled model" /><span class="revision-label">{{ projectRevision() ? 'r' + projectRevision() : 'draft' }}</span></div><label class="builder-target"><span>Target</span><select [value]="backend()" (change)="backend.set($any($event.target).value)"><option value="tensorflow">TensorFlow</option><option value="pytorch">PyTorch</option><option value="jax">JAX / Flax</option></select></label><div class="builder-history"><button type="button" title="Undo" aria-label="Undo" (click)="store.undo()" [disabled]="!store.canUndo()"><svg lucideUndo2 [size]="14"></svg></button><button type="button" title="Redo" aria-label="Redo" (click)="store.redo()" [disabled]="!store.canRedo()"><svg lucideRedo2 [size]="14"></svg></button></div><div class="builder-actions"><button class="secondary" type="button" (click)="store.copySelected()"><svg lucideCopy [size]="14"></svg>Copy</button><button class="secondary" type="button" (click)="store.autoLayout(store.nodes(), store.edges())"><svg lucideLayoutGrid [size]="14"></svg>Auto layout</button><button class="primary" type="button" (click)="save()" [disabled]="busy()"><svg lucideSave [size]="14"></svg>{{ busy() ? 'Saving…' : 'Save' }}</button></div></header>
      <div class="builder-workspace-grid"><nw-node-catalog></nw-node-catalog><section class="builder-center-workspace"><div class="model-tab-bar panel"><span class="model-tab active">{{ name() || 'Untitled model' }}</span><a class="secondary inline-action" routerLink="/models">Back to library</a></div><nw-graph-canvas></nw-graph-canvas><section class="validation-panel panel" [class.valid]="validation()?.valid"><div class="validation-summary"><span><strong>{{ validation() ? (validation()?.valid ? 'Graph is valid' : 'Graph needs attention') : 'Validation idle' }}</strong><small>{{ validation() ? validation()?.diagnostics?.length + ' diagnostics' : 'Save or validate the graph when ready.' }}</small></span><button class="secondary" type="button" (click)="validate()">Validate</button></div></section></section><nw-properties-inspector></nw-properties-inspector></div>
      @if (notice()) { <button class="notice builder-notice" type="button" (click)="notice.set(null)">{{ notice() }}</button> }
    </div>
  `,
})
export class ModelBuilderPageComponent {
  readonly modelId = input<string | undefined>()
  readonly resources = inject(AppResourcesService)
  readonly store = inject(GraphEditorStore)
  private readonly api = inject(ApiService)
  private readonly router = inject(Router)
  readonly name = signal('Untitled model')
  readonly description = signal('')
  readonly backend = signal<'tensorflow' | 'pytorch' | 'jax'>('pytorch')
  readonly kerasEnabled = signal(true)
  readonly projectRevision = signal<number | null>(null)
  readonly validation = signal<Awaited<ReturnType<ApiService['validate']>> | null>(null)
  readonly busy = signal(false)
  readonly notice = signal<string | null>(null)
  private savedSnapshot = ''

  constructor() {
    effect(() => { this.store.setManifests(this.resources.manifests()); const id = this.modelId(); if (id) void this.loadProject(id) })
  }

  dirty(): boolean { return this.snapshot() !== this.savedSnapshot }
  async loadProject(id: string): Promise<void> { try { const project = await this.api.getProject(id); this.name.set(project.name); this.description.set(project.description); this.backend.set(project.execution_backend); this.kerasEnabled.set(project.keras_enabled); this.projectRevision.set(project.current_revision); this.store.replaceGraph(project); this.savedSnapshot = this.snapshot() } catch (reason) { this.notice.set(`Unable to load model: ${reason instanceof Error ? reason.message : String(reason)}`) } }
  async validate(): Promise<void> { this.busy.set(true); try { const result = await this.api.validate(this.store.toGraph(), this.backend(), this.kerasEnabled()); this.validation.set(result); this.store.applyShapeSummaries(result.shape_summaries); this.notice.set(result.valid ? 'Graph validation passed.' : 'Graph validation found blocking diagnostics.') } catch (reason) { this.notice.set(`Validation failed: ${reason instanceof Error ? reason.message : String(reason)}`) } finally { this.busy.set(false) } }
  async save(): Promise<void> { this.busy.set(true); try { const graph = this.store.toGraph(); const currentId = this.modelId(); const project = currentId ? await this.api.updateProject(currentId, { name: this.name().trim() || 'Untitled model', description: this.description(), execution_backend: this.backend(), keras_enabled: this.kerasEnabled(), graph, expected_revision: this.projectRevision() ?? 0 }) : await this.api.createProject({ name: this.name().trim() || 'Untitled model', description: this.description(), execution_backend: this.backend(), keras_enabled: this.kerasEnabled(), graph }); this.projectRevision.set(project.current_revision); this.name.set(project.name); this.savedSnapshot = this.snapshot(); await this.resources.refreshProjects(); this.notice.set(`Saved revision ${project.current_revision}.`); if (!currentId) await this.router.navigate(['/builder', project.id]) } catch (reason) { this.notice.set(`Save failed: ${reason instanceof Error ? reason.message : String(reason)}`) } finally { this.busy.set(false) } }
  private snapshot(): string { return JSON.stringify({ name: this.name(), description: this.description(), backend: this.backend(), keras: this.kerasEnabled(), graph: this.store.toGraph() }) }
}
