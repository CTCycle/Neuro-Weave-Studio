import { Component, computed, inject, input, signal } from '@angular/core'
import { LucideSearch } from '@lucide/angular'
import type { ExecutionBackend, NodeManifest } from '../../../domain/types'
import { implementationKey } from '../../../domain/types'
import { GraphEditorStore } from '../state/graph-editor.store'

@Component({
  selector: 'nw-node-catalog',
  standalone: true,
  imports: [LucideSearch],
  template: `
    <aside class="catalog-panel panel" aria-label="Node catalog">
      <div class="panel-heading"><div><span class="eyebrow">BUILDING BLOCKS</span><h2>Node catalog</h2></div><span class="catalog-count">{{ filteredManifests().length }} node{{ filteredManifests().length === 1 ? '' : 's' }}</span></div>
      <label class="search-field"><svg lucideSearch [size]="16"></svg><input aria-label="Search nodes" placeholder="Search layers and operations" [value]="query()" (input)="query.set($any($event.target).value)" /></label>
      <div class="catalog-list">
        @if (!groups().length) { <div class="catalog-empty-state"><strong>No matching nodes</strong><span>Try a different layer, operation, or category.</span></div> }
        @for (group of groups(); track group.category) {
          <section class="catalog-category"><button type="button" class="catalog-category-toggle" [attr.aria-expanded]="categoryOpen(group.category)" [attr.aria-controls]="'catalog-' + group.category" (click)="toggleCategory(group.category)"><span><strong>{{ formatCategory(group.category) }}</strong><small>{{ group.manifests.length }} node{{ group.manifests.length === 1 ? '' : 's' }}</small></span><span aria-hidden="true">{{ categoryOpen(group.category) ? '−' : '+' }}</span></button>@if (categoryOpen(group.category)) { <div class="catalog-category-items" [id]="'catalog-' + group.category">@for (manifest of group.manifests; track manifest.id + manifest.version) { <button class="catalog-item" [class.export-limited]="!isSupported(manifest)" type="button" draggable="true" [title]="isSupported(manifest) ? 'Drag to the canvas or activate to add' : 'Available for editing; export is limited for this execution profile'" (dragstart)="startDrag($event, manifest)" (click)="addNode(manifest)"><div><strong>{{ manifest.display_name }}</strong><small>{{ manifest.description || manifest.id }}</small></div>@if (!isSupported(manifest)) { <span class="compatibility-badge">Export limited</span> }</button> }</div> }</section>
        }
      </div>
    </aside>
  `,
})
export class NodeCatalogComponent {
  private readonly store = inject(GraphEditorStore)
  readonly executionBackend = input<ExecutionBackend>('pytorch')
  readonly kerasEnabled = input(true)
  readonly query = signal('')
  readonly filteredManifests = computed(() => { const query = this.query().trim().toLowerCase(); return this.store.manifests().filter((manifest) => `${manifest.display_name} ${manifest.id} ${manifest.description} ${manifest.category}`.toLowerCase().includes(query)) })
  readonly categories = computed(() => [...new Set(this.store.manifests().map((manifest) => manifest.category))])
  readonly groups = computed(() => this.categories().map((category) => ({ category, manifests: this.filteredManifests().filter((manifest) => manifest.category === category) })).filter((group) => group.manifests.length > 0))
  categoryOpen(category: string): boolean { return Boolean(this.query().trim()) || Boolean(this.openCategories()[category]) }
  private readonly openCategories = signal<Record<string, boolean>>({})
  toggleCategory(category: string): void { this.openCategories.update((current) => ({ ...current, [category]: !current[category] })) }
  formatCategory(category: string): string { return category.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase()) }
  isSupported(manifest: NodeManifest): boolean { return manifest.implementations[implementationKey(this.executionBackend(), this.kerasEnabled())]?.supported ?? false }
  addNode(manifest: NodeManifest): void { this.store.addNode(manifest, { x: 48 + (this.store.nodes().length % 4) * 176, y: 48 + Math.floor(this.store.nodes().length / 4) * 128 }) }
  startDrag(event: DragEvent, manifest: NodeManifest): void { event.dataTransfer?.setData('application/neuro-weave-node', JSON.stringify({ id: manifest.id, version: manifest.version })); if (event.dataTransfer) event.dataTransfer.effectAllowed = 'copy' }
}
