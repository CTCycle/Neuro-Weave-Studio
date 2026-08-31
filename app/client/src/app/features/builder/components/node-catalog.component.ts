import { Component, computed, inject, signal } from '@angular/core'
import { LucideSearch } from '@lucide/angular'
import type { NodeManifest } from '../../../domain/types'
import { GraphEditorStore } from '../state/graph-editor.store'

@Component({
  selector: 'nw-node-catalog',
  standalone: true,
  imports: [LucideSearch],
  template: `
    <aside class="builder-catalog panel" aria-label="Node catalog">
      <div class="builder-catalog-heading"><div><span class="eyebrow">NODE CATALOG</span><h2>Operations</h2></div><span class="catalog-count">{{ filteredManifests().length }}</span></div>
      <label class="search-field catalog-search"><svg lucideSearch [size]="14"></svg><input aria-label="Search node catalog" placeholder="Search operations" [value]="query()" (input)="query.set($any($event.target).value)" /></label>
      <div class="catalog-groups">
        @for (category of categories(); track category) {
          <section class="catalog-group"><h3>{{ category }}</h3>@for (manifest of manifestsFor(category); track manifest.id + manifest.version) { <button class="catalog-node" type="button" draggable="true" (dragstart)="startDrag($event, manifest)" (click)="addNode(manifest)"><span class="catalog-node-icon" [style.--node-accent]="manifest.ui.accent">{{ manifest.ui.icon || '·' }}</span><span><strong>{{ manifest.display_name }}</strong><small>{{ manifest.id }}</small></span></button> }</section>
        }
        @if (!filteredManifests().length) { <p class="muted catalog-empty-copy">No operations match this search.</p> }
      </div>
    </aside>
  `,
})
export class NodeCatalogComponent {
  private readonly store = inject(GraphEditorStore)
  readonly query = signal('')
  readonly filteredManifests = computed(() => { const query = this.query().trim().toLowerCase(); return this.store.manifests().filter((manifest) => `${manifest.display_name} ${manifest.id} ${manifest.category}`.toLowerCase().includes(query)) })
  readonly categories = computed(() => [...new Set(this.filteredManifests().map((manifest) => manifest.category))])
  manifestsFor(category: string): NodeManifest[] { return this.filteredManifests().filter((manifest) => manifest.category === category) }
  addNode(manifest: NodeManifest): void { this.store.addNode(manifest, { x: 48 + (this.store.nodes().length % 4) * 176, y: 48 + Math.floor(this.store.nodes().length / 4) * 128 }) }
  startDrag(event: DragEvent, manifest: NodeManifest): void { event.dataTransfer?.setData('application/neuro-weave-node', JSON.stringify({ id: manifest.id, version: manifest.version })); if (event.dataTransfer) event.dataTransfer.effectAllowed = 'copy' }
}
