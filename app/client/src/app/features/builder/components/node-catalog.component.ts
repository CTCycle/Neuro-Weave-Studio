import { Component, HostListener, computed, effect, inject, input, output, signal, viewChild, type ElementRef } from '@angular/core'
import { LucideSearch, LucideX } from '@lucide/angular'
import type { ExecutionBackend, NodeManifest } from '../../../domain/types'
import { implementationKey } from '../../../domain/types'
import { GraphEditorStore } from '../state/graph-editor.store'

@Component({
  selector: 'nw-node-catalog',
  standalone: true,
  imports: [LucideSearch, LucideX],
  template: `
    <aside class="catalog-panel panel" [attr.role]="mobileOpen() ? 'dialog' : 'complementary'" [attr.aria-modal]="mobileOpen() ? 'true' : null" aria-label="Node catalog" (click)="$event.stopPropagation()">
      <div class="panel-heading"><div><span class="eyebrow">BUILDING BLOCKS</span><h2>Node catalog</h2></div><div class="panel-heading-actions"><span class="catalog-count">{{ filteredManifests().length }} node{{ filteredManifests().length === 1 ? '' : 's' }}</span><button #closeButton type="button" class="panel-close" (click)="close.emit()" aria-label="Close node catalog"><svg lucideX [size]="14"></svg></button></div></div>
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
  readonly mobileOpen = input(false)
  readonly close = output<void>()
  readonly query = signal('')
  readonly filteredManifests = computed(() => { const query = this.query().trim().toLowerCase(); return this.store.manifests().filter((manifest) => `${manifest.display_name} ${manifest.id} ${manifest.description} ${manifest.category}`.toLowerCase().includes(query)) })
  readonly categories = computed(() => [...new Set(this.store.manifests().map((manifest) => manifest.category))])
  readonly groups = computed(() => this.categories().map((category) => ({ category, manifests: this.filteredManifests().filter((manifest) => manifest.category === category) })).filter((group) => group.manifests.length > 0))
  categoryOpen(category: string): boolean { return Boolean(this.query().trim()) || Boolean(this.openCategories()[category]) }
  private readonly openCategories = signal<Record<string, boolean>>({})
  private readonly closeButton = viewChild<ElementRef<HTMLButtonElement>>('closeButton')

  constructor() {
    effect(() => {
      if (this.mobileOpen()) queueMicrotask(() => this.closeButton()?.nativeElement.focus())
    })
  }

  @HostListener('keydown', ['$event'])
  keepFocusInside(event: KeyboardEvent): void {
    if (!this.mobileOpen() || event.key !== 'Tab') return
    const focusable = [...this.closeButton()?.nativeElement.closest('aside')?.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href]') ?? []].filter((element) => element.offsetParent !== null)
    if (!focusable.length) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
  }

  toggleCategory(category: string): void { this.openCategories.update((current) => ({ ...current, [category]: !current[category] })) }
  formatCategory(category: string): string { return category.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase()) }
  isSupported(manifest: NodeManifest): boolean { return manifest.implementations[implementationKey(this.executionBackend(), this.kerasEnabled())]?.supported ?? false }
  addNode(manifest: NodeManifest): void { this.store.addNode(manifest, { x: 48 + (this.store.nodes().length % 4) * 176, y: 48 + Math.floor(this.store.nodes().length / 4) * 128 }) }
  startDrag(event: DragEvent, manifest: NodeManifest): void { event.dataTransfer?.setData('application/neuro-weave-node', JSON.stringify({ id: manifest.id, version: manifest.version })); if (event.dataTransfer) event.dataTransfer.effectAllowed = 'copy' }
}
