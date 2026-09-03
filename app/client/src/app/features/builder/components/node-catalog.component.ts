import { Component, HostListener, computed, effect, inject, input, output, signal, viewChild, type ElementRef } from '@angular/core'
import { LucideSearch, LucideX } from '@lucide/angular'
import { CustomNodeApiService } from '../../../core/api/custom-node-api.service'
import type { CustomNodeDefinition } from '../../../domain/custom-node.types'
import type { ExecutionBackend, NodeManifest } from '../../../domain/types'
import { implementationAvailable } from '../../../domain/types'
import { CustomNodeMaterializerService } from '../services/custom-node-materializer.service'
import { GraphEditorStore } from '../state/graph-editor.store'

@Component({
  selector: 'nw-node-catalog',
  standalone: true,
  imports: [LucideSearch, LucideX],
  template: `
    <aside class="catalog-panel panel" [attr.role]="mobileOpen() ? 'dialog' : 'complementary'" [attr.aria-modal]="mobileOpen() ? 'true' : null" aria-label="Node catalog" (click)="$event.stopPropagation()">
      <div class="panel-heading">
        <div><span class="eyebrow">BUILDING BLOCKS</span><h2>Node catalog</h2></div>
        <div class="panel-heading-actions"><span class="catalog-count">{{ filteredManifests().length + customTemplates().length }} available</span><button #closeButton type="button" class="panel-close" (click)="close.emit()" aria-label="Close node catalog"><svg lucideX [size]="14"></svg></button></div>
      </div>

      <label class="search-field"><svg lucideSearch [size]="16"></svg><input aria-label="Search nodes" placeholder="Search layers, operations, templates" [value]="query()" (input)="query.set($any($event.target).value)" /></label>

      <div class="catalog-list">
        @if (customTemplates().length || referenceDefinitions().length || customError()) {
          <section class="catalog-category custom-category">
            <div class="custom-heading"><span><strong>Custom templates</strong><small>Expand into canonical graph nodes</small></span></div>
            @for (definition of customTemplates(); track definition.id) {
              <button class="catalog-item custom-item" type="button" [title]="definition.kind === 'composite' ? 'Insert sequential built-in block' : 'Insert safe mathematical operation'" (click)="addCustom(definition)">
                <span class="node-accent"></span><span class="catalog-item-copy"><strong>{{ definition.display_name }}</strong><small>{{ definition.kind === 'composite' ? 'Sequential block' : 'Mathematical operation' }} · v{{ definition.version }}</small></span><span class="custom-badge">Custom</span>
              </button>
            }
            @for (definition of referenceDefinitions(); track definition.id) {
              <button class="catalog-item reference-item" type="button" disabled title="Verified repository reference only; no trusted execution adapter is installed">
                <span class="node-accent"></span><span class="catalog-item-copy"><strong>{{ definition.display_name }}</strong><small>Hugging Face reference · not executable</small></span><span class="reference-badge">Reference</span>
              </button>
            }
            @if (customError()) { <p class="custom-error">Custom templates unavailable: {{ customError() }}</p> }
          </section>
        }

        @if (!groups().length && !customTemplates().length && !referenceDefinitions().length) { <div class="catalog-empty-state"><strong>No matching nodes</strong><span>Broaden the search to find another definition.</span></div> }
        @for (group of groups(); track group.category) {
          <section class="catalog-category">
            <button type="button" class="catalog-category-toggle" [attr.aria-expanded]="categoryOpen(group.category)" [attr.aria-controls]="'catalog-' + group.category" (click)="toggleCategory(group.category)"><span><strong>{{ formatCategory(group.category) }}</strong><small>{{ group.manifests.length }} node{{ group.manifests.length === 1 ? '' : 's' }}</small></span><span aria-hidden="true">{{ categoryOpen(group.category) ? '−' : '+' }}</span></button>
            @if (categoryOpen(group.category)) {
              <div class="catalog-category-items" [id]="'catalog-' + group.category">
                @for (manifest of group.manifests; track manifest.id + manifest.version) {
                  <button class="catalog-item" [class.export-limited]="!isSupported(manifest)" type="button" draggable="true" [title]="isSupported(manifest) ? 'Drag to the canvas or activate to add' : 'Available for editing; export is limited for this execution profile'" (dragstart)="startDrag($event, manifest)" (click)="addNode(manifest)">
                    <span class="node-accent"></span><span class="catalog-item-copy"><strong>{{ manifest.display_name }}</strong><small>{{ compactMetadata(manifest) }}</small></span>@if (!isSupported(manifest)) { <span class="compatibility-badge">Limited</span> }
                  </button>
                }
              </div>
            }
          </section>
        }
      </div>
      @if (materializeError()) { <button class="catalog-notice" type="button" (click)="materializeError.set(null)">{{ materializeError() }}</button> }
    </aside>
  `,
  styles: [`
    .catalog-list{gap:0}.catalog-category{border-bottom:1px solid color-mix(in srgb,var(--border) 72%,transparent)}.catalog-category-items{gap:0;padding:2px 0 7px}.catalog-item{position:relative;display:grid;grid-template-columns:3px minmax(0,1fr) auto;align-items:center;gap:9px;width:100%;min-height:42px;border:0;border-radius:3px;background:transparent;color:var(--text);padding:5px 6px 5px 3px;text-align:left;box-shadow:none;transform:none}.catalog-item:not(:disabled){cursor:pointer}.catalog-item:not(:disabled):hover,.catalog-item:not(:disabled):focus-visible{background:color-mix(in srgb,var(--surface-2) 82%,transparent);outline:none}.catalog-item:focus-visible{box-shadow:inset 0 0 0 1px var(--cyan)}.catalog-item:disabled{opacity:.58}.node-accent{align-self:stretch;width:2px;border-radius:2px;background:var(--cyan);opacity:.55}.catalog-item.export-limited .node-accent,.reference-item .node-accent{background:var(--accent-bright);opacity:.7}.catalog-item-copy{min-width:0;display:grid;gap:3px}.catalog-item-copy strong,.catalog-item-copy small{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.catalog-item-copy strong{font-size:12px}.catalog-item-copy small{color:var(--muted);font:500 9px/1.3 'IBM Plex Mono',Consolas,monospace}.compatibility-badge,.custom-badge,.reference-badge{font:600 8px/1 'IBM Plex Mono',Consolas,monospace;padding:3px 5px;border:1px solid var(--border);border-radius:999px}.custom-badge{color:var(--cyan)}.reference-badge{color:var(--muted)}.custom-category{padding-bottom:6px}.custom-heading{display:flex;padding:10px 4px 6px}.custom-heading>span{display:grid;gap:3px}.custom-heading strong{font-size:11px}.custom-heading small{color:var(--muted);font-size:9px}.custom-error{margin:5px 4px 8px;color:var(--error);font-size:9px;line-height:1.4}.catalog-notice{margin:8px 0 0;width:100%;border:1px solid color-mix(in srgb,var(--error) 48%,var(--border));background:rgba(239,106,114,.08);color:var(--error);padding:7px 8px;border-radius:4px;text-align:left;font-size:9px;cursor:pointer}
  `],
})
export class NodeCatalogComponent {
  private readonly store = inject(GraphEditorStore)
  private readonly customApi = inject(CustomNodeApiService)
  private readonly materializer = inject(CustomNodeMaterializerService)
  readonly executionBackend = input<ExecutionBackend>('pytorch')
  readonly kerasEnabled = input(true)
  readonly mobileOpen = input(false)
  readonly close = output<void>()
  readonly query = signal('')
  readonly customs = signal<CustomNodeDefinition[]>([])
  readonly customError = signal<string | null>(null)
  readonly materializeError = signal<string | null>(null)
  private readonly openCategories = signal<Record<string, boolean>>({})
  private readonly closeButton = viewChild<ElementRef<HTMLButtonElement>>('closeButton')

  readonly filteredManifests = computed(() => {
    const query = this.query().trim().toLowerCase()
    return this.store.manifests().filter((manifest) => !query || `${manifest.display_name} ${manifest.id} ${manifest.description} ${manifest.category} ${manifest.operation?.family ?? ''}`.toLowerCase().includes(query))
  })
  readonly categories = computed(() => [...new Set(this.filteredManifests().map((manifest) => manifest.category))].sort())
  readonly groups = computed(() => this.categories().map((category) => ({ category, manifests: this.filteredManifests().filter((manifest) => manifest.category === category) })))
  readonly filteredCustoms = computed(() => {
    const query = this.query().trim().toLowerCase()
    return this.customs().filter((definition) => !query || `${definition.display_name} ${definition.description} ${definition.category} ${definition.kind}`.toLowerCase().includes(query))
  })
  readonly customTemplates = computed(() => this.filteredCustoms().filter((definition) => definition.executable_template))
  readonly referenceDefinitions = computed(() => this.filteredCustoms().filter((definition) => !definition.executable_template))

  constructor() {
    effect(() => { if (this.mobileOpen()) queueMicrotask(() => this.closeButton()?.nativeElement.focus()) })
    void this.loadCustoms()
  }

  private async loadCustoms(): Promise<void> {
    try { this.customs.set(await this.customApi.list()); this.customError.set(null) }
    catch (reason) { this.customError.set(reason instanceof Error ? reason.message : String(reason)) }
  }

  @HostListener('keydown', ['$event'])
  keepFocusInside(event: KeyboardEvent): void {
    if (!this.mobileOpen() || event.key !== 'Tab') return
    const aside = this.closeButton()?.nativeElement.closest('aside')
    const focusable = [...aside?.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), a[href]') ?? []].filter((element) => element.offsetParent !== null)
    if (!focusable.length) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
  }

  categoryOpen(category: string): boolean { return Boolean(this.query().trim()) || Boolean(this.openCategories()[category]) }
  toggleCategory(category: string): void { this.openCategories.update((current) => ({ ...current, [category]: !current[category] })) }
  formatCategory(value: string): string { return value.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase()) }
  isSupported(manifest: NodeManifest): boolean { return implementationAvailable(manifest, this.executionBackend(), this.kerasEnabled()) }
  compactMetadata(manifest: NodeManifest): string { return `${manifest.id} · ${manifest.operation?.family ?? manifest.semantic_role ?? manifest.category}` }
  addNode(manifest: NodeManifest): void { this.store.addNode(manifest, this.nextPosition()) }
  startDrag(event: DragEvent, manifest: NodeManifest): void { event.dataTransfer?.setData('application/neuro-weave-node', JSON.stringify({ id: manifest.id, version: manifest.version })); if (event.dataTransfer) event.dataTransfer.effectAllowed = 'copy' }
  addCustom(definition: CustomNodeDefinition): void {
    this.materializeError.set(null)
    try { this.materializer.materialize(definition, this.nextPosition()) }
    catch (reason) { this.materializeError.set(reason instanceof Error ? reason.message : String(reason)) }
  }
  private nextPosition(): { x: number; y: number } {
    const index = this.store.nodes().length
    return { x: 48 + (index % 3) * 300, y: 48 + Math.floor(index / 3) * 160 }
  }
}
