import { Component, HostListener, computed, effect, inject, input, output, signal, viewChild, type ElementRef } from '@angular/core'
import { LucideSearch, LucideX } from '@lucide/angular'
import { CustomNodeApiService } from '../../../core/api/custom-node-api.service'
import type { CustomNodeDefinition } from '../../../domain/custom-node.types'
import type { ExecutionBackend, NodeManifest, PortKind } from '../../../domain/types'
import { resolveImplementationCompatibility } from '../../../domain/types'
import { CustomNodeMaterializerService } from '../services/custom-node-materializer.service'
import { GraphEditorStore } from '../state/graph-editor.store'

type CompatibilityFilter = 'all' | 'current' | 'keras' | ExecutionBackend

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

      <div class="filter-toolbar">
        <button type="button" class="filter-trigger" [class.active]="advancedOpen() || activeFilterCount()" [attr.aria-expanded]="advancedOpen()" aria-controls="catalog-advanced-filters" (click)="advancedOpen.update((value) => !value)">Filters @if (activeFilterCount()) { <span>{{ activeFilterCount() }}</span> }</button>
        @if (activeFilterCount()) { <button type="button" class="clear-filters" (click)="clearFilters()">Clear all</button> }
      </div>

      @if (activeFilterLabels().length) {
        <div class="active-filters" aria-label="Active node filters">
          @for (filter of activeFilterLabels(); track filter.key) { <button type="button" (click)="clearFilter(filter.key)" [attr.aria-label]="'Clear ' + filter.label + ' filter'">{{ filter.label }} <span aria-hidden="true">×</span></button> }
        </div>
      }

      @if (advancedOpen()) {
        <section id="catalog-advanced-filters" class="advanced-filters" aria-label="Advanced built-in node filters">
          <label><span>Category</span><select [value]="categoryFilter()" (change)="categoryFilter.set($any($event.target).value)"><option value="all">All categories</option>@for (category of categories(); track category) { <option [value]="category">{{ formatCategory(category) }}</option> }</select></label>
          <label><span>Compatibility</span><select [value]="compatibilityFilter()" (change)="compatibilityFilter.set($any($event.target).value)"><option value="all">Any implementation</option><option value="current">Current profile</option><option value="keras">Keras</option><option value="tensorflow">TensorFlow native</option><option value="pytorch">PyTorch native</option><option value="jax">JAX native</option></select></label>
          <label><span>Port type</span><select [value]="portKindFilter()" (change)="portKindFilter.set($any($event.target).value)"><option value="all">Any I/O type</option>@for (kind of portKinds(); track kind) { <option [value]="kind">{{ formatCategory(kind) }}</option> }</select></label>
          <label><span>Family</span><select [value]="familyFilter()" (change)="familyFilter.set($any($event.target).value)"><option value="all">Any family</option>@for (family of families(); track family) { <option [value]="family">{{ formatCategory(family) }}</option> }</select></label>
          <label><span>Operation</span><select [value]="operationFilter()" (change)="operationFilter.set($any($event.target).value)"><option value="all">Any operation type</option>@for (operation of operationTypes(); track operation) { <option [value]="operation">{{ formatCategory(operation) }}</option> }</select></label>
        </section>
      }

      <div class="catalog-list">
        @if (customTemplates().length || referenceDefinitions().length || customError()) {
          <section class="catalog-category custom-category">
            <div class="custom-heading"><span><strong>Custom templates</strong><small>Expand into canonical graph nodes</small></span></div>
            @for (definition of customTemplates(); track definition.id) {
              <button class="catalog-item custom-item" type="button" [disabled]="!isCustomSupported(definition)" [title]="customTitle(definition)" (click)="addCustom(definition)"><span class="node-accent" aria-hidden="true"></span><span class="catalog-item-copy"><strong>{{ definition.display_name }}</strong><small>{{ definition.kind === 'composite' ? 'Sequential block' : 'Mathematical operation' }} · v{{ definition.version }}</small></span>@if (!isCustomSupported(definition)) { <span class="compatibility-badge">Limited</span> } @else { <span class="custom-badge">Custom</span> }</button>
            }
            @for (definition of referenceDefinitions(); track definition.id) {
              <button class="catalog-item reference-item" type="button" disabled title="Verified repository reference only; no trusted execution adapter is installed"><span class="node-accent" aria-hidden="true"></span><span class="catalog-item-copy"><strong>{{ definition.display_name }}</strong><small>Hugging Face reference · not executable</small></span><span class="reference-badge">Reference</span></button>
            }
            @if (customError()) { <p class="custom-error">Custom templates unavailable: {{ customError() }}</p> }
          </section>
        }

        @if (!groups().length && !customTemplates().length && !referenceDefinitions().length) { <div class="catalog-empty-state"><strong>No matching nodes</strong><span>Clear a filter or broaden the search.</span></div> }
        @for (group of groups(); track group.category) {
          <section class="catalog-category">
            <button type="button" class="catalog-category-toggle" [attr.aria-expanded]="categoryOpen(group.category)" [attr.aria-controls]="'catalog-' + group.category" (click)="toggleCategory(group.category)"><span><strong>{{ formatCategory(group.category) }}</strong><small>{{ group.manifests.length }} node{{ group.manifests.length === 1 ? '' : 's' }}</small></span><span aria-hidden="true">{{ categoryOpen(group.category) ? '−' : '+' }}</span></button>
            @if (categoryOpen(group.category)) {
              <div class="catalog-category-items" [id]="'catalog-' + group.category">
                @for (manifest of group.manifests; track manifest.id + manifest.version) {
                  <button class="catalog-item" [class.export-limited]="!isSupported(manifest)" type="button" [draggable]="isSupported(manifest)" [disabled]="!isSupported(manifest)" [title]="isSupported(manifest) ? 'Drag to the canvas or activate to add' : compatibilityMessage(manifest)" (dragstart)="startDrag($event, manifest)" (click)="addNode(manifest)"><span class="node-accent" aria-hidden="true"></span><span class="catalog-item-copy"><strong>{{ manifest.display_name }}</strong><small>{{ compactMetadata(manifest) }}</small></span>@if (!isSupported(manifest)) { <span class="compatibility-badge">Limited</span> }</button>
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
    .filter-toolbar{display:flex;align-items:center;gap:8px;padding:8px 0 2px}.filter-trigger,.clear-filters{border:0;background:transparent;color:var(--muted);min-height:28px;padding:3px 0;font:600 11px/1 'IBM Plex Mono',Consolas,monospace;cursor:pointer}.filter-trigger{display:inline-flex;align-items:center;gap:7px}.filter-trigger span{min-width:18px;height:18px;display:inline-grid;place-items:center;border:1px solid color-mix(in srgb,var(--cyan) 45%,var(--border));border-radius:9px;color:var(--cyan)}.filter-trigger.active{color:var(--text)}.clear-filters{margin-left:auto;color:var(--accent-bright)}.active-filters{display:flex;flex-wrap:wrap;gap:5px;padding:4px 0 7px}.active-filters button{border:1px solid color-mix(in srgb,var(--cyan) 26%,var(--border));background:color-mix(in srgb,var(--surface-2) 84%,transparent);color:var(--muted);border-radius:4px;padding:4px 6px;font:500 10px/1.2 'IBM Plex Mono',Consolas,monospace;cursor:pointer}.advanced-filters{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:9px 0 10px;border-top:1px solid var(--border);border-bottom:1px solid var(--border)}.advanced-filters label{display:grid;gap:4px;min-width:0}.advanced-filters label:last-child:nth-child(odd){grid-column:1/-1}.advanced-filters span{color:var(--muted);font:600 9px/1 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase;letter-spacing:.08em}.advanced-filters select{width:100%;min-width:0;min-height:31px;padding:4px 24px 4px 7px;font-size:11px}.catalog-list{gap:0}.catalog-category{border-bottom:1px solid color-mix(in srgb,var(--border) 72%,transparent)}.catalog-category-items{gap:0;padding:2px 0 7px}.catalog-item{position:relative;display:grid;grid-template-columns:3px minmax(0,1fr) auto;align-items:center;gap:9px;width:100%;min-height:42px;border:0;border-radius:3px;background:transparent;color:var(--text);padding:5px 6px 5px 3px;text-align:left;box-shadow:none;transform:none}.catalog-item:not(:disabled){cursor:pointer}.catalog-item:not(:disabled):hover,.catalog-item:not(:disabled):focus-visible{background:color-mix(in srgb,var(--surface-2) 82%,transparent);outline:none}.catalog-item:focus-visible{box-shadow:inset 0 0 0 1px var(--cyan)}.catalog-item:disabled{opacity:.58}.node-accent{align-self:stretch;width:2px;border-radius:2px;background:var(--cyan);opacity:.55}.catalog-item.export-limited .node-accent,.reference-item .node-accent{background:var(--accent-bright);opacity:.7}.catalog-item-copy{min-width:0;display:grid;gap:3px}.catalog-item-copy strong,.catalog-item-copy small{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.catalog-item-copy strong{font-size:12px}.catalog-item-copy small{color:var(--muted);font:500 9px/1.3 'IBM Plex Mono',Consolas,monospace}.compatibility-badge,.custom-badge,.reference-badge{font:600 8px/1 'IBM Plex Mono',Consolas,monospace;padding:3px 5px;border:1px solid var(--border);border-radius:999px}.custom-badge{color:var(--cyan)}.reference-badge{color:var(--muted)}.custom-category{padding-bottom:6px}.custom-heading{display:flex;padding:10px 4px 6px}.custom-heading>span{display:grid;gap:3px}.custom-heading strong{font-size:11px}.custom-heading small{color:var(--muted);font-size:9px}.custom-error{margin:5px 4px 8px;color:var(--error);font-size:9px;line-height:1.4}.catalog-notice{margin:8px 0 0;width:100%;border:1px solid color-mix(in srgb,var(--error) 48%,var(--border));background:rgba(239,106,114,.08);color:var(--error);padding:7px 8px;border-radius:4px;text-align:left;font-size:9px;cursor:pointer}@media(max-width:700px){.advanced-filters{grid-template-columns:1fr}}
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
  readonly advancedOpen = signal(false)
  readonly categoryFilter = signal('all')
  readonly compatibilityFilter = signal<CompatibilityFilter>('all')
  readonly portKindFilter = signal<'all' | PortKind>('all')
  readonly familyFilter = signal('all')
  readonly operationFilter = signal('all')
  readonly customs = signal<CustomNodeDefinition[]>([])
  readonly customError = signal<string | null>(null)
  readonly materializeError = signal<string | null>(null)
  private readonly openCategories = signal<Record<string, boolean>>({})
  private readonly closeButton = viewChild<ElementRef<HTMLButtonElement>>('closeButton')

  readonly categories = computed(() => [...new Set(this.store.manifests().map((manifest) => manifest.category))].sort())
  readonly portKinds = computed(() => [...new Set(this.store.manifests().flatMap((manifest) => [...manifest.inputs, ...manifest.outputs].map((port) => port.kind)))].sort())
  readonly families = computed(() => [...new Set(this.store.manifests().map((manifest) => manifest.operation?.family ?? manifest.semantic_role ?? manifest.category))].sort())
  readonly operationTypes = computed(() => [...new Set(this.store.manifests().map((manifest) => manifest.operation?.shape_behavior).filter((value): value is NonNullable<typeof value> => Boolean(value)))].sort())
  readonly filteredManifests = computed(() => {
    const query = this.query().trim().toLowerCase()
    return this.store.manifests().filter((manifest) => {
      const searchable = `${manifest.display_name} ${manifest.id} ${manifest.description} ${manifest.category} ${manifest.operation?.family ?? ''} ${manifest.operation?.shape_behavior ?? ''}`.toLowerCase()
      if (query && !searchable.includes(query)) return false
      if (this.categoryFilter() !== 'all' && manifest.category !== this.categoryFilter()) return false
      if (this.portKindFilter() !== 'all' && ![...manifest.inputs, ...manifest.outputs].some((port) => port.kind === this.portKindFilter())) return false
      if (this.familyFilter() !== 'all' && (manifest.operation?.family ?? manifest.semantic_role ?? manifest.category) !== this.familyFilter()) return false
      if (this.operationFilter() !== 'all' && manifest.operation?.shape_behavior !== this.operationFilter()) return false
      return this.matchesCompatibility(manifest, this.compatibilityFilter())
    })
  })
  readonly groups = computed(() => this.categories().map((category) => ({ category, manifests: this.filteredManifests().filter((manifest) => manifest.category === category) })).filter((group) => group.manifests.length > 0))
  readonly activeFilterCount = computed(() => [this.categoryFilter(), this.compatibilityFilter(), this.portKindFilter(), this.familyFilter(), this.operationFilter()].filter((value) => value !== 'all').length)
  readonly activeFilterLabels = computed(() => [
    this.categoryFilter() !== 'all' ? { key: 'category', label: this.formatCategory(this.categoryFilter()) } : null,
    this.compatibilityFilter() !== 'all' ? { key: 'compatibility', label: this.compatibilityLabel(this.compatibilityFilter()) } : null,
    this.portKindFilter() !== 'all' ? { key: 'port', label: this.formatCategory(this.portKindFilter()) } : null,
    this.familyFilter() !== 'all' ? { key: 'family', label: this.formatCategory(this.familyFilter()) } : null,
    this.operationFilter() !== 'all' ? { key: 'operation', label: this.formatCategory(this.operationFilter()) } : null,
  ].filter((value): value is { key: string; label: string } => Boolean(value)))
  readonly filteredCustoms = computed(() => {
    const query = this.query().trim().toLowerCase()
    const category = this.categoryFilter()
    return this.customs().filter((definition) => (!query || `${definition.display_name} ${definition.description} ${definition.category} ${definition.kind}`.toLowerCase().includes(query)) && (category === 'all' || definition.category === category))
  })
  readonly customTemplates = computed(() => this.filteredCustoms().filter((definition) => definition.executable_template))
  readonly referenceDefinitions = computed(() => this.filteredCustoms().filter((definition) => !definition.executable_template))

  constructor() {
    effect(() => this.store.setExecutionProfile(this.executionBackend(), this.kerasEnabled()))
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
    const focusable = [...this.closeButton()?.nativeElement.closest('aside')?.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href]') ?? []].filter((element) => element.offsetParent !== null)
    if (!focusable.length) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
  }

  categoryOpen(category: string): boolean { return Boolean(this.query().trim()) || Boolean(this.activeFilterCount()) || Boolean(this.openCategories()[category]) }
  toggleCategory(category: string): void { this.openCategories.update((current) => ({ ...current, [category]: !current[category] })) }
  formatCategory(value: string): string { return value.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase()) }
  isSupported(manifest: NodeManifest): boolean { return this.store.nodeCompatibility(manifest).available }
  compatibilityMessage(manifest: NodeManifest): string { return this.store.nodeCompatibility(manifest).reason ?? 'Unavailable for the current execution profile.' }
  compactMetadata(manifest: NodeManifest): string { return `${manifest.id} · ${manifest.operation?.family ?? manifest.semantic_role ?? manifest.category}` }
  addNode(manifest: NodeManifest): void {
    if (!this.isSupported(manifest)) {
      this.materializeError.set(this.compatibilityMessage(manifest))
      return
    }
    if (this.store.addNode(manifest, this.nextPosition()) === null) this.materializeError.set(this.compatibilityMessage(manifest))
  }
  startDrag(event: DragEvent, manifest: NodeManifest): void {
    if (!this.isSupported(manifest)) {
      event.preventDefault()
      event.stopPropagation()
      return
    }
    event.dataTransfer?.setData('application/neuro-weave-node', JSON.stringify({ id: manifest.id, version: manifest.version })); if (event.dataTransfer) event.dataTransfer.effectAllowed = 'copy'
  }
  addCustom(definition: CustomNodeDefinition): void {
    this.materializeError.set(null)
    if (!this.isCustomSupported(definition)) {
      this.materializeError.set(this.customTitle(definition))
      return
    }
    try { this.materializer.materialize(definition, this.nextPosition()) }
    catch (reason) { this.materializeError.set(reason instanceof Error ? reason.message : String(reason)) }
  }
  isCustomSupported(definition: CustomNodeDefinition): boolean { return this.materializer.compatibility(definition).available }
  customTitle(definition: CustomNodeDefinition): string {
    const result = this.materializer.compatibility(definition)
    return result.available ? (definition.kind === 'composite' ? 'Insert sequential built-in block' : 'Insert safe mathematical operation') : result.reason
  }
  clearFilters(): void { this.categoryFilter.set('all'); this.compatibilityFilter.set('all'); this.portKindFilter.set('all'); this.familyFilter.set('all'); this.operationFilter.set('all') }
  clearFilter(key: string): void {
    if (key === 'category') this.categoryFilter.set('all')
    else if (key === 'compatibility') this.compatibilityFilter.set('all')
    else if (key === 'port') this.portKindFilter.set('all')
    else if (key === 'family') this.familyFilter.set('all')
    else if (key === 'operation') this.operationFilter.set('all')
  }
  private matchesCompatibility(manifest: NodeManifest, filter: CompatibilityFilter): boolean {
    if (filter === 'all') return true
    if (filter === 'current') return this.isSupported(manifest)
    if (filter === 'keras') {
      return (['tensorflow', 'pytorch', 'jax'] as ExecutionBackend[]).some((backend) => resolveImplementationCompatibility(manifest, backend, true).available)
    }
    return resolveImplementationCompatibility(manifest, filter, false).available
  }
  private compatibilityLabel(filter: CompatibilityFilter): string {
    if (filter === 'current') return 'Current profile'
    if (filter === 'keras') return 'Keras'
    return `${this.formatCategory(filter)} native`
  }
  private nextPosition(): { x: number; y: number } {
    const index = this.store.nodes().length
    return { x: 48 + (index % 3) * 300, y: 48 + Math.floor(index / 3) * 160 }
  }
}
