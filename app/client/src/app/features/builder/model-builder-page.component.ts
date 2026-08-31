import { Component, HostListener, computed, effect, inject, input, signal, untracked, viewChild, type ElementRef } from '@angular/core'
import { Router } from '@angular/router'
import { LucideFileText, LucideHammer, LucideLayoutGrid, LucidePlay, LucidePlus, LucideRedo2, LucideSave, LucideUndo2, LucideX } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import type { ExecutionBackend, ProjectDocument, ValidationResult } from '../../domain/types'
import { executionBackendShortLabels } from '../../domain/types'
import type { GraphEditorSession } from '../../domain/graph-mapper'
import { GraphEditorStore } from './state/graph-editor.store'
import { BuilderTabsStore, type BuilderTab, type NewBuilderTab } from './state/builder-tabs.store'
import { GraphCanvasComponent } from './components/graph-canvas.component'
import { NodeCatalogComponent } from './components/node-catalog.component'
import { PropertiesInspectorComponent } from './components/properties-inspector.component'
import { ResizableValidationRegionComponent, DEFAULT_VALIDATION_HEIGHT } from './components/resizable-validation-region.component'
import { ValidationPanelComponent } from './components/validation-panel.component'

function draftSnapshot(name: string, description: string, executionBackend: ExecutionBackend, kerasEnabled: boolean, graph: unknown): string {
  return JSON.stringify({ name, description, executionBackend, kerasEnabled, graph })
}

function preferredBackend(statuses: { id: ExecutionBackend; available: boolean }[]): ExecutionBackend {
  if (statuses.some((item) => item.id === 'pytorch' && item.available)) return 'pytorch'
  return statuses.find((item) => item.available)?.id ?? 'pytorch'
}

function projectTabInput(project: ProjectDocument, graph: GraphEditorSession): NewBuilderTab {
  return {
    projectId: project.id,
    name: project.name,
    description: project.description,
    executionBackend: project.execution_backend,
    kerasEnabled: project.keras_enabled,
    revision: project.current_revision,
    graph,
    validation: null,
    validationExpanded: false,
    validationHeight: DEFAULT_VALIDATION_HEIGHT,
    savedBaseline: draftSnapshot(project.name, project.description, project.execution_backend, project.keras_enabled, project.graph),
  }
}

function untitledTabInput(executionBackend: ExecutionBackend, graph: GraphEditorSession, modelGraph: unknown): NewBuilderTab {
  return {
    projectId: null,
    name: 'Untitled model',
    description: '',
    executionBackend,
    kerasEnabled: true,
    revision: 1,
    graph,
    validation: null,
    validationExpanded: false,
    validationHeight: DEFAULT_VALIDATION_HEIGHT,
    savedBaseline: draftSnapshot('Untitled model', '', executionBackend, true, modelGraph),
  }
}

type DraftMetadataPatch = Partial<Pick<BuilderTab, 'name' | 'description' | 'executionBackend' | 'kerasEnabled'>>

@Component({
  selector: 'nw-model-builder-page',
  standalone: true,
  imports: [LucideFileText, LucideHammer, LucideLayoutGrid, LucidePlay, LucidePlus, LucideRedo2, LucideSave, LucideUndo2, LucideX, GraphCanvasComponent, NodeCatalogComponent, PropertiesInspectorComponent, ResizableValidationRegionComponent, ValidationPanelComponent],
  template: `
    @if (!initialized() || !activeTab()) {
      <section class="page-state panel"><span class="eyebrow">MODEL BUILDER</span><h1>{{ notice() || 'Loading authoring workspace…' }}</h1><p>Preparing the catalog, runtime profile, and isolated editor session.</p></section>
    } @else {
      <div class="builder-page" [class.is-full-canvas]="fullCanvas()">
        <section class="builder-toolbar panel">
          <div class="builder-toolbar-heading" [class.is-dirty]="dirty()"><span class="eyebrow">MODEL BUILDER</span><span class="builder-badge" aria-live="polite"><svg lucideHammer [size]="11"></svg><span>{{ busy() ? 'Saving…' : dirty() ? 'Drafting' : 'Saved' }}</span></span></div>
          <div class="builder-identity"><input class="project-title" aria-label="Model name" [value]="name()" (input)="updateDraftMetadata({ name: $any($event.target).value })" placeholder="Untitled model" /><div class="description-control" #descriptionControl><button type="button" class="toolbar-description" [class.has-description]="description()" [attr.aria-expanded]="descriptionOpen()" aria-haspopup="dialog" (click)="toggleDescription()"><svg lucideFileText [size]="14"></svg><span>Description</span>@if (description()) { <i aria-hidden="true"></i> }</button>@if (descriptionOpen()) { <div class="description-popover" role="dialog" aria-label="Model description editor"><div class="description-popover-heading"><div><span class="eyebrow">MODEL METADATA</span><strong>Description</strong></div><button type="button" class="popover-close" (click)="cancelDescription()" aria-label="Close description editor"><svg lucideX [size]="14"></svg></button></div><textarea #descriptionInput [value]="descriptionDraft()" (input)="descriptionDraft.set($any($event.target).value)" maxlength="2000" placeholder="Describe the model purpose, inputs, or expected outputs." aria-label="Model description"></textarea><div class="description-popover-footer"><small>{{ descriptionDraft().length }}/2000</small><div><button type="button" class="secondary" (click)="cancelDescription()">Cancel</button>@if (description() || descriptionDraft()) { <button type="button" class="secondary danger-text" (click)="clearDescription()">Clear</button> }<button type="button" class="primary" (click)="saveDescription()">Save</button></div></div></div> }</div></div>
          <label class="builder-target"><span>Backend</span><select aria-label="Execution backend" [value]="backend()" (change)="updateDraftMetadata({ executionBackend: $any($event.target).value })">@for (option of backendOptions; track option[0]) { <option [value]="option[0]">{{ option[1] }}</option> }</select></label><label class="toolbar-toggle" title="Optional Keras API layer over the selected native backend."><span>Enable Keras</span><input type="checkbox" aria-label="Enable Keras" [checked]="kerasEnabled()" (change)="updateDraftMetadata({ kerasEnabled: $any($event.target).checked })" /><strong>{{ kerasEnabled() ? 'On' : 'Off' }}</strong></label>
          <div class="builder-history"><button type="button" title="Undo" aria-label="Undo" (click)="store.undo()" [disabled]="!store.canUndo()"><svg lucideUndo2 [size]="14"></svg></button><button type="button" title="Redo" aria-label="Redo" (click)="store.redo()" [disabled]="!store.canRedo()"><svg lucideRedo2 [size]="14"></svg></button></div><div class="builder-actions"><button class="secondary" type="button" (click)="validate()" [disabled]="busy()"><svg lucideLayoutGrid [size]="14"></svg><span class="action-label">Validate</span></button><button class="primary" type="button" (click)="save()" [disabled]="busy()"><svg lucideSave [size]="14"></svg><span class="action-label">{{ busy() ? 'Saving…' : 'Save model' }}</span></button>@if (activeTab()?.projectId) { <button class="secondary" type="button" (click)="openWorkspace()" [disabled]="busy()"><svg lucidePlay [size]="14"></svg><span class="action-label">Open workspace</span></button> }</div>
        </section>
        <div class="builder-mobile-actions"><button type="button" [class.active]="mobilePanel() === 'catalog'" (click)="togglePanel('catalog')">Catalog</button><button type="button" [class.active]="mobilePanel() === 'inspector'" (click)="togglePanel('inspector')">Inspector</button></div>
        <div class="builder-workspace-grid"><div class="drawer-slot builder-catalog" [class.is-open]="mobilePanel() === 'catalog'"><nw-node-catalog [executionBackend]="backend()" [kerasEnabled]="kerasEnabled()"></nw-node-catalog></div><section id="builder-editor-workspace" class="builder-center-workspace"><nav class="model-tab-bar panel" aria-label="Open models"><div class="model-tab-list" role="tablist" aria-label="Model editor tabs">@for (tab of tabs(); track tab.id) { <div class="model-tab" [class.active]="tab.id === activeTabId()"><button type="button" role="tab" class="model-tab-select" [attr.aria-selected]="tab.id === activeTabId()" title="Open {{ tab.name || 'Untitled model' }}" (click)="switchTab(tab.id)"><span class="model-tab-status" [class.dirty]="tab.dirty">{{ tab.dirty ? '●' : '○' }}</span><span class="model-tab-name">{{ tab.name || 'Untitled model' }}</span><small>{{ tab.projectId ? 'r' + tab.revision : 'draft' }}</small></button><button type="button" class="model-tab-close" [attr.aria-label]="'Close ' + (tab.name || 'Untitled model')" (click)="requestCloseTab(tab.id)"><svg lucideX [size]="14"></svg></button></div> }</div><button type="button" class="model-tab-new" aria-label="New model" (click)="openNewTab()"><svg lucidePlus [size]="15"></svg><span class="action-label">New model</span></button></nav><nw-graph-canvas [modelName]="name()" [dirty]="dirty()" [busy]="busy()" [fullCanvas]="fullCanvas()" (enterFocus)="fullCanvas.set(true)" (exitFocus)="fullCanvas.set(false)" (save)="save()"></nw-graph-canvas><nw-resizable-validation-region [expanded]="validationExpanded()" [height]="validationHeight()" (heightChange)="updateActiveTab({ validationHeight: $event })"><nw-validation-panel [result]="validation()" [expanded]="validationExpanded()" (expandedChange)="updateActiveTab({ validationExpanded: $event })"></nw-validation-panel></nw-resizable-validation-region></section><div class="drawer-slot builder-inspector" [class.is-open]="mobilePanel() === 'inspector'"><nw-properties-inspector [executionBackend]="backend()" [kerasEnabled]="kerasEnabled()"></nw-properties-inspector></div></div>
        @if (leaveNavigation()) { <div class="dialog-backdrop"><section class="confirm-dialog panel" role="dialog" aria-modal="true" aria-labelledby="leave-builder-title"><span class="eyebrow">UNSAVED CHANGES</span><h2 id="leave-builder-title">Leave the Model Builder?</h2><p>{{ dirtyTabCount() === 1 ? 'One model tab has unsaved changes.' : dirtyTabCount() + ' model tabs have unsaved changes.' }} Leave and discard them?</p><div class="dialog-actions"><button type="button" class="secondary" (click)="stayOnBuilder()">Stay</button><button type="button" class="primary" (click)="discardAndLeave()">Discard changes</button></div></section></div> } @if (pendingCloseTabId()) { <div class="dialog-backdrop"><section class="confirm-dialog panel" role="dialog" aria-modal="true" aria-labelledby="close-tab-title"><span class="eyebrow">UNSAVED TAB</span><h2 id="close-tab-title">Close this model tab?</h2><p>Its architecture edits have not been saved. Close and discard them?</p><div class="dialog-actions"><button type="button" class="secondary" (click)="pendingCloseTabId.set(null)">Keep tab</button><button type="button" class="primary" (click)="confirmCloseTab()">Close tab</button></div></section></div> } @if (notice()) { <button class="notice builder-notice" type="button" (click)="notice.set(null)">{{ notice() }}</button> }
      </div>
    }
  `,
})
export class ModelBuilderPageComponent {
  readonly modelId = input<string | undefined>()
  readonly resources = inject(AppResourcesService)
  readonly store = inject(GraphEditorStore)
  private readonly tabsStore = inject(BuilderTabsStore)
  private readonly api = inject(ApiService)
  private readonly router = inject(Router)
  readonly tabs = this.tabsStore.tabs
  readonly activeTabId = this.tabsStore.activeTabId
  readonly activeTab = computed(() => this.tabs().find((tab) => tab.id === this.activeTabId()) ?? null)
  readonly dirtyTabCount = computed(() => this.tabs().filter((tab) => tab.dirty).length)
  readonly name = signal('Untitled model')
  readonly description = signal('')
  readonly descriptionDraft = signal('')
  readonly descriptionOpen = signal(false)
  readonly backend = signal<ExecutionBackend>('pytorch')
  readonly kerasEnabled = signal(true)
  readonly projectRevision = signal<number | null>(null)
  readonly validation = signal<ValidationResult | null>(null)
  readonly validationExpanded = signal(false)
  readonly validationHeight = signal(DEFAULT_VALIDATION_HEIGHT)
  readonly busy = signal(false)
  readonly notice = signal<string | null>(null)
  readonly mobilePanel = signal<'catalog' | 'inspector' | null>(null)
  readonly fullCanvas = signal(false)
  readonly initialized = signal(false)
  readonly leaveNavigation = signal<string | null>(null)
  readonly pendingCloseTabId = signal<string | null>(null)
  readonly backendOptions = Object.entries(executionBackendShortLabels) as [ExecutionBackend, string][]
  private initializedKey: string | null = null
  private allowNextNavigation = false
  private loadSequence = 0
  private readonly descriptionControl = viewChild<ElementRef<HTMLElement>>('descriptionControl')

  constructor() {
    effect(() => {
      this.store.setManifests(this.resources.manifests())
      const id = this.modelId()
      const routeKey = id ? `project:${id}` : 'new'
      if (!this.resources.catalogReady() || routeKey === this.initializedKey) return
      this.initializedKey = routeKey
      void this.initialize(id, ++this.loadSequence)
    })
    effect(() => {
      const activeId = this.activeTabId()
      const nodes = this.store.nodes()
      const edges = this.store.edges()
      this.store.modelConfiguration()
      this.store.exportSettings()
      this.store.futureTrainingConfiguration()
      this.store.selectedNodeId()
      this.store.selectedEdgeId()
      this.store.copiedNodeIds()
      this.store.past()
      this.store.future()
      const metadata = { name: this.name(), description: this.description(), executionBackend: this.backend(), kerasEnabled: this.kerasEnabled() }
      if (!this.initialized() || !activeId) return
      const current = untracked(() => this.tabsStore.tabs().find((tab) => tab.id === activeId))
      if (!current) return
      const graph = untracked(() => this.store.captureEditorSession())
      const dirty = current.savedBaseline !== draftSnapshot(metadata.name, metadata.description, metadata.executionBackend, metadata.kerasEnabled, untracked(() => this.store.toGraph()))
      this.tabsStore.updateTab(activeId, { ...metadata, graph, dirty })
      void nodes
      void edges
    })
  }

  dirty(): boolean { return this.activeTab()?.dirty ?? false }

  private async initialize(modelId: string | undefined, sequence: number): Promise<void> {
    this.initialized.set(false)
    this.notice.set(null)
    const existing = modelId
      ? this.tabsStore.tabs().find((tab) => tab.projectId === modelId)
      : this.tabsStore.tabs().find((tab) => tab.id === this.tabsStore.activeTabId() && tab.projectId === null)
    if (existing) { this.restoreTab(existing); this.tabsStore.focusTab(existing.id); this.initialized.set(true); return }
    if (!modelId) {
      this.store.reset()
      const graph = this.store.captureEditorSession()
      const tabId = this.tabsStore.openTab(untitledTabInput(preferredBackend(this.resources.backendStatuses()), graph, this.store.toGraph()))
      this.restoreTab(this.tabsStore.tabs().find((tab) => tab.id === tabId) as BuilderTab)
      this.initialized.set(true)
      return
    }
    try {
      const project = await this.api.getProject(modelId)
      if (sequence !== this.loadSequence) return
      this.store.replaceGraph(project)
      const graph = this.store.captureEditorSession()
      const tabId = this.tabsStore.openTab(projectTabInput(project, graph))
      this.restoreTab(this.tabsStore.tabs().find((tab) => tab.id === tabId) as BuilderTab)
    } catch (reason) {
      if (sequence === this.loadSequence) this.notice.set(`Unable to open model: ${reason instanceof Error ? reason.message : String(reason)}`)
    } finally {
      if (sequence === this.loadSequence) this.initialized.set(true)
    }
  }

  private restoreTab(tab: BuilderTab): void {
    this.store.restoreEditorSession(tab.graph)
    this.name.set(tab.name)
    this.description.set(tab.description)
    this.descriptionDraft.set(tab.description)
    this.backend.set(tab.executionBackend)
    this.kerasEnabled.set(tab.kerasEnabled)
    this.projectRevision.set(tab.projectId ? tab.revision : null)
    this.validation.set(tab.validation)
    this.validationExpanded.set(tab.validationExpanded)
    this.validationHeight.set(tab.validationHeight)
  }

  updateDraftMetadata(patch: DraftMetadataPatch): void {
    if (patch.name !== undefined) this.name.set(patch.name)
    if (patch.description !== undefined) this.description.set(patch.description)
    if (patch.executionBackend !== undefined) this.backend.set(patch.executionBackend)
    if (patch.kerasEnabled !== undefined) this.kerasEnabled.set(patch.kerasEnabled)
  }

  updateActiveTab(patch: Partial<Pick<BuilderTab, 'validation' | 'validationExpanded' | 'validationHeight'>>): void {
    const id = this.activeTabId()
    if (id) {
      if (patch.validation !== undefined) this.validation.set(patch.validation)
      if (patch.validationExpanded !== undefined) this.validationExpanded.set(patch.validationExpanded)
      if (patch.validationHeight !== undefined) this.validationHeight.set(patch.validationHeight)
      this.tabsStore.updateTab(id, patch)
    }
  }

  toggleDescription(): void { if (this.descriptionOpen()) this.cancelDescription(); else { this.descriptionDraft.set(this.description()); this.descriptionOpen.set(true) } }
  saveDescription(): void { this.updateDraftMetadata({ description: this.descriptionDraft().trim() }); this.descriptionOpen.set(false) }
  clearDescription(): void { this.descriptionDraft.set(''); this.updateDraftMetadata({ description: '' }); this.descriptionOpen.set(false) }
  cancelDescription(): void { this.descriptionDraft.set(this.description()); this.descriptionOpen.set(false) }
  togglePanel(panel: 'catalog' | 'inspector'): void { this.mobilePanel.set(this.mobilePanel() === panel ? null : panel) }

  switchTab(tabId: string): void {
    if (tabId === this.activeTabId()) return
    this.syncActiveTab()
    const target = this.tabsStore.tabs().find((tab) => tab.id === tabId)
    if (!target) return
    this.restoreTab(target)
    this.tabsStore.focusTab(tabId)
    this.initializedKey = target.projectId ? `project:${target.projectId}` : 'new'
    const nextPath = target.projectId ? `/builder/${target.projectId}` : '/builder'
    if (this.router.url !== nextPath) void this.router.navigateByUrl(nextPath, { replaceUrl: true })
  }

  openNewTab(): void {
    this.syncActiveTab()
    this.store.reset()
    const graph = this.store.captureEditorSession()
    const tabId = this.tabsStore.openTab(untitledTabInput(preferredBackend(this.resources.backendStatuses()), graph, this.store.toGraph()))
    const tab = this.tabsStore.tabs().find((item) => item.id === tabId)
    if (tab) this.restoreTab(tab)
    this.initializedKey = 'new'
    this.initialized.set(true)
    if (this.router.url !== '/builder') void this.router.navigateByUrl('/builder', { replaceUrl: true })
  }

  requestCloseTab(tabId: string): void { const tab = this.tabsStore.tabs().find((item) => item.id === tabId); if (!tab) return; if (tab.dirty) this.pendingCloseTabId.set(tabId); else this.closeTabAndNavigate(tabId) }
  confirmCloseTab(): void { const id = this.pendingCloseTabId(); this.pendingCloseTabId.set(null); if (id) this.closeTabAndNavigate(id) }

  private syncActiveTab(): void {
    const id = this.activeTabId();
    if (!id) return
    const current = this.tabsStore.tabs().find((tab) => tab.id === id)
    if (!current) return
    const graph = this.store.captureEditorSession()
    this.tabsStore.updateTab(id, { graph, name: this.name(), description: this.description(), executionBackend: this.backend(), kerasEnabled: this.kerasEnabled(), dirty: current.savedBaseline !== draftSnapshot(this.name(), this.description(), this.backend(), this.kerasEnabled(), this.store.toGraph()) })
  }

  private closeTabAndNavigate(tabId: string): void {
    const wasActive = this.activeTabId() === tabId
    if (wasActive) this.syncActiveTab()
    this.tabsStore.closeTab(tabId)
    if (!wasActive) return
    const next = this.tabsStore.tabs().find((tab) => tab.id === this.activeTabId())
    if (!next) { this.openNewTab(); return }
    this.restoreTab(next)
    const nextPath = next.projectId ? `/builder/${next.projectId}` : '/builder'
    if (this.router.url !== nextPath) void this.router.navigateByUrl(nextPath, { replaceUrl: true })
  }

  async validate(): Promise<void> {
    this.busy.set(true)
    try {
      const result = await this.api.validate(this.store.toGraph(), this.backend(), this.kerasEnabled())
      this.validation.set(result)
      this.updateActiveTab({ validation: result })
      this.store.applyShapeSummaries(result.shape_summaries)
      this.notice.set(result.valid ? 'Model execution profile is valid.' : 'Model has blocking findings.')
    } catch (reason) { this.notice.set(`Validation failed: ${reason instanceof Error ? reason.message : String(reason)}`) } finally { this.busy.set(false) }
  }

  async save(): Promise<void> {
    const tab = this.activeTab(); if (!tab) return
    this.busy.set(true)
    try {
      const graph = this.store.toGraph()
      const payload = { name: this.name().trim() || 'Untitled model', description: this.description(), execution_backend: this.backend(), keras_enabled: this.kerasEnabled(), graph }
      const project = tab.projectId ? await this.api.updateProject(tab.projectId, { ...payload, expected_revision: tab.revision }) : await this.api.createProject(payload)
      const session = this.store.captureEditorSession()
      this.tabsStore.updateTab(tab.id, { projectId: project.id, name: project.name, description: project.description, executionBackend: project.execution_backend, kerasEnabled: project.keras_enabled, revision: project.current_revision, graph: session, savedBaseline: draftSnapshot(project.name, project.description, project.execution_backend, project.keras_enabled, graph), dirty: false })
      this.name.set(project.name); this.description.set(project.description); this.backend.set(project.execution_backend); this.kerasEnabled.set(project.keras_enabled); this.projectRevision.set(project.current_revision)
      await this.resources.refreshProjects()
      this.notice.set(`Saved revision ${project.current_revision}. Open the model workspace to compile or run inference.`)
      if (!tab.projectId) { this.initializedKey = `project:${project.id}`; await this.router.navigate(['/builder', project.id], { replaceUrl: true }) }
    } catch (reason) { this.notice.set(`Save failed: ${reason instanceof Error ? reason.message : String(reason)}`) } finally { this.busy.set(false) }
  }

  openWorkspace(): void { const id = this.activeTab()?.projectId; if (id) void this.router.navigate(['/models', id]) }

  canDeactivate(nextUrl = ''): boolean {
    if (this.allowNextNavigation) { this.allowNextNavigation = false; return true }
    if (nextUrl.startsWith('/builder') || this.dirtyTabCount() === 0) return true
    this.leaveNavigation.set(nextUrl)
    return false
  }

  stayOnBuilder(): void { this.leaveNavigation.set(null) }
  discardAndLeave(): void {
    const target = this.leaveNavigation()
    if (!target) return
    for (const tab of this.tabsStore.tabs().filter((item) => item.dirty)) this.tabsStore.closeTab(tab.id)
    this.leaveNavigation.set(null)
    this.allowNextNavigation = true
    void this.router.navigateByUrl(target)
  }

  @HostListener('window:beforeunload', ['$event'])
  protectBeforeUnload(event: BeforeUnloadEvent): void { if (!this.dirtyTabCount()) return; event.preventDefault(); event.returnValue = '' }

  @HostListener('document:pointerdown', ['$event'])
  closeDescriptionOnOutsideClick(event: PointerEvent): void { const control = this.descriptionControl()?.nativeElement; if (this.descriptionOpen() && control && event.target instanceof Node && !control.contains(event.target)) this.cancelDescription() }

  @HostListener('document:keydown', ['$event'])
  closeOnEscape(event: KeyboardEvent): void { if (event.key === 'Escape' && this.descriptionOpen()) { event.preventDefault(); this.cancelDescription(); return } if (event.key === 'Escape' && this.fullCanvas()) { event.preventDefault(); this.fullCanvas.set(false) } }
}
