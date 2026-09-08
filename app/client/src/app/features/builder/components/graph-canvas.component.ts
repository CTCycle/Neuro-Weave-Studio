import { Component, EventEmitter, HostListener, computed, effect, inject, Input, Output, signal, viewChild } from '@angular/core'
import type { OnChanges, SimpleChanges } from '@angular/core'
import { LucideLock, LucideMaximize2, LucideSave, LucideUnlock } from '@lucide/angular'
import { FCanvasComponent, FFlowModule, FZoomDirective, type FCreateConnectionEvent, type FDeleteSelectedEvent, type FEventTrigger, type FMoveNodesEvent, type FSelectionChangeEvent } from '@foblex/flow'
import type { EditorNode } from '../../../domain/editor.types'
import { ApiService } from '../../../core/api/api.service'
import type { ExecutionBackend, ValidationDiagnostic, ValidationResult } from '../../../domain/types'
import { GraphEditorStore } from '../state/graph-editor.store'
import { connectionCandidates, wouldCreateCycle } from '../graph/graph-validation'
import { GRAPH_NODE_SIZE, connectorId, flowPositionFromDrop, graphFitPadding, normalizeConnection, portOffset } from '../graph/foblex-graph.adapter'

const NO_CONNECTION_TARGET_ID = '__neuro_weave_no_compatible_target__'

@Component({
  selector: 'nw-graph-canvas',
  standalone: true,
  imports: [FFlowModule, LucideLock, LucideMaximize2, LucideSave, LucideUnlock],
  template: `
    <div class="canvas-shell" [class.is-read-only]="readOnly" [class.is-viewport-locked]="viewportLocked()" [class.has-connection-feedback]="!!activeConnectionFeedback()" (contextmenu)="openCanvasMenu($event)" (dragover)="allowDrop($event)" (drop)="dropNode($event)">
      <f-flow fDraggable [fDraggableDisabled]="readOnly" [fCanvasMoveTrigger]="canvasMoveTrigger" [vCellSize]="16" [hCellSize]="16" [fCellSizeWhileDragging]="16" (fDragStarted)="startDrag()" (fDragEnded)="store.endDrag()" (fMoveNodes)="moveNodes($event)" (fSelectionChange)="select($event)" (fDeleteSelected)="deleteSelected($event)" (fCreateConnection)="createConnection($event)" (fNodesRendered)="fitOnNodesRendered()">
        <f-background><f-circle-pattern color="rgba(118, 150, 164, .16)" [radius]="24"></f-circle-pattern></f-background>
        @if (minimapVisible()) { <f-minimap [fMinSize]="50" [fNodeRenderLimit]="1000"></f-minimap> }
        <f-canvas fZoom [fZoom]="!viewportLocked()" [fZoomMinimum]="0.2" [fZoomMaximum]="2.5" [fZoomStep]="0.1">
          <f-connection-for-create fType="straight" fOffset="20"></f-connection-for-create>
          <ng-container ngProjectAs="[fConnections]">@for (edge of store.edges(); track edge.id) { <f-connection fType="straight" fOffset="20" [fSelectionDisabled]="readOnly" [fConnectionId]="edge.id" [fSourceId]="connectorId(edge.source, 'output', edge.sourceHandle || 'output')" [fTargetId]="connectorId(edge.target, 'input', edge.targetHandle || 'input')" [class.is-selected]="store.selectedEdgeId() === edge.id" [class.is-invalid]="hasEdgeError(edge.id)"></f-connection> }</ng-container>
          <ng-container ngProjectAs="[fNodes]">@for (node of store.nodes(); track node.id) {
            <div fNode [fNodeId]="node.id" [fNodePosition]="node.position" [fNodeDraggingDisabled]="readOnly || node.locked" [fNodeSelectionDisabled]="readOnly" [fNodeSize]="nodeSize(node)" [class.is-selected]="store.selectedNodeId() === node.id" [class.has-validation-error]="hasNodeError(node.id)" [attr.aria-label]="node.data.label" (click)="selectNode(node.id)">
              <div fDragHandle class="nw-flow-node" [style.--node-accent]="node.data.manifest.ui.accent" (contextmenu)="openNodeMenu($event, node.id)"><div class="nw-flow-node-status" aria-hidden="true">@if (node.locked) { <svg lucideLock [size]="13"></svg> }</div>@if (store.isEffectiveOutput(node.id)) { <span class="nw-flow-output-indicator" [attr.aria-label]="store.isOutput(node.id) ? 'Explicit workflow output' : 'Inferred workflow output'" [title]="store.isOutput(node.id) ? 'Explicit workflow output' : 'Inferred workflow output'"></span> }<div class="nw-flow-node-eyebrow">{{ node.data.manifest.category }}</div><div class="nw-flow-node-heading"><strong>{{ node.data.label }}</strong></div><small>{{ node.data.manifest.id }}</small><div class="nw-flow-node-footer">{{ shapeLabel(node) }}</div><div class="nw-flow-port-layer inputs">@for (port of node.data.manifest.inputs; track port.id; let index = $index) { @if (portEnabled(node, 'input', port.id)) { <span class="nw-flow-port input" [style.top]="portOffset(index, node.data.manifest.inputs.length) + '%'" [attr.data-port-id]="port.id" [attr.data-port-direction]="'input'" [attr.aria-label]="'Input ' + port.display_name" [title]="port.display_name + ' · ' + port.kind"><i fConnector fConnectorType="target" [fConnectorId]="connectorId(node.id, 'input', port.id)" [fConnectorMultiple]="port.accepts_multiple"></i></span> } }</div><div class="nw-flow-port-layer outputs">@for (port of node.data.manifest.outputs; track port.id; let index = $index) { @if (portEnabled(node, 'output', port.id)) { <span class="nw-flow-port output" [style.top]="portOffset(index, node.data.manifest.outputs.length) + '%'" [attr.data-port-id]="port.id" [attr.data-port-direction]="'output'" [attr.aria-label]="'Output ' + port.display_name" [title]="port.display_name + ' · ' + port.kind"><i fConnector fConnectorType="source" [fConnectorId]="connectorId(node.id, 'output', port.id)" [fConnectorMultiple]="true" [fCanBeConnectedTo]="connectionTargets(node.id, port.id)"></i></span> } }</div></div>
            </div>
          } </ng-container>
        </f-canvas>
        <div class="flow-controls canvas-flow-controls" role="group" aria-label="Canvas zoom controls"><button type="button" class="canvas-tool" (click)="zoomIn()" [disabled]="viewportLocked()" aria-label="Zoom in" [title]="viewportLocked() ? 'Unlock canvas to zoom in' : 'Zoom in'">+</button><button type="button" class="canvas-tool" (click)="zoomOut()" [disabled]="viewportLocked()" aria-label="Zoom out" [title]="viewportLocked() ? 'Unlock canvas to zoom out' : 'Zoom out'">−</button><button type="button" class="canvas-tool" (click)="resetView()" [disabled]="viewportLocked()" aria-label="Reset view" [title]="viewportLocked() ? 'Unlock canvas to reset the view' : 'Reset view'"><svg lucideMaximize2 [size]="13"></svg></button></div>
        @if (!readOnly) { <div class="canvas-editor-tools" [class.is-focus-mode]="fullCanvas" role="toolbar" aria-label="Graph editor controls"><div class="canvas-tool-group"><button type="button" class="canvas-tool" (click)="minimapVisible.set(!minimapVisible())" [attr.aria-label]="minimapVisible() ? 'Hide minimap' : 'Show minimap'" [title]="minimapVisible() ? 'Hide minimap' : 'Show minimap'">{{ minimapVisible() ? 'Hide map' : 'Show map' }}</button><button type="button" class="canvas-tool canvas-lock-tool" [class.is-active]="viewportLocked()" [attr.aria-pressed]="viewportLocked()" [attr.aria-label]="viewportLocked() ? 'Unlock canvas' : 'Lock canvas'" [title]="viewportLocked() ? 'Unlock canvas' : 'Lock canvas'" (click)="toggleViewportLock()">@if (viewportLocked()) { <svg lucideUnlock [size]="13"></svg><span>Unlock canvas</span> } @else { <svg lucideLock [size]="13"></svg><span>Lock canvas</span> }</button></div>@if (fullCanvas) { <div class="canvas-focus-bar"><span class="canvas-focus-model">{{ modelName }}</span><span class="canvas-focus-draft" [class.is-dirty]="dirty">{{ busy ? 'Saving…' : dirty ? 'Draft' : 'Saved' }}</span><button type="button" class="canvas-tool canvas-focus-save" (click)="save.emit()" [disabled]="busy" aria-label="Save model"><svg lucideSave [size]="14"></svg><span>Save</span></button><button type="button" class="canvas-tool canvas-focus-exit" (click)="exitFocus.emit()" [disabled]="busy" aria-label="Exit full canvas">Exit full canvas</button></div> } @else { <button type="button" class="canvas-tool" (click)="enterFocus.emit()" aria-label="Enter full canvas">Focus</button> }</div> }
      </f-flow>
      @if (!readOnly && activeConnectionFeedback(); as feedback) { <div class="connection-feedback" role="status" aria-live="assertive">{{ feedback }}</div> }
      @if (!readOnly && !store.nodes().length) { <div class="canvas-empty-state"><span class="eyebrow">EMPTY GRAPH</span><strong>Drop an operation here</strong><span>Choose a node from the catalog to start a model.</span></div> }
      @if (readOnly && !store.nodes().length) { <div class="preview-empty">No renderable nodes in this revision.</div> }
      @if (!readOnly) { <div class="canvas-hint">Drop a node to build the model • Ctrl+L auto-layout</div> }
      @if (!readOnly && canvasMenu(); as menu) { <div class="canvas-context-menu" role="menu" [style.left.px]="menu.x" [style.top.px]="menu.y" (pointerdown)="$event.stopPropagation()"><button type="button" role="menuitem" (click)="fitView(); closeMenus()">Fit workflow to view</button><button type="button" role="menuitem" (click)="autoLayout(); closeMenus()">Auto layout</button><button type="button" role="menuitem" class="is-destructive" (click)="requestReset(); closeMenus()">Reset canvas</button></div> }
      @if (!readOnly && nodeMenu(); as menu) { <div class="canvas-context-menu node-context-menu" role="menu" [style.left.px]="menu.x" [style.top.px]="menu.y" (pointerdown)="$event.stopPropagation()"><button type="button" role="menuitem" (click)="duplicateWithDefaults(menu.nodeId); closeMenus()">Duplicate with defaults</button><button type="button" role="menuitem" (click)="duplicateExact(menu.nodeId); closeMenus()">Duplicate exact</button><button type="button" role="menuitem" (click)="copyNode(menu.nodeId); closeMenus()">Copy node</button><button type="button" role="menuitem" (click)="resetParameters(menu.nodeId); closeMenus()">Reset parameters</button><button type="button" role="menuitem" (click)="disconnectNode(menu.nodeId); closeMenus()">Disconnect all</button><button type="button" role="menuitem" [disabled]="isInput(menu.nodeId)" [title]="isInput(menu.nodeId) ? 'Input nodes cannot be workflow outputs' : ''" (click)="toggleOutput(menu.nodeId); closeMenus()">{{ store.isOutput(menu.nodeId) ? 'Remove from outputs' : 'Mark as output' }}</button><button type="button" role="menuitem" (click)="toggleLock(menu.nodeId); closeMenus()">{{ isLocked(menu.nodeId) ? 'Unlock node' : 'Lock node' }}</button><button type="button" role="menuitem" class="is-destructive" (click)="deleteNode(menu.nodeId); closeMenus()">Delete node</button></div> }
    </div>
  `,
})
export class GraphCanvasComponent implements OnChanges {
  @Input() readOnly = false
  @Input() modelName = ''
  @Input() dirty = false
  @Input() busy = false
  @Input() executionBackend: ExecutionBackend = 'pytorch'
  @Input() kerasEnabled = true
  private currentValidation: ValidationResult | null = null
  @Input()
  set validation(value: ValidationResult | null) {
    if (this.currentValidation !== value) this.setConnectionFeedback(null)
    this.currentValidation = value
  }
  get validation(): ValidationResult | null { return this.currentValidation }
  @Input() fullCanvas = false
  @Output() readonly enterFocus = new EventEmitter<void>()
  @Output() readonly exitFocus = new EventEmitter<void>()
  @Output() readonly save = new EventEmitter<void>()
  @Output() readonly resetRequested = new EventEmitter<void>()
  readonly store = inject(GraphEditorStore)
  private readonly api = inject(ApiService)
  readonly minimapVisible = signal(true)
  readonly viewportLocked = signal(false)
  readonly canvasMoveTrigger: FEventTrigger = () => !this.viewportLocked()
  private readonly canvas = viewChild(FCanvasComponent)
  private readonly zoom = viewChild(FZoomDirective)
  private graphFitted = false
  readonly canvasMenu = signal<{ x: number; y: number } | null>(null)
  readonly nodeMenu = signal<{ x: number; y: number; nodeId: string } | null>(null)
  readonly connectionFeedback = signal<string | null>(null)
  private readonly connectionFeedbackRevision = signal<number | null>(null)
  readonly activeConnectionFeedback = computed(() => {
    const feedback = this.connectionFeedback()
    return feedback !== null && this.connectionFeedbackRevision() === this.store.graphRevision()
      ? feedback
      : null
  })
  private readonly noConnectionTargets = [NO_CONNECTION_TARGET_ID]
  private readonly connectionTargetMap = signal(new Map<string, string[]>())

  constructor() {
    effect(() => {
      this.store.graphRevision()
      this.setConnectionFeedback(null)
      if (!this.store.nodes().length) this.graphFitted = false
      void this.refreshConnectionTargets()
    })
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['executionBackend'] || changes['kerasEnabled'] || changes['readOnly']) {
      void this.refreshConnectionTargets()
    }
  }

  private async refreshConnectionTargets(): Promise<void> {
    const nodes = this.store.nodes()
    const edges = this.store.edges()
    const emptyTargets = new Map<string, string[]>()
    const candidateTargets = new Map<string, { sourceKey: string; targetId: string; isCycle: boolean }>()
    const candidates: Array<{
      id: string
      source_node_id: string
      source_port_id: string
      target_node_id: string
      target_port_id: string
    }> = []
    let candidateIndex = 0

    for (const node of nodes) {
      for (const port of node.data.manifest.outputs) {
        const sourceKey = `${node.id}|${port.id}`
        emptyTargets.set(sourceKey, this.noConnectionTargets)
        for (const connection of connectionCandidates(
          nodes,
          edges,
          node.id,
          port.id,
          this.executionBackend,
          this.kerasEnabled,
        )) {
          const id = `snap-${candidateIndex++}`
          candidates.push({
            id,
            source_node_id: connection.source,
            source_port_id: connection.sourceHandle,
            target_node_id: connection.target,
            target_port_id: connection.targetHandle,
          })
          candidateTargets.set(id, {
            sourceKey,
            targetId: connectorId(connection.target, 'input', connection.targetHandle),
            isCycle: wouldCreateCycle(nodes, edges, connection.source, connection.target),
          })
        }
      }
    }

    this.connectionTargetMap.set(emptyTargets)
    if (this.readOnly || !candidates.length) return

    const expectedRevision = this.store.graphRevision()
    const expectedBackend = this.executionBackend
    const expectedKeras = this.kerasEnabled
    try {
      const result = await this.api.validateConnectionCandidates(
        this.store.toGraph(),
        candidates,
        expectedBackend,
        expectedKeras,
      )
      if (expectedRevision !== this.store.graphRevision()
        || expectedBackend !== this.executionBackend
        || expectedKeras !== this.kerasEnabled) return

      const validatedTargets = new Map<string, string[]>()
      for (const sourceKey of emptyTargets.keys()) validatedTargets.set(sourceKey, [])
      for (const id of result.valid_connection_ids) {
        const candidate = candidateTargets.get(id)
        if (!candidate) continue
        validatedTargets.get(candidate.sourceKey)?.push(candidate.targetId)
      }
      for (const candidate of candidateTargets.values()) {
        if (candidate.isCycle) validatedTargets.get(candidate.sourceKey)?.push(candidate.targetId)
      }
      for (const [sourceKey, targetIds] of validatedTargets) {
        if (!targetIds.length) validatedTargets.set(sourceKey, this.noConnectionTargets)
      }
      this.connectionTargetMap.set(validatedTargets)
    } catch {
      // Keep all connectors non-snappable if authoritative validation is unavailable.
    }
  }

  private setConnectionFeedback(message: string | null): void {
    this.connectionFeedbackRevision.set(message === null ? null : this.store.graphRevision())
    this.connectionFeedback.set(message)
  }

  allowDrop(event: DragEvent): void { if (!this.readOnly) event.preventDefault() }
  startDrag(): void {
    this.setConnectionFeedback(null)
    this.store.beginDrag()
  }
  openCanvasMenu(event: MouseEvent): void {
    if (this.readOnly) return
    event.preventDefault()
    this.nodeMenu.set(null)
    this.canvasMenu.set(this.menuPosition(event))
  }
  openNodeMenu(event: MouseEvent, nodeId: string): void {
    if (this.readOnly) return
    event.preventDefault()
    event.stopPropagation()
    this.store.selectNode(nodeId)
    this.canvasMenu.set(null)
    this.nodeMenu.set({ ...this.menuPosition(event), nodeId })
  }
  closeMenus(): void { this.canvasMenu.set(null); this.nodeMenu.set(null) }
  requestReset(): void { this.resetRequested.emit() }
  autoLayout(): void { this.store.autoLayout(this.store.nodes(), this.store.edges()); if (!this.viewportLocked()) this.fitView() }
  duplicateWithDefaults(nodeId: string): void { this.store.duplicateWithDefaults(nodeId) }
  duplicateExact(nodeId: string): void { this.store.duplicateExact(nodeId) }
  copyNode(nodeId: string): void { this.store.selectNode(nodeId); this.store.copySelected() }
  resetParameters(nodeId: string): void { this.store.resetParameters(nodeId) }
  disconnectNode(nodeId: string): void { this.store.disconnectNode(nodeId) }
  toggleOutput(nodeId: string, portId?: string): void { this.store.toggleNodeOutput(nodeId, portId) }
  toggleLock(nodeId: string): void { this.store.toggleNodeLocked(nodeId) }
  deleteNode(nodeId: string): void { this.store.selectNode(nodeId); this.store.deleteSelected() }
  isLocked(nodeId: string): boolean { return this.store.nodes().find((node) => node.id === nodeId)?.locked ?? false }
  isInput(nodeId: string): boolean {
    return this.store.nodes().find((node) => node.id === nodeId)?.data.manifest.semantic_role === 'input'
  }
  private menuPosition(event: MouseEvent): { x: number; y: number } {
    const target = event.currentTarget instanceof HTMLElement ? event.currentTarget : null
    const shell = target?.closest('.canvas-shell') ?? target
    const bounds = shell?.getBoundingClientRect()
    return { x: Math.max(8, event.clientX - (bounds?.left ?? 0)), y: Math.max(8, event.clientY - (bounds?.top ?? 0)) }
  }
  dropNode(event: DragEvent): void {
    if (this.readOnly) return
    event.preventDefault()
    const payload = event.dataTransfer?.getData('application/neuro-weave-node')
    if (!payload) return
    try {
      const value = JSON.parse(payload) as { id: string; version: number }
      const manifest = this.store.manifests().find((item) => item.id === value.id && item.version === value.version)
      const canvas = event.currentTarget instanceof HTMLElement ? event.currentTarget : null
      if (manifest) this.store.addNode(manifest, canvas ? flowPositionFromDrop(event, canvas) : { x: event.offsetX, y: event.offsetY })
    } catch { /* The drop payload is external UI data; ignore malformed items. */ }
  }
  moveNodes(event: FMoveNodesEvent): void { this.store.moveNodes(event.nodes.map((node) => ({ id: node.id, position: { x: node.position.x, y: node.position.y } }))) }
  selectNode(nodeId: string): void { if (!this.readOnly) this.store.selectNode(nodeId) }
  select(event: FSelectionChangeEvent): void { if (this.readOnly) return; if (event.connectionIds.length) this.store.selectEdge(event.connectionIds[0]); else if (event.nodeIds.length) this.store.selectNode(event.nodeIds[0]); else this.store.clearSelection() }
  deleteSelected(_event: FDeleteSelectedEvent): void { if (!this.readOnly) this.store.deleteSelected() }
  async createConnection(event: FCreateConnectionEvent): Promise<void> {
    if (this.readOnly) return
    if (!event.targetId) {
      this.setConnectionFeedback('Connection not created. Drop on a compatible input port.')
      return
    }
    const connection = normalizeConnection(event.sourceId, event.targetId)
    if (!connection) {
      this.setConnectionFeedback('Connection not created. The selected ports are not compatible.')
      return
    }
    const expectedRevision = this.store.graphRevision()
    const expectedBackend = this.executionBackend
    const expectedKeras = this.kerasEnabled
    this.setConnectionFeedback('Checking connection…')
    try {
      const result = await this.api.validateConnection(
        this.store.toGraph(),
        {
          id: `candidate-${Date.now()}`,
          source_node_id: connection.source,
          source_port_id: connection.sourceHandle,
          target_node_id: connection.target,
          target_port_id: connection.targetHandle,
        },
        expectedBackend,
        expectedKeras,
      )
      if (expectedRevision !== this.store.graphRevision()
        || expectedBackend !== this.executionBackend
        || expectedKeras !== this.kerasEnabled) {
        this.setConnectionFeedback('Connection not created. The graph changed while it was being checked.')
        return
      }
      if (!result.valid) {
        this.setConnectionFeedback(this.connectionDiagnosticMessage(result.diagnostics))
        return
      }
      if (!this.store.commitConnection(connection, expectedRevision)) {
        this.setConnectionFeedback('Connection not created. The graph changed while it was being checked.')
        return
      }
      this.setConnectionFeedback(null)
    } catch (reason) {
      this.setConnectionFeedback(`Connection not created. ${reason instanceof Error ? reason.message : String(reason)}`)
    }
  }
  connectionTargets(nodeId: string, portId: string): string[] {
    return this.connectionTargetMap().get(`${nodeId}|${portId}`) ?? this.noConnectionTargets
  }
  connectorId = connectorId
  portOffset = portOffset
  shapeLabel(node: EditorNode): string {
    const output = Object.values(this.store.shapeSummary(node.id)?.outputs ?? {})[0]
    const shape = output?.shape.map((value) => value === null ? '?' : typeof value === 'object' ? value.name : value).join(' × ')
    return shape || 'Shape pending'
  }
  nodeSize(_node: EditorNode): { width: number; height: number } { return GRAPH_NODE_SIZE }

  hasNodeError(nodeId: string): boolean {
    return this.validation?.diagnostics.some((diagnostic) => (
      diagnostic.node_id === nodeId
      && (diagnostic.severity === 'error' || diagnostic.severity === 'compatibility')
    )) ?? false
  }

  hasEdgeError(edgeId: string): boolean {
    return this.validation?.diagnostics.some((diagnostic) => (
      diagnostic.edge_id === edgeId
      && (diagnostic.severity === 'error' || diagnostic.severity === 'compatibility')
    )) ?? false
  }

  private connectionDiagnosticMessage(diagnostics: ValidationDiagnostic[]): string {
    const diagnostic = diagnostics.find((item) => item.severity === 'error' || item.severity === 'compatibility')
      ?? diagnostics[0]
    return diagnostic?.message ?? 'The server rejected this connection.'
  }

  portEnabled(node: EditorNode, direction: 'input' | 'output', portId: string): boolean {
    const ports = direction === 'input' ? node.data.manifest.inputs : node.data.manifest.outputs
    const port = ports.find((item) => item.id === portId)
    return port ? this.conditionMatches(port.enabled_when ?? {}, node.data.parameters) : false
  }

  private conditionMatches(condition: Record<string, unknown>, parameters: Record<string, unknown>): boolean {
    return Object.entries(condition).every(([parameterId, expected]) => {
      const actual = parameters[parameterId]
      if (Array.isArray(expected)) return expected.includes(actual)
      if (expected && typeof expected === 'object') {
        const details = expected as Record<string, unknown>
        if ('equals' in details && actual !== details['equals']) return false
        if ('not_equals' in details && actual === details['not_equals']) return false
        if ('in' in details && Array.isArray(details['in']) && !details['in'].includes(actual)) return false
        if ('not_in' in details && Array.isArray(details['not_in']) && details['not_in'].includes(actual)) return false
      } else if (actual !== expected) return false
      return true
    })
  }

  toggleViewportLock(): void { this.viewportLocked.update((locked) => !locked) }
  zoomIn(): void { if (!this.viewportLocked()) this.zoom()?.zoomIn() }
  zoomOut(): void { if (!this.viewportLocked()) this.zoom()?.zoomOut() }
  resetView(): void { if (!this.viewportLocked()) this.canvas()?.resetScaleAndCenter(true) }
  fitView(): void { if (!this.viewportLocked()) this.canvas()?.fitToScreen(graphFitPadding(this.store.nodes(), 0.1, GRAPH_NODE_SIZE), true) }
  fitOnNodesRendered(): void {
    if (this.viewportLocked() || this.graphFitted || !this.store.nodes().length) return
    this.canvas()?.fitToScreen(graphFitPadding(this.store.nodes(), this.readOnly ? 0.18 : 0.1, GRAPH_NODE_SIZE), false)
    this.graphFitted = true
  }

  @HostListener('document:keydown', ['$event'])
  handleKeyboard(event: KeyboardEvent): void {
    if (event.key === 'Escape' && (this.canvasMenu() || this.nodeMenu())) { event.preventDefault(); this.closeMenus(); return }
    const command = event.ctrlKey || event.metaKey
    const target = event.target
    const editing = target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement
    if (command && event.key.toLowerCase() === 'z') {
      event.preventDefault()
      if (event.shiftKey) this.store.redo()
      else this.store.undo()
      return
    }
    if (command && event.key.toLowerCase() === 'y') { event.preventDefault(); this.store.redo(); return }
    if (command && event.key.toLowerCase() === 'c' && !editing) { this.store.copySelected(); return }
    if (command && event.key.toLowerCase() === 'v' && !editing) { event.preventDefault(); this.store.pasteCopied(); return }
    if (command && event.key.toLowerCase() === 'l' && !editing) { event.preventDefault(); this.store.autoLayout(this.store.nodes(), this.store.edges()); if (!this.viewportLocked()) this.fitView(); return }
    if (!editing && (event.key === 'Delete' || event.key === 'Backspace')) { event.preventDefault(); this.store.deleteSelected() }
  }

  @HostListener('document:pointerdown', ['$event'])
  closeMenusOnOutsidePointer(event: PointerEvent): void {
    const target = event.target instanceof Element ? event.target : null
    if (target?.closest('.canvas-context-menu')) return
    this.closeMenus()
  }
}
