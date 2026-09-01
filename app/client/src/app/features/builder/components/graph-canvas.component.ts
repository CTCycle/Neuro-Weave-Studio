import { Component, EventEmitter, HostListener, effect, inject, Input, Output, signal, viewChild } from '@angular/core'
import { LucideLock, LucideMaximize2, LucideSave, LucideUnlock } from '@lucide/angular'
import { FCanvasComponent, FFlowModule, FZoomDirective, type FCreateConnectionEvent, type FDeleteSelectedEvent, type FEventTrigger, type FMoveNodesEvent, type FSelectionChangeEvent } from '@foblex/flow'
import type { EditorNode } from '../../../domain/editor.types'
import { GraphEditorStore } from '../state/graph-editor.store'
import { GRAPH_NODE_SIZE, connectorId, flowPositionFromDrop, graphFitPadding, normalizeConnection, portOffset } from '../graph/foblex-graph.adapter'

@Component({
  selector: 'nw-graph-canvas',
  standalone: true,
  imports: [FFlowModule, LucideLock, LucideMaximize2, LucideSave, LucideUnlock],
  template: `
    <div class="canvas-shell" [class.is-read-only]="readOnly" [class.is-viewport-locked]="viewportLocked()" (contextmenu)="openCanvasMenu($event)" (dragover)="allowDrop($event)" (drop)="dropNode($event)">
      <f-flow fDraggable [fDraggableDisabled]="readOnly" [fCanvasMoveTrigger]="canvasMoveTrigger" [vCellSize]="16" [hCellSize]="16" [fCellSizeWhileDragging]="16" (fDragStarted)="store.beginDrag()" (fDragEnded)="store.endDrag()" (fMoveNodes)="moveNodes($event)" (fSelectionChange)="select($event)" (fDeleteSelected)="deleteSelected($event)" (fCreateConnection)="createConnection($event)" (fNodesRendered)="fitOnNodesRendered()">
        <f-background><f-circle-pattern color="rgba(118, 150, 164, .16)" [radius]="24"></f-circle-pattern></f-background>
        @if (minimapVisible()) { <f-minimap [fMinSize]="50" [fNodeRenderLimit]="1000"></f-minimap> }
        <f-canvas fZoom [fZoom]="!viewportLocked()" [fZoomMinimum]="0.2" [fZoomMaximum]="2.5" [fZoomStep]="0.1">
          <f-connection-for-create fType="segment" fOffset="20"></f-connection-for-create>
          <ng-container ngProjectAs="[fConnections]">@for (edge of store.edges(); track edge.id) { <f-connection fType="segment" fOffset="20" [fSelectionDisabled]="readOnly" [fConnectionId]="edge.id" [fSourceId]="connectorId(edge.source, 'output', edge.sourceHandle || 'output')" [fTargetId]="connectorId(edge.target, 'input', edge.targetHandle || 'input')" [class.is-selected]="store.selectedEdgeId() === edge.id"></f-connection> }</ng-container>
          <ng-container ngProjectAs="[fNodes]">@for (node of store.nodes(); track node.id) {
            <div fNode [fNodeId]="node.id" [fNodePosition]="node.position" [fNodeDraggingDisabled]="readOnly || node.locked" [fNodeSelectionDisabled]="readOnly" [fNodeSize]="nodeSize(node)" [class.is-selected]="store.selectedNodeId() === node.id" [attr.aria-label]="node.data.label" (click)="selectNode(node.id)">
              <div fDragHandle class="nw-flow-node" [style.--node-accent]="node.data.manifest.ui.accent" (contextmenu)="openNodeMenu($event, node.id)"><div class="nw-flow-node-status" aria-hidden="true">@if (node.locked) { <svg lucideLock [size]="13"></svg> }</div>@if (store.isEffectiveOutput(node.id)) { <span class="nw-flow-output-indicator" [attr.aria-label]="store.isOutput(node.id) ? 'Explicit workflow output' : 'Inferred workflow output'" [title]="store.isOutput(node.id) ? 'Explicit workflow output' : 'Inferred workflow output'"></span> }<div class="nw-flow-node-eyebrow">{{ node.data.manifest.category }}</div><div class="nw-flow-node-heading"><strong>{{ node.data.label }}</strong></div><small>{{ node.data.manifest.id }}</small><div class="nw-flow-node-footer">{{ shapeLabel(node) }}</div><div class="nw-flow-port-layer inputs">@for (port of node.data.manifest.inputs; track port.id; let index = $index) { @if (portEnabled(node, 'input', port.id)) { <span class="nw-flow-port input" [style.top]="portOffset(index, node.data.manifest.inputs.length) + '%'" [title]="port.display_name + ' · ' + port.kind"><i fConnector fConnectorType="target" [fConnectorId]="connectorId(node.id, 'input', port.id)" [fConnectorMultiple]="port.accepts_multiple"></i></span> } }</div><div class="nw-flow-port-layer outputs">@for (port of node.data.manifest.outputs; track port.id; let index = $index) { @if (portEnabled(node, 'output', port.id)) { <span class="nw-flow-port output" [style.top]="portOffset(index, node.data.manifest.outputs.length) + '%'" [title]="port.display_name + ' · ' + port.kind"><i fConnector fConnectorType="source" [fConnectorId]="connectorId(node.id, 'output', port.id)" [fConnectorMultiple]="port.accepts_multiple"></i></span> } }</div></div>
            </div>
          } </ng-container>
        </f-canvas>
        <div class="flow-controls canvas-flow-controls" role="group" aria-label="Canvas zoom controls"><button type="button" class="canvas-tool" (click)="zoomIn()" [disabled]="viewportLocked()" aria-label="Zoom in" [title]="viewportLocked() ? 'Unlock canvas to zoom in' : 'Zoom in'">+</button><button type="button" class="canvas-tool" (click)="zoomOut()" [disabled]="viewportLocked()" aria-label="Zoom out" [title]="viewportLocked() ? 'Unlock canvas to zoom out' : 'Zoom out'">−</button><button type="button" class="canvas-tool" (click)="resetView()" [disabled]="viewportLocked()" aria-label="Reset view" [title]="viewportLocked() ? 'Unlock canvas to reset the view' : 'Reset view'"><svg lucideMaximize2 [size]="13"></svg></button></div>
        @if (!readOnly) { <div class="canvas-editor-tools" [class.is-focus-mode]="fullCanvas" role="toolbar" aria-label="Graph editor controls"><div class="canvas-tool-group"><button type="button" class="canvas-tool" (click)="minimapVisible.set(!minimapVisible())" [attr.aria-label]="minimapVisible() ? 'Hide minimap' : 'Show minimap'" [title]="minimapVisible() ? 'Hide minimap' : 'Show minimap'">{{ minimapVisible() ? 'Hide map' : 'Show map' }}</button><button type="button" class="canvas-tool canvas-lock-tool" [class.is-active]="viewportLocked()" [attr.aria-pressed]="viewportLocked()" [attr.aria-label]="viewportLocked() ? 'Unlock canvas' : 'Lock canvas'" [title]="viewportLocked() ? 'Unlock canvas' : 'Lock canvas'" (click)="toggleViewportLock()">@if (viewportLocked()) { <svg lucideUnlock [size]="13"></svg><span>Unlock canvas</span> } @else { <svg lucideLock [size]="13"></svg><span>Lock canvas</span> }</button></div>@if (fullCanvas) { <div class="canvas-focus-bar"><span class="canvas-focus-model">{{ modelName }}</span><span class="canvas-focus-draft" [class.is-dirty]="dirty">{{ busy ? 'Saving…' : dirty ? 'Draft' : 'Saved' }}</span><button type="button" class="canvas-tool canvas-focus-save" (click)="save.emit()" [disabled]="busy" aria-label="Save model"><svg lucideSave [size]="14"></svg><span>Save</span></button><button type="button" class="canvas-tool canvas-focus-exit" (click)="exitFocus.emit()" [disabled]="busy" aria-label="Exit full canvas">Exit full canvas</button></div> } @else { <button type="button" class="canvas-tool" (click)="enterFocus.emit()" aria-label="Enter full canvas">Focus</button> }</div> }
      </f-flow>
      @if (!readOnly && !store.nodes().length) { <div class="canvas-empty-state"><span class="eyebrow">EMPTY GRAPH</span><strong>Drop an operation here</strong><span>Choose a node from the catalog to start a model.</span></div> }
      @if (readOnly && !store.nodes().length) { <div class="preview-empty">No renderable nodes in this revision.</div> }
      @if (!readOnly) { <div class="canvas-hint">Drop a node to build the model • Ctrl+L auto-layout</div> }
      @if (!readOnly && canvasMenu(); as menu) { <div class="canvas-context-menu" role="menu" [style.left.px]="menu.x" [style.top.px]="menu.y" (pointerdown)="$event.stopPropagation()"><button type="button" role="menuitem" (click)="fitView(); closeMenus()">Fit workflow to view</button><button type="button" role="menuitem" (click)="autoLayout(); closeMenus()">Auto layout</button><button type="button" role="menuitem" class="is-destructive" (click)="requestReset(); closeMenus()">Reset canvas</button></div> }
      @if (!readOnly && nodeMenu(); as menu) { <div class="canvas-context-menu node-context-menu" role="menu" [style.left.px]="menu.x" [style.top.px]="menu.y" (pointerdown)="$event.stopPropagation()"><button type="button" role="menuitem" (click)="duplicateWithDefaults(menu.nodeId); closeMenus()">Duplicate with defaults</button><button type="button" role="menuitem" (click)="duplicateExact(menu.nodeId); closeMenus()">Duplicate exact</button><button type="button" role="menuitem" (click)="copyNode(menu.nodeId); closeMenus()">Copy node</button><button type="button" role="menuitem" (click)="resetParameters(menu.nodeId); closeMenus()">Reset parameters</button><button type="button" role="menuitem" (click)="disconnectNode(menu.nodeId); closeMenus()">Disconnect all</button><button type="button" role="menuitem" [disabled]="isInput(menu.nodeId)" [title]="isInput(menu.nodeId) ? 'Input nodes cannot be workflow outputs' : ''" (click)="toggleOutput(menu.nodeId); closeMenus()">{{ store.isOutput(menu.nodeId) ? 'Remove from outputs' : 'Mark as output' }}</button><button type="button" role="menuitem" (click)="toggleLock(menu.nodeId); closeMenus()">{{ isLocked(menu.nodeId) ? 'Unlock node' : 'Lock node' }}</button><button type="button" role="menuitem" class="is-destructive" (click)="deleteNode(menu.nodeId); closeMenus()">Delete node</button></div> }
    </div>
  `,
})
export class GraphCanvasComponent {
  @Input() readOnly = false
  @Input() modelName = ''
  @Input() dirty = false
  @Input() busy = false
  @Input() fullCanvas = false
  @Output() readonly enterFocus = new EventEmitter<void>()
  @Output() readonly exitFocus = new EventEmitter<void>()
  @Output() readonly save = new EventEmitter<void>()
  @Output() readonly resetRequested = new EventEmitter<void>()
  readonly store = inject(GraphEditorStore)
  readonly minimapVisible = signal(true)
  readonly viewportLocked = signal(false)
  readonly canvasMoveTrigger: FEventTrigger = () => !this.viewportLocked()
  private readonly canvas = viewChild(FCanvasComponent)
  private readonly zoom = viewChild(FZoomDirective)
  private graphFitted = false
  readonly canvasMenu = signal<{ x: number; y: number } | null>(null)
  readonly nodeMenu = signal<{ x: number; y: number; nodeId: string } | null>(null)

  constructor() {
    effect(() => {
      if (!this.store.nodes().length) this.graphFitted = false
    })
  }

  allowDrop(event: DragEvent): void { if (!this.readOnly) event.preventDefault() }
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
  toggleOutput(nodeId: string): void { this.store.toggleNodeOutput(nodeId) }
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
  createConnection(event: FCreateConnectionEvent): void { if (!this.readOnly && event.targetId) { const connection = normalizeConnection(event.sourceId, event.targetId); if (connection) this.store.connect(connection) } }
  connectorId = connectorId
  portOffset = portOffset
  shapeLabel(node: EditorNode): string {
    const output = node.data.shapeSummary ? Object.values(node.data.shapeSummary.outputs)[0] : undefined
    const shape = output?.shape.map((value) => value === null ? '?' : typeof value === 'object' ? value.name : value).join(' × ')
    return shape || 'Shape pending'
  }
  nodeSize(_node: EditorNode): { width: number; height: number } { return GRAPH_NODE_SIZE }

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
