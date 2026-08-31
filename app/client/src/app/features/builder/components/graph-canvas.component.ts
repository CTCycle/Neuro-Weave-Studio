import { Component, EventEmitter, HostListener, inject, Input, Output, signal, viewChild } from '@angular/core'
import { FCanvasComponent, FFlowModule, FZoomDirective, type FCreateConnectionEvent, type FDeleteSelectedEvent, type FMoveNodesEvent, type FSelectionChangeEvent } from '@foblex/flow'
import type { EditorNode } from '../../../domain/editor.types'
import { GraphEditorStore } from '../state/graph-editor.store'
import { connectorId, flowPositionFromDrop, normalizeConnection, portOffset } from '../graph/foblex-graph.adapter'

@Component({
  selector: 'nw-graph-canvas',
  standalone: true,
  imports: [FFlowModule],
  template: `
    <div class="canvas-shell" (dragover)="allowDrop($event)" (drop)="dropNode($event)">
      <f-flow fDraggable [fDraggableDisabled]="readOnly" [vCellSize]="16" [hCellSize]="16" [fCellSizeWhileDragging]="16" (fDragStarted)="store.beginDrag()" (fDragEnded)="store.endDrag()" (fMoveNodes)="moveNodes($event)" (fSelectionChange)="select($event)" (fDeleteSelected)="deleteSelected($event)" (fCreateConnection)="createConnection($event)">
        <f-background><f-circle-pattern color="rgba(118, 150, 164, .16)" [radius]="1.2"></f-circle-pattern></f-background>
        @if (minimapVisible()) { <f-minimap [fMinSize]="50" [fNodeRenderLimit]="1000"></f-minimap> }
        <f-canvas fZoom [fZoomMinimum]="0.2" [fZoomMaximum]="2.5" [fZoomStep]="0.1">
          <f-connection-for-create></f-connection-for-create>
          @for (edge of store.edges(); track edge.id) { <f-connection [fConnectionId]="edge.id" [fSourceId]="connectorId(edge.source, 'output', edge.sourceHandle || 'output')" [fTargetId]="connectorId(edge.target, 'input', edge.targetHandle || 'input')"></f-connection> }
          @for (node of store.nodes(); track node.id) {
            <div fNode [fNodeId]="node.id" [fNodePosition]="node.position" [fNodeDraggingDisabled]="readOnly" [fNodeSelectionDisabled]="readOnly" [fNodeSize]="nodeSize(node)" [class.is-selected]="store.selectedNodeId() === node.id" [attr.aria-label]="node.data.label" (click)="selectNode(node.id)">
              <div fDragHandle class="nw-flow-node" [style.--node-accent]="node.data.manifest.ui.accent"><div class="nw-flow-node-heading"><span class="nw-flow-node-icon">{{ node.data.manifest.ui.icon || '·' }}</span><strong>{{ node.data.label }}</strong></div><small>{{ node.data.manifest.id }}</small><div class="nw-flow-port-layer inputs">@for (port of node.data.manifest.inputs; track port.id; let index = $index) { <span class="nw-flow-port input" [style.top]="portOffset(index, node.data.manifest.inputs.length) + '%'" [title]="port.display_name + ' · ' + port.kind"><i fConnector fConnectorType="target" [fConnectorId]="connectorId(node.id, 'input', port.id)" [fConnectorMultiple]="port.accepts_multiple"></i><span>{{ port.display_name }}</span></span> }</div><div class="nw-flow-port-layer outputs">@for (port of node.data.manifest.outputs; track port.id; let index = $index) { <span class="nw-flow-port output" [style.top]="portOffset(index, node.data.manifest.outputs.length) + '%'" [title]="port.display_name + ' · ' + port.kind"><i fConnector fConnectorType="source" [fConnectorId]="connectorId(node.id, 'output', port.id)" [fConnectorMultiple]="port.accepts_multiple"></i><span>{{ port.display_name }}</span></span> }</div></div>
            </div>
          }
        </f-canvas>
        <div class="canvas-editor-tools" [class.is-focus-mode]="fullCanvas" role="toolbar" aria-label="Graph editor controls"><div class="canvas-zoom-controls"><button type="button" class="canvas-tool" (click)="zoomOut()" aria-label="Zoom out" title="Zoom out">−</button><button type="button" class="canvas-tool" (click)="resetView()" aria-label="Reset view" title="Reset view">Reset</button><button type="button" class="canvas-tool" (click)="zoomIn()" aria-label="Zoom in" title="Zoom in">+</button><button type="button" class="canvas-tool" (click)="fitView()" aria-label="Fit graph to view" title="Fit graph to view">Fit</button></div><button type="button" class="canvas-tool" (click)="minimapVisible.set(!minimapVisible())" [attr.aria-label]="minimapVisible() ? 'Hide minimap' : 'Show minimap'" [title]="minimapVisible() ? 'Hide minimap' : 'Show minimap'">{{ minimapVisible() ? 'Hide map' : 'Show map' }}</button>@if (fullCanvas) { <div class="canvas-focus-bar"><span class="canvas-focus-model">{{ modelName }}</span><span class="canvas-focus-draft" [class.is-dirty]="dirty">{{ busy ? 'Saving…' : dirty ? 'Draft' : 'Saved' }}</span><button type="button" class="canvas-tool" (click)="exitFocus.emit()" [disabled]="busy" aria-label="Exit full canvas">Exit focus</button></div> } @else { <button type="button" class="canvas-tool" (click)="enterFocus.emit()" aria-label="Enter full canvas">Focus</button> }</div>
      </f-flow>
      @if (!store.nodes().length) { <div class="canvas-empty-state"><span class="eyebrow">EMPTY GRAPH</span><strong>Drop an operation here</strong><span>Choose a node from the catalog to start a model.</span></div> }
      <div class="canvas-contract-note">Foblex view adapter · 16 px snap · zoom 20–250%</div>
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
  readonly store = inject(GraphEditorStore)
  readonly minimapVisible = signal(true)
  private readonly canvas = viewChild(FCanvasComponent)
  private readonly zoom = viewChild(FZoomDirective)

  allowDrop(event: DragEvent): void { if (!this.readOnly) event.preventDefault() }
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
  select(event: FSelectionChangeEvent): void { if (this.readOnly) return; if (event.nodeIds.length) this.store.selectNode(event.nodeIds[0]); else if (event.connectionIds.length) this.store.selectEdge(event.connectionIds[0]); else this.store.clearSelection() }
  deleteSelected(_event: FDeleteSelectedEvent): void { if (!this.readOnly) this.store.deleteSelected() }
  createConnection(event: FCreateConnectionEvent): void { if (!this.readOnly && event.targetId) { const connection = normalizeConnection(event.sourceId, event.targetId); if (connection) this.store.connect(connection) } }
  connectorId = connectorId
  portOffset = portOffset
  nodeSize(node: EditorNode): { width: number; height: number } { return { width: Math.max(196, node.data.manifest.ui.width || 196), height: Math.max(96, node.data.manifest.ui.height || 96) } }

  zoomIn(): void { this.zoom()?.zoomIn() }
  zoomOut(): void { this.zoom()?.zoomOut() }
  resetView(): void { this.canvas()?.resetScaleAndCenter(true) }
  fitView(): void { this.canvas()?.fitToScreen({ x: 0.2, y: 0.2 }, true) }

  @HostListener('document:keydown', ['$event'])
  handleKeyboard(event: KeyboardEvent): void {
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
    if (command && event.key.toLowerCase() === 'l' && !editing) { event.preventDefault(); this.store.autoLayout(this.store.nodes(), this.store.edges()); this.fitView(); return }
    if (!editing && (event.key === 'Delete' || event.key === 'Backspace')) { event.preventDefault(); this.store.deleteSelected() }
  }
}
