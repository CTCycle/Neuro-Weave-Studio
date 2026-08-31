import { Component, inject, Input } from '@angular/core'
import { FFlowModule, type FCreateConnectionEvent, type FDeleteSelectedEvent, type FMoveNodesEvent, type FSelectionChangeEvent } from '@foblex/flow'
import type { EditorNode } from '../../../domain/editor.types'
import type { NodeManifest } from '../../../domain/types'
import { GraphEditorStore } from '../state/graph-editor.store'

@Component({
  selector: 'nw-graph-canvas',
  standalone: true,
  imports: [FFlowModule],
  template: `
    <div class="canvas-shell" (dragover)="allowDrop($event)" (drop)="dropNode($event)">
      <f-flow fDraggable [fDraggableDisabled]="readOnly" [vCellSize]="16" [hCellSize]="16" [fCellSizeWhileDragging]="16" (fDragStarted)="store.beginDrag()" (fDragEnded)="store.endDrag()" (fMoveNodes)="moveNodes($event)" (fSelectionChange)="select($event)" (fDeleteSelected)="deleteSelected($event)" (fCreateConnection)="createConnection($event)">
        <f-background><f-circle-pattern color="rgba(118, 150, 164, .16)" [radius]="1.2"></f-circle-pattern></f-background>
        <f-canvas fZoom [fZoomMinimum]="0.2" [fZoomMaximum]="2.5" [fZoomStep]="0.1">
          <f-connection-for-create></f-connection-for-create>
          @for (edge of store.edges(); track edge.id) { <f-connection [fConnectionId]="edge.id" [fSourceId]="connectorId(edge.source, 'output', edge.sourceHandle || 'output')" [fTargetId]="connectorId(edge.target, 'input', edge.targetHandle || 'input')"></f-connection> }
          @for (node of store.nodes(); track node.id) {
            <div fNode [fNodeId]="node.id" [fNodePosition]="node.position" [fNodeDraggingDisabled]="readOnly" [fNodeSize]="nodeSize(node)" [class.is-selected]="store.selectedNodeId() === node.id" (click)="store.selectNode(node.id)">
              <div fDragHandle class="nw-flow-node" [style.--node-accent]="node.data.manifest.ui.accent"><div class="nw-flow-node-heading"><span class="nw-flow-node-icon">{{ node.data.manifest.ui.icon || '·' }}</span><strong>{{ node.data.label }}</strong></div><small>{{ node.data.manifest.id }}</small><div class="nw-flow-node-ports"><span class="nw-flow-node-port-list">@for (port of node.data.manifest.inputs; track port.id) { <span class="nw-flow-port"><i fConnector fConnectorType="target" [fConnectorId]="connectorId(node.id, 'input', port.id)" [fConnectorMultiple]="port.accepts_multiple"></i>{{ port.display_name }}</span> }</span><span class="nw-flow-node-port-list output">@for (port of node.data.manifest.outputs; track port.id) { <span class="nw-flow-port"><i fConnector fConnectorType="source" [fConnectorId]="connectorId(node.id, 'output', port.id)" [fConnectorMultiple]="port.accepts_multiple"></i>{{ port.display_name }}</span> }</span></div></div>
            </div>
          }
        </f-canvas>
      </f-flow>
      @if (!store.nodes().length) { <div class="canvas-empty-state"><span class="eyebrow">EMPTY GRAPH</span><strong>Drop an operation here</strong><span>Choose a node from the catalog to start a model.</span></div> }
      <div class="canvas-contract-note">Foblex view adapter · 16 px snap · zoom 20–250%</div>
    </div>
  `,
})
export class GraphCanvasComponent {
  @Input() readOnly = false
  readonly store = inject(GraphEditorStore)

  allowDrop(event: DragEvent): void { if (!this.readOnly) event.preventDefault() }
  dropNode(event: DragEvent): void {
    if (this.readOnly) return
    event.preventDefault()
    const payload = event.dataTransfer?.getData('application/neuro-weave-node')
    if (!payload) return
    try {
      const value = JSON.parse(payload) as { id: string; version: number }
      const manifest = this.store.manifests().find((item) => item.id === value.id && item.version === value.version)
      if (manifest) this.store.addNode(manifest, { x: event.offsetX, y: event.offsetY })
    } catch { /* The drop payload is external UI data; ignore malformed items. */ }
  }
  moveNodes(event: FMoveNodesEvent): void { this.store.moveNodes(event.nodes.map((node) => ({ id: node.id, position: { x: node.position.x, y: node.position.y } }))) }
  select(event: FSelectionChangeEvent): void { this.store.selectNode(event.nodeIds[0] ?? null); this.store.selectEdge(event.connectionIds[0] ?? null) }
  deleteSelected(_event: FDeleteSelectedEvent): void { if (!this.readOnly) this.store.deleteSelected() }
  createConnection(event: FCreateConnectionEvent): void { if (!this.readOnly && event.targetId) { const source = parseConnector(event.sourceId); const target = parseConnector(event.targetId); if (source?.direction === 'output' && target?.direction === 'input') this.store.connect({ source: source.nodeId, sourceHandle: source.portId, target: target.nodeId, targetHandle: target.portId }) } }
  connectorId(nodeId: string, direction: 'input' | 'output', portId: string): string { return `${nodeId}|${direction}|${portId}` }
  nodeSize(node: EditorNode): { width: number; height: number } { return { width: Math.max(196, node.data.manifest.ui.width || 196), height: Math.max(96, node.data.manifest.ui.height || 96) } }
}

interface ParsedConnector { nodeId: string; direction: 'input' | 'output'; portId: string }

function parseConnector(value: string): ParsedConnector | null {
  const [nodeId, direction, ...port] = value.split('|')
  if ((direction !== 'input' && direction !== 'output') || !nodeId || !port.length) return null
  return { nodeId, direction, portId: port.join('|') }
}
