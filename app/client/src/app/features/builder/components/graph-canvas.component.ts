import { Component, EventEmitter, HostListener, computed, effect, inject, Input, Output, signal, viewChild } from '@angular/core'
import type { OnChanges, SimpleChanges } from '@angular/core'
import { LucideLock, LucideMaximize2, LucideSave, LucideUnlock } from '@lucide/angular'
import { FCanvasComponent, FFlowModule, FZoomDirective, type FCreateConnectionEvent, type FDeleteSelectedEvent, type FEventTrigger, type FMoveNodesEvent, type FSelectionChangeEvent } from '@foblex/flow'
import type { EditorNode } from '../../../domain/editor.types'
import { ApiService } from '../../../core/api/api.service'
import type { ExecutionBackend, ValidationDiagnostic, ValidationResult } from '../../../domain/types'
import { GraphEditorStore } from '../state/graph-editor.store'
import { TensorInspectionCoordinator } from '../../tensor-inspection/tensor-inspection.coordinator'
import { connectionCandidates } from '../graph/graph-validation'
import { GRAPH_NODE_SIZE, connectorId, flowPositionFromDrop, graphFitPadding, normalizeConnection, portOffset } from '../graph/foblex-graph.adapter'

const NO_CONNECTION_TARGET_ID = '__neuro_weave_no_compatible_target__'
type KeyboardPort = { nodeId: string; direction: 'input' | 'output'; portId: string }
type NormalizedConnection = NonNullable<ReturnType<typeof normalizeConnection>>

@Component({
  selector: 'nw-graph-canvas',
  standalone: true,
  imports: [FFlowModule, LucideLock, LucideMaximize2, LucideSave, LucideUnlock],
  template: `
    <div class="canvas-shell" role="region" aria-label="Model graph canvas" [class.is-read-only]="readOnly" [class.is-viewport-locked]="viewportLocked()" [class.has-connection-feedback]="!!activeConnectionFeedback()" (contextmenu)="openCanvasMenu($event)" (dragover)="allowDrop($event)" (drop)="dropNode($event)">
      <f-flow fDraggable [fDraggableDisabled]="readOnly" [fCanvasMoveTrigger]="canvasMoveTrigger" [vCellSize]="16" [hCellSize]="16" [fCellSizeWhileDragging]="16" (fDragStarted)="startDrag()" (fDragEnded)="store.endDrag()" (fMoveNodes)="moveNodes($event)" (fSelectionChange)="select($event)" (fDeleteSelected)="deleteSelected($event)" (fCreateConnection)="createConnection($event)" (fNodesRendered)="fitOnNodesRendered()">
        <f-background><f-circle-pattern color="rgba(118, 150, 164, .16)" [radius]="24"></f-circle-pattern></f-background>
        @if (minimapVisible()) { <f-minimap [fMinSize]="50" [fNodeRenderLimit]="1000"></f-minimap> }
        <f-canvas fZoom [fZoom]="!viewportLocked()" [fZoomMinimum]="0.2" [fZoomMaximum]="2.5" [fZoomStep]="0.1">
          <f-connection-for-create fType="straight" fOffset="20"></f-connection-for-create>
          <ng-container ngProjectAs="[fConnections]">@for (edge of store.edges(); track edge.id) { <f-connection fType="straight" fOffset="20" [fSelectionDisabled]="readOnly" [fConnectionId]="edge.id" [fSourceId]="connectorId(edge.source, 'output', edge.sourceHandle || 'output')" [fTargetId]="connectorId(edge.target, 'input', edge.targetHandle || 'input')" [class.is-selected]="store.selectedEdgeId() === edge.id" [class.is-invalid]="hasEdgeError(edge.id)"></f-connection> }</ng-container>
          <ng-container ngProjectAs="[fNodes]">@for (node of store.nodes(); track node.id) {
            <div fNode [fNodeId]="node.id" [fNodePosition]="node.position" [fNodeDraggingDisabled]="readOnly || node.locked" [fNodeSelectionDisabled]="readOnly" [fNodeSize]="nodeSize(node)" [class.is-selected]="store.selectedNodeId() === node.id" [class.has-validation-error]="hasNodeError(node.id)" [class.is-profile-incompatible]="!isProfileCompatible(node.id)" [attr.data-node-id]="node.id" [attr.tabindex]="0" role="group" [attr.aria-label]="nodeAccessibilityLabel(node)" [title]="nodeCompatibilityTitle(node.id)" (click)="selectNode(node.id)" (keydown)="handleNodeKeydown($event, node.id)">
              <div fDragHandle class="nw-flow-node" [style.--node-accent]="node.data.manifest.ui.accent" (contextmenu)="openNodeMenu($event, node.id)"><div class="nw-flow-node-status" [attr.aria-label]="!isProfileCompatible(node.id) ? nodeCompatibilityTitle(node.id) : null">@if (node.locked) { <svg lucideLock [size]="13"></svg> } @if (!isProfileCompatible(node.id)) { <span class="nw-flow-node-incompatible" aria-hidden="true">!</span> }</div>@if (store.isEffectiveOutput(node.id)) { <span class="nw-flow-output-indicator" [attr.aria-label]="store.isOutput(node.id) ? 'Explicit workflow output' : 'Inferred workflow output'" [title]="store.isOutput(node.id) ? 'Explicit workflow output' : 'Inferred workflow output'"></span> }<div class="nw-flow-node-eyebrow">{{ node.data.manifest.category }}</div><div class="nw-flow-node-heading"><strong>{{ node.data.label }}</strong></div><small>{{ node.data.manifest.id }}</small><div class="nw-flow-node-footer">{{ shapeLabel(node) }}</div><div class="nw-flow-port-layer inputs">@for (port of node.data.manifest.inputs; track port.id; let index = $index) { @if (portEnabled(node, 'input', port.id)) { <span class="nw-flow-port input" [class.is-profile-incompatible]="!isProfileCompatible(node.id)" [style.top]="portOffset(index, node.data.manifest.inputs.length) + '%'" [attr.data-port-id]="port.id" [attr.data-port-direction]="'input'" [attr.tabindex]="isProfileCompatible(node.id) ? 0 : -1" [attr.aria-disabled]="!isProfileCompatible(node.id)" role="button" [attr.aria-label]="keyboardPortLabel(node, 'input', port.id)" [class.is-keyboard-target]="isKeyboardTarget(node.id, port.id)" [title]="portTitle(node.id, port.display_name, port.kind)" (keydown)="handlePortKeydown($event, node.id, 'input', port.id)"><i fConnector fConnectorType="target" [fConnectorId]="connectorId(node.id, 'input', port.id)" [fConnectorMultiple]="port.accepts_multiple"></i></span> } }</div><div class="nw-flow-port-layer outputs">@for (port of node.data.manifest.outputs; track port.id; let index = $index) { @if (portEnabled(node, 'output', port.id)) { <span class="nw-flow-port output" [class.is-profile-incompatible]="!isProfileCompatible(node.id)" [style.top]="portOffset(index, node.data.manifest.outputs.length) + '%'" [attr.data-port-id]="port.id" [attr.data-port-direction]="'output'" [attr.tabindex]="isProfileCompatible(node.id) ? 0 : -1" [attr.aria-disabled]="!isProfileCompatible(node.id)" role="button" [attr.aria-pressed]="isKeyboardSource(node.id, port.id)" [attr.aria-label]="keyboardPortLabel(node, 'output', port.id)" [class.is-keyboard-source]="isKeyboardSource(node.id, port.id)" [title]="portTitle(node.id, port.display_name, port.kind)" (keydown)="handlePortKeydown($event, node.id, 'output', port.id)"><i fConnector fConnectorType="source" [fConnectorId]="connectorId(node.id, 'output', port.id)" [fConnectorMultiple]="true" [fCanBeConnectedTo]="connectionTargets(node.id, port.id)"></i></span> } }</div></div>
            </div>
          } </ng-container>
        </f-canvas>
        <div class="flow-controls canvas-flow-controls" role="group" aria-label="Canvas zoom controls"><button type="button" class="canvas-tool" (click)="zoomIn()" [disabled]="viewportLocked()" aria-label="Zoom in" [title]="viewportLocked() ? 'Unlock canvas to zoom in' : 'Zoom in'">+</button><button type="button" class="canvas-tool" (click)="zoomOut()" [disabled]="viewportLocked()" aria-label="Zoom out" [title]="viewportLocked() ? 'Unlock canvas to zoom out' : 'Zoom out'">−</button><button type="button" class="canvas-tool" (click)="resetView()" [disabled]="viewportLocked()" aria-label="Reset view" [title]="viewportLocked() ? 'Unlock canvas to reset the view' : 'Reset view'"><svg lucideMaximize2 [size]="13"></svg></button></div>
        @if (!readOnly) { <div class="canvas-editor-tools" [class.is-focus-mode]="fullCanvas" role="toolbar" aria-label="Graph editor controls"><div class="canvas-tool-group"><button type="button" class="canvas-tool" (click)="minimapVisible.set(!minimapVisible())" [attr.aria-label]="minimapVisible() ? 'Hide minimap' : 'Show minimap'" [title]="minimapVisible() ? 'Hide minimap' : 'Show minimap'">{{ minimapVisible() ? 'Hide map' : 'Show map' }}</button><button type="button" class="canvas-tool canvas-lock-tool" [class.is-active]="viewportLocked()" [attr.aria-pressed]="viewportLocked()" [attr.aria-label]="viewportLocked() ? 'Unlock canvas' : 'Lock canvas'" [title]="viewportLocked() ? 'Unlock canvas' : 'Lock canvas'" (click)="toggleViewportLock()">@if (viewportLocked()) { <svg lucideUnlock [size]="13"></svg><span>Unlock canvas</span> } @else { <svg lucideLock [size]="13"></svg><span>Lock canvas</span> }</button></div>@if (fullCanvas) { <div class="canvas-focus-bar"><span class="canvas-focus-model">{{ modelName }}</span><span class="canvas-focus-draft" [class.is-dirty]="dirty">{{ busy ? 'Saving…' : dirty ? 'Draft' : 'Saved' }}</span><button type="button" class="canvas-tool canvas-focus-save" (click)="save.emit()" [disabled]="busy" aria-label="Save model"><svg lucideSave [size]="14"></svg><span>Save</span></button><button type="button" class="canvas-tool canvas-focus-exit" (click)="exitFocus.emit()" [disabled]="busy" aria-label="Exit full canvas">Exit full canvas</button></div> } @else { <button type="button" class="canvas-tool" (click)="enterFocus.emit()" aria-label="Enter full canvas">Focus</button> }</div> }
      </f-flow>
      @if (!readOnly && activeConnectionFeedback(); as feedback) { <div class="connection-feedback" role="status" aria-live="assertive">{{ feedback }}</div> }
      @if (!readOnly && keyboardConnectionHint(); as hint) { <div class="connection-feedback keyboard-connection-feedback" role="status" aria-live="polite">{{ hint }}</div> }
      @if (!readOnly && !store.nodes().length) { <div class="canvas-empty-state"><span class="eyebrow">EMPTY GRAPH</span><strong>Drop an operation here</strong><span>Choose a node from the catalog to start a model.</span></div> }
      @if (readOnly && !store.nodes().length) { <div class="preview-empty">No renderable nodes in this revision.</div> }
      @if (!readOnly) { <div class="canvas-hint">Drop a node to build the model • Ctrl+L auto-layout</div> }
      @if (!readOnly && canvasMenu(); as menu) { <div class="canvas-context-menu" role="menu" [style.left.px]="menu.x" [style.top.px]="menu.y" (pointerdown)="$event.stopPropagation()"><button type="button" role="menuitem" [disabled]="!hasNodes()" [title]="hasNodes() ? 'Fit all nodes in the canvas' : 'Add a node before fitting the workflow'" (click)="fitView(); closeMenus()">Fit workflow to view</button><button type="button" role="menuitem" [disabled]="!hasNodes()" [title]="hasNodes() ? 'Arrange nodes using the workflow layout' : 'Add a node before applying a layout'" (click)="autoLayout(); closeMenus()">Auto layout</button><button type="button" role="menuitem" class="is-destructive" [disabled]="!hasNodes()" [title]="hasNodes() ? 'Remove all nodes and connections from the canvas' : 'The canvas is already empty'" (click)="requestReset(); closeMenus()">Reset canvas</button></div> }
      @if (nodeMenu(); as menu) { <div class="canvas-context-menu node-context-menu" role="menu" [style.left.px]="menu.x" [style.top.px]="menu.y" (pointerdown)="$event.stopPropagation()"><button type="button" role="menuitem" [disabled]="!canInspectTensor()" [title]="inspectionTitle()" (click)="inspectTensor(menu.nodeId); closeMenus()">Inspect tensor</button>@if (!readOnly) { <button type="button" role="menuitem" [disabled]="!canDuplicate(menu.nodeId)" [title]="duplicateTitle(menu.nodeId)" (click)="duplicateWithDefaults(menu.nodeId); closeMenus()">Duplicate with defaults</button><button type="button" role="menuitem" [disabled]="!canDuplicate(menu.nodeId)" [title]="duplicateTitle(menu.nodeId)" (click)="duplicateExact(menu.nodeId); closeMenus()">Duplicate exact</button><button type="button" role="menuitem" (click)="copyNode(menu.nodeId); closeMenus()">Copy node</button><button type="button" role="menuitem" [disabled]="!hasParameters(menu.nodeId)" [title]="hasParameters(menu.nodeId) ? 'Restore node parameters to their framework defaults' : 'This node has no parameters to reset'" (click)="resetParameters(menu.nodeId); closeMenus()">Reset parameters</button><button type="button" role="menuitem" [disabled]="!hasConnections(menu.nodeId)" [title]="hasConnections(menu.nodeId) ? 'Remove every connection attached to this node' : 'This node has no connections to disconnect'" (click)="disconnectNode(menu.nodeId); closeMenus()">Disconnect all</button><button type="button" role="menuitem" [disabled]="isOutputActionDisabled(menu.nodeId)" [title]="outputActionTitle(menu.nodeId)" (click)="toggleOutput(menu.nodeId); closeMenus()">{{ outputActionLabel(menu.nodeId) }}</button><button type="button" role="menuitem" (click)="toggleLock(menu.nodeId); closeMenus()">{{ isLocked(menu.nodeId) ? 'Unlock node' : 'Lock node' }}</button><button type="button" role="menuitem" class="is-destructive" (click)="deleteNode(menu.nodeId); closeMenus()">Delete node</button> }</div> }
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
  @Input() inspectionProjectId: string | null = null
  @Input() inspectionRevision: number | null = null
  @Input() inspectionEnabled = false
  @Input() inspectionDisabledReason = 'Save and validate this revision before inspecting.'
  @Output() readonly enterFocus = new EventEmitter<void>()
  @Output() readonly exitFocus = new EventEmitter<void>()
  @Output() readonly save = new EventEmitter<void>()
  @Output() readonly resetRequested = new EventEmitter<void>()
  readonly store = inject(GraphEditorStore)
  private readonly tensorInspection = inject(TensorInspectionCoordinator)
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
  readonly keyboardConnectionSource = signal<KeyboardPort | null>(null)
  readonly keyboardConnectionHint = computed(() => {
    const source = this.keyboardConnectionSource()
    if (!source) return null
    const node = this.store.nodes().find((item) => item.id === source.nodeId)
    const port = node?.data.manifest.outputs.find((item) => item.id === source.portId)
    if (!node || !port) return null
    return `Connection source ${port.display_name} on ${node.data.label} selected. Choose an input port and press Enter to connect, or press Escape to cancel.`
  })
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
      this.store.setExecutionProfile(this.executionBackend, this.kerasEnabled)
      void this.refreshConnectionTargets()
    }
  }

  private refreshConnectionTargets(): void {
    const nodes = this.store.nodes()
    const edges = this.store.edges()
    const targets = new Map<string, string[]>()

    for (const node of nodes) {
      for (const port of node.data.manifest.outputs) {
        const sourceKey = `${node.id}|${port.id}`
        const targetIds = connectionCandidates(
          nodes,
          edges,
          node.id,
          port.id,
          this.executionBackend,
          this.kerasEnabled,
        ).map((connection) => connectorId(connection.target, 'input', connection.targetHandle))
        targets.set(sourceKey, targetIds.length ? targetIds : this.noConnectionTargets)
      }
    }

    this.connectionTargetMap.set(targets)
  }

  private setConnectionFeedback(message: string | null): void {
    this.connectionFeedbackRevision.set(message === null ? null : this.store.graphRevision())
    this.connectionFeedback.set(message)
  }

  allowDrop(event: DragEvent): void { if (!this.readOnly) event.preventDefault() }
  startDrag(): void {
    this.keyboardConnectionSource.set(null)
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
    event.preventDefault()
    event.stopPropagation()
    if (!this.readOnly) this.store.selectNode(nodeId)
    this.canvasMenu.set(null)
    this.nodeMenu.set({ ...this.menuPosition(event), nodeId })
  }
  closeMenus(): void { this.canvasMenu.set(null); this.nodeMenu.set(null) }
  canInspectTensor(): boolean { return Boolean(this.inspectionEnabled && this.inspectionProjectId && this.inspectionRevision !== null) }
  inspectionTitle(): string { return this.canInspectTensor() ? 'Capture intermediate tensors for this saved revision' : this.inspectionDisabledReason }
  inspectTensor(nodeId: string): void {
    if (!this.canInspectTensor() || !this.inspectionProjectId || this.inspectionRevision === null) return
    void this.tensorInspection.start({ projectId: this.inspectionProjectId, revision: this.inspectionRevision, nodeId })
  }
  requestReset(): void { if (this.hasNodes()) this.resetRequested.emit() }
  autoLayout(): void { if (!this.hasNodes()) return; this.store.autoLayout(this.store.nodes(), this.store.edges()); if (!this.viewportLocked()) this.fitView() }
  duplicateWithDefaults(nodeId: string): void { this.store.duplicateWithDefaults(nodeId) }
  duplicateExact(nodeId: string): void { this.store.duplicateExact(nodeId) }
  copyNode(nodeId: string): void { this.store.selectNode(nodeId); this.store.copySelected() }
  resetParameters(nodeId: string): void { this.store.resetParameters(nodeId) }
  disconnectNode(nodeId: string): void { this.store.disconnectNode(nodeId) }
  toggleOutput(nodeId: string, portId?: string): void { if (portId || this.store.canToggleNodeOutput(nodeId)) this.store.toggleNodeOutput(nodeId, portId) }
  toggleLock(nodeId: string): void { this.store.toggleNodeLocked(nodeId) }
  deleteNode(nodeId: string): void { this.store.selectNode(nodeId); this.store.deleteSelected() }
  hasNodes(): boolean { return this.store.nodes().length > 0 }
  hasParameters(nodeId: string): boolean { return (this.store.nodes().find((node) => node.id === nodeId)?.data.manifest.parameters.length ?? 0) > 0 }
  hasConnections(nodeId: string): boolean { return this.store.edges().some((edge) => edge.source === nodeId || edge.target === nodeId) }
  isOutputActionDisabled(nodeId: string): boolean { return !this.store.canToggleNodeOutput(nodeId) }
  outputActionLabel(nodeId: string): string { return this.store.isOutput(nodeId) ? 'Remove from outputs' : 'Use as output' }
  outputActionTitle(nodeId: string): string {
    if (this.store.isOutput(nodeId)) return 'Remove this node from the explicit workflow outputs'
    if (this.isInput(nodeId)) return 'Input nodes cannot be workflow outputs'
    if (!this.store.hasEnabledOutputPort(nodeId)) return 'This node has no enabled value-producing output port'
    if (this.store.isInferredOutput(nodeId)) return 'This terminal node is already the inferred workflow output'
    return 'Use this node as a workflow output'
  }
  isLocked(nodeId: string): boolean { return this.store.nodes().find((node) => node.id === nodeId)?.locked ?? false }
  isInput(nodeId: string): boolean {
    return this.store.nodes().find((node) => node.id === nodeId)?.data.manifest.semantic_role === 'input'
  }
  canDuplicate(nodeId: string): boolean { return this.store.canDuplicate(nodeId) }
  duplicateTitle(nodeId: string): string {
    const node = this.store.nodes().find((item) => item.id === nodeId)
    if (!node) return 'The selected node is no longer available'
    if (this.store.canDuplicate(nodeId)) return 'Duplicate this compatible node'
    return `Unavailable: ${this.store.nodeCompatibility(node.data.manifest).reason ?? 'the stored parameters are not compatible with this profile'}`
  }
  isProfileCompatible(nodeId: string): boolean {
    const node = this.store.nodes().find((item) => item.id === nodeId)
    return Boolean(node && this.store.nodeCompatibility(node.data.manifest).available)
  }
  nodeCompatibilityTitle(nodeId: string): string | null {
    const node = this.store.nodes().find((item) => item.id === nodeId)
    if (!node) return null
    const compatibility = this.store.nodeCompatibility(node.data.manifest)
    return compatibility.available ? null : `Repair required: ${compatibility.reason ?? 'unavailable for the active execution profile'}`
  }
  nodeAccessibilityLabel(node: EditorNode): string {
    const reason = this.nodeCompatibilityTitle(node.id)
    return reason ? `Node ${node.data.label}. ${reason}` : `Node ${node.data.label}`
  }
  portTitle(nodeId: string, displayName: string, kind: string): string {
    const reason = this.nodeCompatibilityTitle(nodeId)
    return reason ? `${displayName} · ${kind}. ${reason}` : `${displayName} · ${kind}`
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
      if (manifest) {
        const nodeId = this.store.addNode(manifest, canvas ? flowPositionFromDrop(event, canvas) : { x: event.offsetX, y: event.offsetY })
        if (nodeId === null) this.setConnectionFeedback(`Node not added. ${this.store.nodeCompatibility(manifest).reason ?? 'Unavailable for the active execution profile.'}`)
      }
    } catch { /* The drop payload is external UI data; ignore malformed items. */ }
  }
  moveNodes(event: FMoveNodesEvent): void { this.store.moveNodes(event.nodes.map((node) => ({ id: node.id, position: { x: node.position.x, y: node.position.y } }))) }
  selectNode(nodeId: string): void { if (!this.readOnly) this.store.selectNode(nodeId) }
  select(event: FSelectionChangeEvent): void { if (this.readOnly) return; if (event.connectionIds.length) this.store.selectEdge(event.connectionIds[0]); else if (event.nodeIds.length) this.store.selectNode(event.nodeIds[0]); else this.store.clearSelection() }
  handleNodeKeydown(event: KeyboardEvent, nodeId: string): void {
    if (event.key === 'ContextMenu' || (event.key === 'F10' && event.shiftKey)) {
      event.preventDefault()
      event.stopPropagation()
      this.openKeyboardNodeMenu(event, nodeId)
      return
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      this.selectNode(nodeId)
      return
    }
    const direction = event.key === 'ArrowUp' ? 'up'
      : event.key === 'ArrowDown' ? 'down'
        : event.key === 'ArrowLeft' ? 'left'
          : event.key === 'ArrowRight' ? 'right'
            : null
    if (!direction) return
    event.preventDefault()
    this.focusNearestNode(nodeId, direction)
  }
  private openKeyboardNodeMenu(event: KeyboardEvent, nodeId: string): void {
    const target = event.currentTarget instanceof HTMLElement ? event.currentTarget : null
    const shell = target?.closest('.canvas-shell') ?? target
    const bounds = shell?.getBoundingClientRect()
    const targetBounds = target?.getBoundingClientRect()
    if (!bounds || !targetBounds) return
    if (!this.readOnly) this.store.selectNode(nodeId)
    this.canvasMenu.set(null)
    this.nodeMenu.set({
      x: Math.max(8, targetBounds.right - bounds.left),
      y: Math.max(8, targetBounds.top - bounds.top),
      nodeId,
    })
  }
  async handlePortKeydown(event: KeyboardEvent, nodeId: string, direction: 'input' | 'output', portId: string): Promise<void> {
    if (event.key === 'Escape') {
      if (this.keyboardConnectionSource()) {
        event.preventDefault()
        event.stopPropagation()
        this.keyboardConnectionSource.set(null)
        this.setConnectionFeedback(null)
      }
      return
    }
    if (event.key !== 'Enter' && event.key !== ' ') return
    event.preventDefault()
    event.stopPropagation()
    if (this.readOnly) return

    if (!this.isProfileCompatible(nodeId)) {
      this.setConnectionFeedback(`Connection not created. ${this.nodeCompatibilityTitle(nodeId)?.replace(/^Repair required: /, '') ?? 'The node is unavailable for the active execution profile.'}`)
      return
    }

    if (direction === 'output') {
      const current = this.keyboardConnectionSource()
      this.selectNode(nodeId)
      this.setConnectionFeedback(null)
      this.keyboardConnectionSource.set(current?.nodeId === nodeId && current.portId === portId ? null : { nodeId, direction, portId })
      return
    }

    const source = this.keyboardConnectionSource()
    if (!source) {
      this.selectNode(nodeId)
      return
    }
    const connection = normalizeConnection(
      connectorId(source.nodeId, 'output', source.portId),
      connectorId(nodeId, 'input', portId),
    )
    if (!connection) {
      this.setConnectionFeedback('Connection not created. The selected ports are not compatible.')
      return
    }
    if (await this.validateAndCommitConnection(connection)) this.keyboardConnectionSource.set(null)
  }
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
    if (!this.isProfileCompatible(connection.source) || !this.isProfileCompatible(connection.target)) {
      this.setConnectionFeedback('Connection not created. One of the selected nodes is unavailable for the active execution profile.')
      return
    }
    this.keyboardConnectionSource.set(null)
    await this.validateAndCommitConnection(connection)
  }
  private async validateAndCommitConnection(connection: NormalizedConnection): Promise<boolean> {
    if (!this.isProfileCompatible(connection.source) || !this.isProfileCompatible(connection.target)) {
      this.setConnectionFeedback('Connection not created. One of the selected nodes is unavailable for the active execution profile.')
      return false
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
        return false
      }
      if (!result.valid) {
        this.setConnectionFeedback(this.connectionDiagnosticMessage(result.diagnostics))
        return false
      }
      if (!this.store.commitConnection(connection, expectedRevision)) {
        this.setConnectionFeedback('Connection not created. The graph changed while it was being checked.')
        return false
      }
      this.setConnectionFeedback(null)
      return true
    } catch (reason) {
      this.setConnectionFeedback(`Connection not created. ${reason instanceof Error ? reason.message : String(reason)}`)
      return false
    }
  }
  connectionTargets(nodeId: string, portId: string): string[] {
    if (!this.isProfileCompatible(nodeId)) return this.noConnectionTargets
    return this.connectionTargetMap().get(`${nodeId}|${portId}`) ?? this.noConnectionTargets
  }
  isKeyboardSource(nodeId: string, portId: string): boolean {
    const source = this.keyboardConnectionSource()
    return source?.direction === 'output' && source.nodeId === nodeId && source.portId === portId
  }
  isKeyboardTarget(nodeId: string, portId: string): boolean {
    const source = this.keyboardConnectionSource()
    return source?.direction === 'output'
      && this.connectionTargets(source.nodeId, source.portId).includes(connectorId(nodeId, 'input', portId))
  }
  keyboardPortLabel(node: EditorNode, direction: 'input' | 'output', portId: string): string {
    const ports = direction === 'input' ? node.data.manifest.inputs : node.data.manifest.outputs
    const port = ports.find((item) => item.id === portId)
    const selected = direction === 'output' && this.isKeyboardSource(node.id, portId)
    const compatibility = this.nodeCompatibilityTitle(node.id)
    return `Port ${port?.display_name ?? portId} (${direction}) on ${node.data.label}${selected ? '. Connection source selected.' : ''}${compatibility ? `. ${compatibility}` : ''}`
  }
  private focusNearestNode(nodeId: string, direction: 'up' | 'down' | 'left' | 'right'): void {
    const current = Array.from(document.querySelectorAll<HTMLElement>('[data-node-id]'))
      .find((element) => element.dataset['nodeId'] === nodeId)
    if (!current) return
    const currentBounds = current.getBoundingClientRect()
    const currentCenter = { x: currentBounds.left + currentBounds.width / 2, y: currentBounds.top + currentBounds.height / 2 }
    const candidates = Array.from(current.closest('.canvas-shell')?.querySelectorAll<HTMLElement>('[data-node-id]') ?? [])
      .filter((element) => element !== current)
      .map((element) => {
        const bounds = element.getBoundingClientRect()
        const center = { x: bounds.left + bounds.width / 2, y: bounds.top + bounds.height / 2 }
        const delta = { x: center.x - currentCenter.x, y: center.y - currentCenter.y }
        const inDirection = direction === 'right' ? delta.x > 4
          : direction === 'left' ? delta.x < -4
            : direction === 'down' ? delta.y > 4
              : delta.y < -4
        if (!inDirection) return null
        const primary = direction === 'left' || direction === 'right' ? Math.abs(delta.x) : Math.abs(delta.y)
        const secondary = direction === 'left' || direction === 'right' ? Math.abs(delta.y) : Math.abs(delta.x)
        return { element, score: primary + secondary * 2 }
      })
      .filter((candidate): candidate is { element: HTMLElement; score: number } => candidate !== null)
      .sort((left, right) => left.score - right.score)
    candidates[0]?.element.focus()
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
  fitView(): void { if (!this.viewportLocked() && this.hasNodes()) this.canvas()?.fitToScreen(graphFitPadding(this.store.nodes(), 0.1, GRAPH_NODE_SIZE), true) }
  fitOnNodesRendered(): void {
    if (this.viewportLocked() || this.graphFitted || !this.store.nodes().length) return
    this.canvas()?.fitToScreen(graphFitPadding(this.store.nodes(), this.readOnly ? 0.18 : 0.1, GRAPH_NODE_SIZE), false)
    this.graphFitted = true
  }

  @HostListener('document:keydown', ['$event'])
  handleKeyboard(event: KeyboardEvent): void {
    if (event.key === 'Escape' && this.keyboardConnectionSource()) {
      event.preventDefault()
      this.keyboardConnectionSource.set(null)
      this.setConnectionFeedback(null)
      return
    }
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
    if (command && event.key.toLowerCase() === 'l' && !editing) { event.preventDefault(); this.autoLayout(); return }
    if (!editing && (event.key === 'Delete' || event.key === 'Backspace')) { event.preventDefault(); this.store.deleteSelected() }
  }

  @HostListener('document:pointerdown', ['$event'])
  closeMenusOnOutsidePointer(event: PointerEvent): void {
    const target = event.target instanceof Element ? event.target : null
    if (target?.closest('.canvas-context-menu')) return
    this.closeMenus()
  }
}
