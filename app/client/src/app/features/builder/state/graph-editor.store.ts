import { Injectable, computed, signal } from '@angular/core'
import type { EditorEdge, EditorNode } from '../../../domain/editor.types'
import { materializedInputs, materializedOutputs } from '../../../domain/ports'
import {
  cloneEditorSession,
  defaultExportSettings,
  editorNodeFromGraph,
  sessionToGraph,
  type EditorHistorySnapshot,
  type GraphEditorSession,
} from '../../../domain/graph-mapper'
import {
  conditionMatches,
  emptyModelConfiguration,
  effectiveParameterDefault,
  parameterValueMatchesDefinition,
  resolveImplementationCompatibility,
  resolveParameterCompatibility,
  type ExecutionBackend,
  type ImplementationCompatibility,
  type ModelConfiguration,
  type ModelGraph,
  type NodeManifest,
  type OutputBinding,
  type ParameterCompatibility,
  type ProjectDocument,
  type ShapeSummary,
} from '../../../domain/types'
import { layeredLayout } from '../graph/graph-layout.service'
import { validateGraphConnection, type GraphConnection, type GraphConnectionValidation } from '../graph/graph-validation'

export type { GraphConnection, GraphConnectionValidation } from '../graph/graph-validation'
export { wouldCreateCycle } from '../graph/graph-validation'

export interface NodePositionChange {
  id: string
  position: { x: number; y: number }
}

@Injectable({ providedIn: 'root' })
export class GraphEditorStore {
  readonly nodes = signal<EditorNode[]>([])
  readonly edges = signal<EditorEdge[]>([])
  readonly outputBindings = signal<OutputBinding[]>([])
  readonly manifests = signal<NodeManifest[]>([])
  readonly modelConfiguration = signal<ModelConfiguration>(emptyModelConfiguration())
  readonly exportSettings = signal<ModelGraph['export_settings']>(defaultExportSettings())
  readonly futureTrainingConfiguration = signal<ModelGraph['future_training_configuration']>({})
  readonly selectedNodeId = signal<string | null>(null)
  readonly selectedEdgeId = signal<string | null>(null)
  readonly copiedNodeIds = signal<string[]>([])
  readonly past = signal<EditorHistorySnapshot[]>([])
  readonly future = signal<EditorHistorySnapshot[]>([])
  readonly graphRevision = signal(0)
  readonly draftRevision = signal(0)
  readonly shapeSummaries = signal<ShapeSummary[]>([])
  readonly shapeSummaryByNode = computed(() => new Map(this.shapeSummaries().map((summary) => [summary.node_id, summary])))
  readonly canUndo = computed(() => this.past().length > 0)
  readonly canRedo = computed(() => this.future().length > 0)
  readonly executionProfile = computed(() => ({
    executionBackend: this.activeExecutionBackend(),
    kerasEnabled: this.activeKerasEnabled(),
  }))
  private readonly activeExecutionBackend = signal<ExecutionBackend>('pytorch')
  private readonly activeKerasEnabled = signal(true)

  private dragging = false
  private dragCheckpointed = false
  private dragChanged = false

  setManifests(manifests: NodeManifest[]): void {
    this.manifests.set(manifests)
  }

  setExecutionProfile(executionBackend: ExecutionBackend, kerasEnabled: boolean): void {
    if (executionBackend === this.activeExecutionBackend() && kerasEnabled === this.activeKerasEnabled()) return
    this.activeExecutionBackend.set(executionBackend)
    this.activeKerasEnabled.set(kerasEnabled)
    this.invalidateDerivedState()
  }

  nodeCompatibility(manifest: NodeManifest): ImplementationCompatibility {
    return resolveImplementationCompatibility(
      manifest,
      this.activeExecutionBackend(),
      this.activeKerasEnabled(),
    )
  }

  parameterCompatibility(
    node: EditorNode,
    definition: NodeManifest['parameters'][number],
    value: unknown,
    parameters = node.data.parameters,
  ): ParameterCompatibility {
    return resolveParameterCompatibility(
      node.data.manifest,
      definition,
      value,
      parameters,
      this.activeExecutionBackend(),
      this.activeKerasEnabled(),
    )
  }

  canDuplicate(nodeId: string): boolean {
    const node = this.nodes().find((item) => item.id === nodeId)
    return Boolean(node && this.canMaterializeNode(node.data.manifest, node.data.parameters))
  }

  nodeParametersFor(
    manifest: NodeManifest,
    overrides: Record<string, unknown> = {},
  ): Record<string, unknown> {
    return { ...this.defaultParameters(manifest), ...structuredClone(overrides) }
  }

  canMaterializeNode(manifest: NodeManifest, parameters: Record<string, unknown>): boolean {
    return this.nodeCompatibility(manifest).available && this.parametersCompatible(manifest, parameters)
  }

  shapeSummary(nodeId: string): ShapeSummary | undefined {
    return this.shapeSummaryByNode().get(nodeId)
  }

  invalidateDerivedState(): void {
    this.graphRevision.update((revision) => revision + 1)
    this.draftRevision.update((revision) => revision + 1)
    this.shapeSummaries.set([])
  }

  checkpoint(): void {
    this.past.set([...this.past(), this.snapshot()].slice(-50))
    this.future.set([])
  }

  beginDrag(): void {
    if (this.dragging) return
    this.dragging = true
    this.dragCheckpointed = false
    this.dragChanged = false
  }

  endDrag(): void {
    if (!this.dragging) return
    if (this.dragChanged) this.draftRevision.update((revision) => revision + 1)
    this.dragging = false
    this.dragCheckpointed = false
    this.dragChanged = false
  }

  moveNodes(changes: readonly NodePositionChange[]): void {
    if (!changes.length) return
    const positions = new Map(changes.map((change) => [change.id, change.position]))
    const changed = this.nodes().some((node) => {
      if (node.locked) return false
      const position = positions.get(node.id)
      return Boolean(position && (position.x !== node.position.x || position.y !== node.position.y))
    })
    if (!changed) return
    if (this.dragging) {
      if (!this.dragCheckpointed) {
        this.checkpoint()
        this.dragCheckpointed = true
      }
      this.dragChanged = true
    }
    this.nodes.update((nodes) => nodes.map((node) => {
      if (node.locked) return node
      const position = positions.get(node.id)
      return position ? { ...node, position: { ...position } } : node
    }))
    if (!this.dragging) this.draftRevision.update((revision) => revision + 1)
  }

  addNode(manifest: NodeManifest, position: { x: number; y: number }): string | null {
    if (!this.nodeCompatibility(manifest).available) return null
    const editorParameters = this.editorParameters(manifest)
    const parameters = this.parametersCompatible(manifest, editorParameters)
      ? editorParameters
      : this.defaultParameters(manifest)
    if (!this.parametersCompatible(manifest, parameters)) return null
    const node: EditorNode = {
      id: newId(manifest.id.replaceAll('.', '-')),
      type: 'modelNode',
      position: snapPosition(position),
      locked: false,
      data: {
        manifest,
        parameters,
        label: manifest.display_name,
      },
    }
    this.checkpoint()
    this.nodes.update((nodes) => [...nodes, node])
    this.selectedNodeId.set(node.id)
    this.selectedEdgeId.set(null)
    this.invalidateDerivedState()
    return node.id
  }

  isConnectionValid(connection: GraphConnection): boolean {
    return validateGraphConnection(
      this.nodes(),
      this.edges(),
      connection,
      this.activeExecutionBackend(),
      this.activeKerasEnabled(),
    ).valid
  }

  connect(connection: GraphConnection): GraphConnectionValidation {
    const result = validateGraphConnection(
      this.nodes(),
      this.edges(),
      connection,
      this.activeExecutionBackend(),
      this.activeKerasEnabled(),
    )
    if (!result.valid) return result
    this.commitConnection(connection, this.graphRevision())
    return result
  }

  commitConnection(connection: GraphConnection, expectedRevision: number): boolean {
    if (expectedRevision !== this.graphRevision()) return false
    const result = validateGraphConnection(
      this.nodes(),
      this.edges(),
      connection,
      this.activeExecutionBackend(),
      this.activeKerasEnabled(),
    )
    if (!result.valid) return false
    this.checkpoint()
    this.edges.update((edges) => [...edges, {
      ...connection,
      id: newId('edge'),
      type: 'bezier',
    }])
    this.invalidateDerivedState()
    return true
  }

  selectNode(nodeId: string | null): void {
    this.selectedNodeId.set(nodeId)
    this.selectedEdgeId.set(null)
  }

  selectEdge(edgeId: string | null): void {
    this.selectedEdgeId.set(edgeId)
    this.selectedNodeId.set(null)
  }

  clearSelection(): void {
    this.selectedNodeId.set(null)
    this.selectedEdgeId.set(null)
  }

  updateNode(nodeId: string, patch: { label?: string; parameters?: Record<string, unknown>; locked?: boolean }): boolean {
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node) return false
    if (patch.parameters !== undefined && !this.canUpdateParameters(node, patch.parameters)) return false
    const nextNode: EditorNode = {
      ...node,
      locked: patch.locked ?? node.locked,
      data: {
        ...node.data,
        label: patch.label ?? node.data.label,
        parameters: patch.parameters ?? node.data.parameters,
      },
    }
    const nextNodes = this.nodes().map((item) => item.id === nodeId ? nextNode : item)
    const nextEdges = this.edges().filter((edge) => (
      (edge.source !== nodeId || this.portStillAvailable(nextNode, 'output', edge.sourceHandle ?? 'output'))
      && (edge.target !== nodeId || this.portStillAvailable(nextNode, 'input', edge.targetHandle ?? 'input'))
    ))
    const nextOutputBindings = this.outputBindings().filter((binding) => (
      binding.node_id !== nodeId || this.portStillAvailable(nextNode, 'output', binding.port_id)
    ))
    this.checkpoint()
    this.nodes.set(nextNodes)
    this.edges.set(nextEdges)
    this.outputBindings.set(uniqueOutputBindings(nextOutputBindings))
    this.invalidateDerivedState()
    return true
  }

  deleteSelected(): void {
    const selectedNodeId = this.selectedNodeId()
    const selectedEdgeId = this.selectedEdgeId()
    if (!selectedNodeId && !selectedEdgeId) return
    this.checkpoint()
    if (selectedEdgeId) {
      this.edges.update((edges) => edges.filter((edge) => edge.id !== selectedEdgeId))
      this.selectedEdgeId.set(null)
      this.invalidateDerivedState()
      return
    }
    this.nodes.update((nodes) => nodes.filter((node) => node.id !== selectedNodeId))
    this.edges.update((edges) => edges.filter((edge) => edge.source !== selectedNodeId && edge.target !== selectedNodeId))
    this.outputBindings.update((bindings) => bindings.filter((binding) => binding.node_id !== selectedNodeId))
    this.selectedNodeId.set(null)
    this.invalidateDerivedState()
  }

  copySelected(): void {
    this.copiedNodeIds.set(this.selectedNodeId() ? [this.selectedNodeId() as string] : [])
  }

  pasteCopied(): boolean {
    const selected = this.nodes().filter((node) => this.copiedNodeIds().includes(node.id))
    if (!selected.length || selected.some((node) => !this.canMaterializeNode(node.data.manifest, node.data.parameters))) return false
    const copies = selected.map((node) => ({
      ...structuredClone(node),
      id: newId(node.data.manifest.id.replaceAll('.', '-')),
      position: { x: node.position.x + 32, y: node.position.y + 32 },
      selected: false,
    }))
    this.checkpoint()
    this.nodes.update((nodes) => [...nodes, ...copies])
    this.selectedNodeId.set(copies[0].id)
    this.selectedEdgeId.set(null)
    this.invalidateDerivedState()
    return true
  }

  duplicateExact(nodeId: string): string | null {
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node || !this.canMaterializeNode(node.data.manifest, node.data.parameters)) return null
    const copy = this.copyNode(node, node.data.parameters, node.data.label)
    this.checkpoint()
    this.nodes.update((nodes) => [...nodes, copy])
    this.selectNode(copy.id)
    this.invalidateDerivedState()
    return copy.id
  }

  duplicateWithDefaults(nodeId: string): string | null {
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node || !this.canMaterializeNode(node.data.manifest, node.data.parameters)) return null
    const parameters = this.defaultParameters(node.data.manifest)
    if (!this.parametersCompatible(node.data.manifest, parameters)) return null
    const copy = this.copyNode(node, parameters, node.data.manifest.display_name)
    this.checkpoint()
    this.nodes.update((nodes) => [...nodes, copy])
    this.selectNode(copy.id)
    this.invalidateDerivedState()
    return copy.id
  }

  toggleNodeLocked(nodeId: string): void {
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node) return
    this.checkpoint()
    this.nodes.update((nodes) => nodes.map((item) => item.id === nodeId ? { ...item, locked: !item.locked } : item))
    this.invalidateDerivedState()
  }

  isOutput(nodeId: string, portId?: string): boolean {
    return this.outputBindings().some((binding) => binding.node_id === nodeId && (!portId || binding.port_id === portId))
  }

  isInferredOutput(nodeId: string, portId?: string): boolean {
    if (this.outputBindings().length) return false
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node || node.data.manifest.semantic_role === 'input') return false
    return this.outputPorts(node).some((port) => {
      if (portId && port.id !== portId) return false
      if (port.enabled_when && !conditionMatches(port.enabled_when, node.data.parameters)) return false
      if (!isValueOutputPort(port.kind)) return false
      return !this.edges().some((edge) => edge.source === nodeId && (edge.sourceHandle ?? 'output') === port.id)
    })
  }

  isEffectiveOutput(nodeId: string): boolean {
    return this.isOutput(nodeId) || this.isInferredOutput(nodeId)
  }

  hasEnabledOutputPort(nodeId: string): boolean {
    const node = this.nodes().find((item) => item.id === nodeId)
    return node ? this.outputPorts(node).some((port) => (
      isValueOutputPort(port.kind)
      && (!port.enabled_when || conditionMatches(port.enabled_when, node.data.parameters))
    )) : false
  }

  canToggleNodeOutput(nodeId: string): boolean {
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node || node.data.manifest.semantic_role === 'input') return false
    if (this.isOutput(nodeId)) return true
    if (!this.nodeCompatibility(node.data.manifest).available) return false
    return this.hasEnabledOutputPort(nodeId) && !this.isInferredOutput(nodeId)
  }

  toggleNodeOutput(nodeId: string, portId?: string): void {
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node) return
    if (node.data.manifest.semantic_role === 'input') return
    const explicitBindings = this.outputBindings().filter((binding) => binding.node_id === nodeId)
    if (!portId && explicitBindings.length) {
      this.checkpoint()
      this.outputBindings.update((bindings) => bindings.filter((binding) => binding.node_id !== nodeId))
      this.invalidateDerivedState()
      return
    }
    if (!this.nodeCompatibility(node.data.manifest).available) return
    const candidate = portId
      ? this.outputPorts(node).find((port) => port.id === portId)
      : this.outputPorts(node).find((port) => (
        isValueOutputPort(port.kind)
        && (!port.enabled_when || conditionMatches(port.enabled_when, node.data.parameters))
        && port.id === 'output'
      )) ?? this.outputPorts(node).find((port) => (
        isValueOutputPort(port.kind)
        && (!port.enabled_when || conditionMatches(port.enabled_when, node.data.parameters))
      ))
    if (!candidate || !isValueOutputPort(candidate.kind)) return
    if (candidate.enabled_when && !conditionMatches(candidate.enabled_when, node.data.parameters)) return
    const existing = this.outputBindings().some((binding) => binding.node_id === nodeId && binding.port_id === candidate.id)
    if (existing) {
      this.checkpoint()
      this.outputBindings.update((bindings) => bindings.filter((binding) => !(binding.node_id === nodeId && binding.port_id === candidate.id)))
      this.invalidateDerivedState()
      return
    }
    if (!this.outputBindings().length && this.isInferredOutput(nodeId, candidate.id)) return
    this.checkpoint()
    this.outputBindings.update((bindings) => uniqueOutputBindings([...bindings, { node_id: nodeId, port_id: candidate.id }]))
    this.invalidateDerivedState()
  }

  disconnectNode(nodeId: string): void {
    if (!this.nodes().some((node) => node.id === nodeId)) return
    if (!this.edges().some((edge) => edge.source === nodeId || edge.target === nodeId)) return
    this.checkpoint()
    this.edges.update((edges) => edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId))
    this.invalidateDerivedState()
  }

  resetParameters(nodeId: string): void {
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node || !node.data.manifest.parameters.length) return
    const parameters = this.defaultParameters(node.data.manifest)
    this.checkpoint()
    this.nodes.update((nodes) => nodes.map((item) => item.id === nodeId ? {
      ...item,
      data: { ...item.data, parameters },
    } : item))
    this.invalidateDerivedState()
  }

  undo(): void {
    const previous = this.past().at(-1)
    if (!previous) return
    this.future.set([this.snapshot(), ...this.future()].slice(0, 50))
    this.past.set(this.past().slice(0, -1))
    this.restoreSnapshot(previous)
    this.selectedNodeId.set(null)
    this.selectedEdgeId.set(null)
    this.invalidateDerivedState()
  }

  redo(): void {
    const next = this.future()[0]
    if (!next) return
    this.past.set([...this.past(), this.snapshot()].slice(-50))
    this.future.set(this.future().slice(1))
    this.restoreSnapshot(next)
    this.selectedNodeId.set(null)
    this.selectedEdgeId.set(null)
    this.invalidateDerivedState()
  }

  replaceGraph(project: ProjectDocument): void {
    const manifestMap = new Map(this.manifests().map((manifest) => [`${manifest.id}@${manifest.version}`, manifest]))
    const nodes = project.graph.nodes.map((node) => {
      const manifest = manifestMap.get(`${node.definition_id}@${node.definition_version}`)
      if (!manifest) throw new Error(`Missing node manifest ${node.definition_id}@${node.definition_version}`)
      return editorNodeFromGraph(node, manifest)
    })
    this.nodes.set(nodes)
    this.edges.set(project.graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source_node_id,
      sourceHandle: edge.source_port_id,
      target: edge.target_node_id,
      targetHandle: edge.target_port_id,
      type: 'bezier',
    })))
    this.outputBindings.set(uniqueOutputBindings(project.graph.output_bindings))
    this.modelConfiguration.set(structuredClone(project.graph.model_configuration))
    this.exportSettings.set(structuredClone(project.graph.export_settings))
    this.futureTrainingConfiguration.set(structuredClone(project.graph.future_training_configuration))
    this.selectedNodeId.set(null)
    this.selectedEdgeId.set(null)
    this.copiedNodeIds.set([])
    this.past.set([])
    this.future.set([])
    this.invalidateDerivedState()
  }

  captureEditorSession(): GraphEditorSession {
    return cloneEditorSession({
      nodes: this.nodes(),
      edges: this.edges(),
      outputBindings: this.outputBindings(),
      modelConfiguration: this.modelConfiguration(),
      exportSettings: this.exportSettings(),
      futureTrainingConfiguration: this.futureTrainingConfiguration(),
      selectedNodeId: this.selectedNodeId(),
      selectedEdgeId: this.selectedEdgeId(),
      copiedNodeIds: this.copiedNodeIds(),
      past: this.past(),
      future: this.future(),
    })
  }

  restoreEditorSession(session: GraphEditorSession): void {
    const cloned = cloneEditorSession(session)
    this.nodes.set(cloned.nodes)
    this.edges.set(cloned.edges)
    this.outputBindings.set(uniqueOutputBindings(cloned.outputBindings))
    this.modelConfiguration.set(cloned.modelConfiguration)
    this.exportSettings.set(cloned.exportSettings)
    this.futureTrainingConfiguration.set(cloned.futureTrainingConfiguration)
    this.selectedNodeId.set(cloned.selectedNodeId)
    this.selectedEdgeId.set(cloned.selectedEdgeId)
    this.copiedNodeIds.set(cloned.copiedNodeIds)
    this.past.set(cloned.past)
    this.future.set(cloned.future)
    this.invalidateDerivedState()
  }

  reset(): void {
    this.nodes.set([])
    this.edges.set([])
    this.outputBindings.set([])
    this.modelConfiguration.set(emptyModelConfiguration())
    this.exportSettings.set(defaultExportSettings())
    this.futureTrainingConfiguration.set({})
    this.selectedNodeId.set(null)
    this.selectedEdgeId.set(null)
    this.copiedNodeIds.set([])
    this.past.set([])
    this.future.set([])
    this.invalidateDerivedState()
  }

  toGraph(): ModelGraph {
    return sessionToGraph({
      nodes: this.nodes(),
      edges: this.edges(),
      outputBindings: this.outputBindings(),
      modelConfiguration: this.modelConfiguration(),
      exportSettings: this.exportSettings(),
      futureTrainingConfiguration: this.futureTrainingConfiguration(),
      selectedNodeId: this.selectedNodeId(),
      selectedEdgeId: this.selectedEdgeId(),
      copiedNodeIds: this.copiedNodeIds(),
      past: this.past(),
      future: this.future(),
    })
  }

  applyShapeSummaries(summaries: ShapeSummary[], expectedRevision = this.graphRevision()): boolean {
    if (expectedRevision !== this.graphRevision()) return false
    this.shapeSummaries.set(structuredClone(summaries))
    return true
  }

  updateConfiguration(configuration: ModelConfiguration): void {
    this.checkpoint()
    this.modelConfiguration.set(structuredClone(configuration))
    this.invalidateDerivedState()
  }

  autoLayout(nodes: EditorNode[], edges: EditorEdge[]): void {
    if (!nodes.length) return
    this.checkpoint()
    const positions = layeredLayout(nodes, edges)
    this.nodes.set(nodes.map((node) => ({
      ...node,
      position: node.locked ? { ...node.position } : positions.get(node.id) ?? { ...node.position },
    })))
    this.edges.set(edges.map((edge) => ({ ...edge })))
    this.invalidateDerivedState()
  }

  private snapshot(): EditorHistorySnapshot {
    return {
      nodes: this.nodes().map((node) => ({ ...node, position: { ...node.position }, data: { ...node.data, parameters: structuredClone(node.data.parameters) } })),
      edges: this.edges().map((edge) => ({ ...edge })),
      outputBindings: structuredClone(this.outputBindings()),
      modelConfiguration: structuredClone(this.modelConfiguration()),
      exportSettings: structuredClone(this.exportSettings()),
      futureTrainingConfiguration: structuredClone(this.futureTrainingConfiguration()),
    }
  }

  private restoreSnapshot(snapshot: EditorHistorySnapshot): void {
    this.nodes.set(snapshot.nodes.map((node) => ({ ...node, position: { ...node.position }, data: { ...node.data, parameters: structuredClone(node.data.parameters) } })))
    this.edges.set(snapshot.edges.map((edge) => ({ ...edge })))
    this.outputBindings.set(uniqueOutputBindings(snapshot.outputBindings))
    this.modelConfiguration.set(structuredClone(snapshot.modelConfiguration))
    this.exportSettings.set(structuredClone(snapshot.exportSettings))
    this.futureTrainingConfiguration.set(structuredClone(snapshot.futureTrainingConfiguration))
  }

  private copyNode(node: EditorNode, parameters: Record<string, unknown>, label: string): EditorNode {
    return {
      id: newId(node.data.manifest.id.replaceAll('.', '-')),
      type: 'modelNode',
      position: { x: node.position.x + 32, y: node.position.y + 32 },
      locked: false,
      selected: false,
      data: {
        manifest: node.data.manifest,
        label,
        parameters: structuredClone(parameters),
      },
    }
  }

  private defaultParameters(manifest: NodeManifest): Record<string, unknown> {
    const mapping = this.nodeCompatibility(manifest).mapping
    return Object.fromEntries(manifest.parameters.flatMap((definition) => {
      const value = effectiveParameterDefault(definition, mapping?.parameters?.[definition.id])
      return value === undefined ? [] : [[definition.id, structuredClone(value)]]
    }))
  }

  private outputPorts(node: EditorNode): NodeManifest['outputs'] {
    try {
      return materializedOutputs(node.data.manifest, node.data.parameters)
    } catch {
      return []
    }
  }

  private portStillAvailable(
    node: EditorNode,
    direction: 'input' | 'output',
    portId: string,
  ): boolean {
    try {
      const ports = direction === 'input'
        ? materializedInputs(node.data.manifest, node.data.parameters)
        : materializedOutputs(node.data.manifest, node.data.parameters)
      const port = ports.find((item) => item.id === portId)
      return Boolean(port && conditionMatches(port.enabled_when, node.data.parameters))
    } catch {
      return false
    }
  }

  private editorParameters(manifest: NodeManifest): Record<string, unknown> {
    return Object.fromEntries(manifest.parameters.flatMap((definition) => {
      const value = definition.editor_default ?? definition.default
      return value === undefined ? [] : [[definition.id, structuredClone(value)]]
    }))
  }

  private parametersCompatible(manifest: NodeManifest, parameters: Record<string, unknown>): boolean {
    if (Object.keys(parameters).some((id) => !manifest.parameters.some((definition) => definition.id === id))) return false
    const mapping = this.nodeCompatibility(manifest).mapping
    return manifest.parameters.every((definition) => {
      const supplied = Object.prototype.hasOwnProperty.call(parameters, definition.id)
      const required = definition.required
        || Boolean(definition.required_when && Object.keys(definition.required_when).length && conditionMatches(definition.required_when, parameters))
      if (required && !supplied) return false
      const value = supplied ? parameters[definition.id] : effectiveParameterDefault(definition, mapping?.parameters?.[definition.id])
      if (!supplied && (value === undefined || value === null)) return true
      if (!definition.visible_when || conditionMatches(definition.visible_when, parameters)) {
        return resolveParameterCompatibility(
          manifest,
          definition,
          value,
          parameters,
          this.activeExecutionBackend(),
          this.activeKerasEnabled(),
        ).valueCompatible
      }
      return true
    })
  }

  private canUpdateParameters(node: EditorNode, parameters: Record<string, unknown>): boolean {
    const definitions = new Map(node.data.manifest.parameters.map((definition) => [definition.id, definition]))
    if (Object.keys(parameters).some((id) => !definitions.has(id))) return false
    const current = node.data.parameters
    for (const definition of node.data.manifest.parameters) {
      const currentlySupplied = Object.prototype.hasOwnProperty.call(current, definition.id)
      const nextSupplied = Object.prototype.hasOwnProperty.call(parameters, definition.id)
      const required = definition.required
        || Boolean(definition.required_when && Object.keys(definition.required_when).length && conditionMatches(definition.required_when, parameters))
      if (!nextSupplied) {
        if (!currentlySupplied) continue
        if (required) return false
        const defaultValue = effectiveParameterDefault(
          definition,
          this.nodeCompatibility(node.data.manifest).mapping?.parameters?.[definition.id],
        )
        if (!sameValue(current[definition.id], defaultValue)
          && !resolveParameterCompatibility(
            node.data.manifest,
            definition,
            defaultValue,
            parameters,
            this.activeExecutionBackend(),
            this.activeKerasEnabled(),
          ).editable) return false
        continue
      }
      const nextValue = parameters[definition.id]
      if (currentlySupplied && sameValue(current[definition.id], nextValue)) continue
      if (!parameterValueMatchesDefinition(
        definition,
        nextValue,
        this.nodeCompatibility(node.data.manifest).mapping?.parameters?.[definition.id],
      )) return false
      const compatibility = resolveParameterCompatibility(
        node.data.manifest,
        definition,
        nextValue,
        parameters,
        this.activeExecutionBackend(),
        this.activeKerasEnabled(),
      )
      if (!compatibility.editable || !compatibility.valueCompatible) return false
    }
    return true
  }
}

function newId(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`
}

function snapPosition(position: { x: number; y: number }): { x: number; y: number } {
  return { x: Math.round(position.x / 16) * 16, y: Math.round(position.y / 16) * 16 }
}

function sameValue(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}

function isValueOutputPort(kind: string): boolean {
  return ['tensor', 'tensor_list', 'mask', 'scalar'].includes(kind)
}

function uniqueOutputBindings(bindings: OutputBinding[]): OutputBinding[] {
  const seen = new Set<string>()
  return bindings.filter((binding) => {
    const key = `${binding.node_id}:${binding.port_id}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  }).map((binding) => ({ ...binding }))
}
