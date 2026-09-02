import { Injectable, computed, signal } from '@angular/core'
import type { EditorEdge, EditorNode } from '../../../domain/editor.types'
import {
  cloneEditorSession,
  defaultExportSettings,
  editorNodeFromGraph,
  type EditorHistorySnapshot,
  type GraphEditorSession,
} from '../../../domain/graph-mapper'
import { emptyModelConfiguration, implementationKey, type ExecutionBackend, type ModelConfiguration, type ModelGraph, type NodeManifest, type OutputBinding, type ProjectDocument, type ShapeSummary } from '../../../domain/types'
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
  readonly canUndo = computed(() => this.past().length > 0)
  readonly canRedo = computed(() => this.future().length > 0)
  private readonly activeExecutionBackend = signal<ExecutionBackend>('pytorch')
  private readonly activeKerasEnabled = signal(true)

  private dragging = false
  private dragCheckpointed = false

  setManifests(manifests: NodeManifest[]): void {
    this.manifests.set(manifests)
  }

  setExecutionProfile(executionBackend: ExecutionBackend, kerasEnabled: boolean): void {
    this.activeExecutionBackend.set(executionBackend)
    this.activeKerasEnabled.set(kerasEnabled)
  }

  checkpoint(): void {
    this.past.set([...this.past(), this.snapshot()].slice(-50))
    this.future.set([])
  }

  beginDrag(): void {
    if (this.dragging) return
    this.dragging = true
    this.dragCheckpointed = false
  }

  endDrag(): void {
    this.dragging = false
    this.dragCheckpointed = false
  }

  moveNodes(changes: readonly NodePositionChange[]): void {
    if (!changes.length) return
    if (this.dragging && !this.dragCheckpointed) {
      this.checkpoint()
      this.dragCheckpointed = true
    }
    const positions = new Map(changes.map((change) => [change.id, change.position]))
    this.nodes.update((nodes) => nodes.map((node) => {
      if (node.locked) return node
      const position = positions.get(node.id)
      return position ? { ...node, position: { ...position } } : node
    }))
  }

  addNode(manifest: NodeManifest, position: { x: number; y: number }): string {
    const node: EditorNode = {
      id: newId(manifest.id.replaceAll('.', '-')),
      type: 'modelNode',
      position: snapPosition(position),
      locked: false,
      data: {
        manifest,
        parameters: Object.fromEntries(manifest.parameters.map((item) => [item.id, structuredClone(editorDefault(item))])),
        label: manifest.display_name,
      },
    }
    this.checkpoint()
    this.nodes.update((nodes) => [...nodes, node])
    this.selectedNodeId.set(node.id)
    this.selectedEdgeId.set(null)
    return node.id
  }

  isConnectionValid(connection: GraphConnection): boolean {
    return validateGraphConnection(this.nodes(), this.edges(), connection).valid
  }

  connect(connection: GraphConnection): GraphConnectionValidation {
    const result = validateGraphConnection(this.nodes(), this.edges(), connection)
    if (!result.valid) return result
    this.checkpoint()
    this.edges.update((edges) => [...edges, {
      ...connection,
      id: newId('edge'),
      type: 'bezier',
    }])
    return result
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

  updateNode(nodeId: string, patch: { label?: string; parameters?: Record<string, unknown>; locked?: boolean }): void {
    this.checkpoint()
    this.nodes.update((nodes) => nodes.map((node) => node.id === nodeId ? {
      ...node,
      locked: patch.locked ?? node.locked,
      data: {
        ...node.data,
        label: patch.label ?? node.data.label,
        parameters: patch.parameters ?? node.data.parameters,
      },
    } : node))
  }

  deleteSelected(): void {
    const selectedNodeId = this.selectedNodeId()
    const selectedEdgeId = this.selectedEdgeId()
    if (!selectedNodeId && !selectedEdgeId) return
    this.checkpoint()
    if (selectedEdgeId) {
      this.edges.update((edges) => edges.filter((edge) => edge.id !== selectedEdgeId))
      this.selectedEdgeId.set(null)
      return
    }
    this.nodes.update((nodes) => nodes.filter((node) => node.id !== selectedNodeId))
    this.edges.update((edges) => edges.filter((edge) => edge.source !== selectedNodeId && edge.target !== selectedNodeId))
    this.outputBindings.update((bindings) => bindings.filter((binding) => binding.node_id !== selectedNodeId))
    this.selectedNodeId.set(null)
  }

  copySelected(): void {
    this.copiedNodeIds.set(this.selectedNodeId() ? [this.selectedNodeId() as string] : [])
  }

  pasteCopied(): void {
    const selected = this.nodes().filter((node) => this.copiedNodeIds().includes(node.id))
    if (!selected.length) return
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
  }

  duplicateExact(nodeId: string): string | null {
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node) return null
    const copy = this.copyNode(node, node.data.parameters, node.data.label)
    this.checkpoint()
    this.nodes.update((nodes) => [...nodes, copy])
    this.selectNode(copy.id)
    return copy.id
  }

  duplicateWithDefaults(nodeId: string, executionBackend?: ExecutionBackend, kerasEnabled = true): string | null {
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node) return null
    const selectedBackend = executionBackend ?? this.activeExecutionBackend()
    const selectedKeras = executionBackend === undefined ? this.activeKerasEnabled() : kerasEnabled
    const implementation = node.data.manifest.implementations[implementationKey(selectedBackend, selectedKeras)]
    const parameters = Object.fromEntries(
      node.data.manifest.parameters.map((item) => [
        item.id,
        structuredClone(frameworkDefault(item, implementation?.parameters?.[item.id])),
      ]),
    )
    const copy = this.copyNode(node, parameters, node.data.manifest.display_name)
    this.checkpoint()
    this.nodes.update((nodes) => [...nodes, copy])
    this.selectNode(copy.id)
    return copy.id
  }

  toggleNodeLocked(nodeId: string): void {
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node) return
    this.checkpoint()
    this.nodes.update((nodes) => nodes.map((item) => item.id === nodeId ? { ...item, locked: !item.locked } : item))
  }

  isOutput(nodeId: string): boolean {
    return this.outputBindings().some((binding) => binding.node_id === nodeId)
  }

  isInferredOutput(nodeId: string): boolean {
    if (this.outputBindings().length) return false
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node || node.data.manifest.semantic_role === 'input') return false
    return node.data.manifest.outputs.some((port) => {
      if (port.enabled_when && !conditionMatches(port.enabled_when, node.data.parameters)) return false
      if (!['tensor', 'tensor_list', 'mask', 'scalar'].includes(port.kind)) return false
      return !this.edges().some((edge) => edge.source === nodeId && (edge.sourceHandle ?? 'output') === port.id)
    })
  }

  isEffectiveOutput(nodeId: string): boolean {
    return this.isOutput(nodeId) || this.isInferredOutput(nodeId)
  }

  toggleNodeOutput(nodeId: string): void {
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node) return
    const existing = this.outputBindings().filter((binding) => binding.node_id === nodeId)
    if (existing.length) {
      this.checkpoint()
      this.outputBindings.update((bindings) => bindings.filter((binding) => binding.node_id !== nodeId))
      return
    }
    if (node.data.manifest.semantic_role === 'input') return
    const port = node.data.manifest.outputs.find((item) => (
      ['tensor', 'tensor_list', 'mask', 'scalar'].includes(item.kind)
      && (!item.enabled_when || conditionMatches(item.enabled_when, node.data.parameters))
      && item.id === 'output'
    )) ?? node.data.manifest.outputs.find((item) => (
      ['tensor', 'tensor_list', 'mask', 'scalar'].includes(item.kind)
      && (!item.enabled_when || conditionMatches(item.enabled_when, node.data.parameters))
    ))
    if (!port) return
    this.checkpoint()
    this.outputBindings.update((bindings) => [...bindings, { node_id: nodeId, port_id: port.id }])
  }

  disconnectNode(nodeId: string): void {
    if (!this.nodes().some((node) => node.id === nodeId)) return
    this.checkpoint()
    this.edges.update((edges) => edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId))
  }

  resetParameters(nodeId: string, executionBackend?: ExecutionBackend, kerasEnabled = true): void {
    const node = this.nodes().find((item) => item.id === nodeId)
    if (!node) return
    const selectedBackend = executionBackend ?? this.activeExecutionBackend()
    const selectedKeras = executionBackend === undefined ? this.activeKerasEnabled() : kerasEnabled
    const implementation = node.data.manifest.implementations[implementationKey(selectedBackend, selectedKeras)]
    const parameters = Object.fromEntries(
      node.data.manifest.parameters.map((item) => [
        item.id,
        structuredClone(frameworkDefault(item, implementation?.parameters?.[item.id])),
      ]),
    )
    this.checkpoint()
    this.nodes.update((nodes) => nodes.map((item) => item.id === nodeId ? {
      ...item,
      data: { ...item.data, parameters, shapeSummary: undefined },
    } : item))
  }

  undo(): void {
    const previous = this.past().at(-1)
    if (!previous) return
    this.future.set([this.snapshot(), ...this.future()].slice(0, 50))
    this.past.set(this.past().slice(0, -1))
    this.restoreSnapshot(previous)
    this.selectedNodeId.set(null)
    this.selectedEdgeId.set(null)
  }

  redo(): void {
    const next = this.future()[0]
    if (!next) return
    this.past.set([...this.past(), this.snapshot()].slice(-50))
    this.future.set(this.future().slice(1))
    this.restoreSnapshot(next)
    this.selectedNodeId.set(null)
    this.selectedEdgeId.set(null)
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
    this.outputBindings.set(structuredClone(project.graph.output_bindings))
    this.modelConfiguration.set(structuredClone(project.graph.model_configuration))
    this.exportSettings.set(structuredClone(project.graph.export_settings))
    this.futureTrainingConfiguration.set(structuredClone(project.graph.future_training_configuration))
    this.selectedNodeId.set(null)
    this.selectedEdgeId.set(null)
    this.copiedNodeIds.set([])
    this.past.set([])
    this.future.set([])
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
    this.outputBindings.set(cloned.outputBindings)
    this.modelConfiguration.set(cloned.modelConfiguration)
    this.exportSettings.set(cloned.exportSettings)
    this.futureTrainingConfiguration.set(cloned.futureTrainingConfiguration)
    this.selectedNodeId.set(cloned.selectedNodeId)
    this.selectedEdgeId.set(cloned.selectedEdgeId)
    this.copiedNodeIds.set(cloned.copiedNodeIds)
    this.past.set(cloned.past)
    this.future.set(cloned.future)
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
  }

  toGraph(): ModelGraph {
    return {
      schema_version: 5,
      nodes: this.nodes().map((node) => ({
        id: node.id,
        definition_id: node.data.manifest.id,
        definition_version: node.data.manifest.version,
        label: node.data.label,
        position: { ...node.position },
        parameters: structuredClone(node.data.parameters),
        locked: node.locked,
      })),
      edges: this.edges().map((edge) => ({
        id: edge.id,
        source_node_id: edge.source,
        source_port_id: edge.sourceHandle ?? 'output',
        target_node_id: edge.target,
        target_port_id: edge.targetHandle ?? 'input',
      })),
      output_bindings: structuredClone(this.outputBindings()),
      model_configuration: structuredClone(this.modelConfiguration()),
      export_settings: structuredClone(this.exportSettings()),
      future_training_configuration: structuredClone(this.futureTrainingConfiguration()),
    }
  }

  applyShapeSummaries(summaries: ShapeSummary[]): void {
    const shapeByNode = new Map(summaries.map((summary) => [summary.node_id, summary]))
    this.nodes.update((nodes) => nodes.map((node) => ({
      ...node,
      data: { ...node.data, shapeSummary: shapeByNode.get(node.id) },
    })))
  }

  updateConfiguration(configuration: ModelConfiguration): void {
    this.checkpoint()
    this.modelConfiguration.set(structuredClone(configuration))
  }

  autoLayout(nodes: EditorNode[], edges: EditorEdge[]): void {
    this.checkpoint()
    const positions = layeredLayout(nodes, edges)
    this.nodes.set(nodes.map((node) => ({
      ...node,
      position: node.locked ? { ...node.position } : positions.get(node.id) ?? { ...node.position },
    })))
    this.edges.set(edges.map((edge) => ({ ...edge })))
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
    this.outputBindings.set(structuredClone(snapshot.outputBindings))
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
}

function editorDefault(definition: NodeManifest['parameters'][number]): unknown {
  return definition.editor_default !== undefined ? definition.editor_default : definition.default
}

function frameworkDefault(
  definition: NodeManifest['parameters'][number],
  capability?: { default?: unknown },
): unknown {
  if (capability?.default !== undefined && capability.default !== null) return capability.default
  return Object.prototype.hasOwnProperty.call(definition, 'framework_default')
    ? definition.framework_default
    : definition.default
}

function newId(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`
}

function snapPosition(position: { x: number; y: number }): { x: number; y: number } {
  return { x: Math.round(position.x / 16) * 16, y: Math.round(position.y / 16) * 16 }
}

function conditionMatches(condition: Record<string, unknown>, parameters: Record<string, unknown>): boolean {
  return Object.entries(condition).every(([parameterId, expected]) => {
    const actual = parameters[parameterId]
    if (typeof expected === 'object' && expected !== null && !Array.isArray(expected)) {
      const rule = expected as Record<string, unknown>
      if ('equals' in rule && !sameValue(actual, rule['equals'])) return false
      if ('not_equals' in rule && sameValue(actual, rule['not_equals'])) return false
      if ('in' in rule && Array.isArray(rule['in']) && !rule['in'].some((item) => sameValue(actual, item))) return false
      if ('not_in' in rule && Array.isArray(rule['not_in']) && rule['not_in'].some((item) => sameValue(actual, item))) return false
      return true
    }
    if (Array.isArray(expected)) return expected.some((item) => sameValue(actual, item))
    return sameValue(actual, expected)
  })
}

function sameValue(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}
