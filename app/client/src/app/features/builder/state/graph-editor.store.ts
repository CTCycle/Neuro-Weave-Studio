import { Injectable, computed, signal } from '@angular/core'
import type { EditorEdge, EditorNode } from '../../../domain/editor.types'
import {
  cloneEditorSession,
  defaultExportSettings,
  editorNodeFromGraph,
  type EditorHistorySnapshot,
  type GraphEditorSession,
} from '../../../domain/graph-mapper'
import { emptyModelConfiguration, type ModelConfiguration, type ModelGraph, type NodeManifest, type ProjectDocument, type ShapeSummary } from '../../../domain/types'
import { layeredLayout } from '../graph/graph-layout.service'
import { isGraphConnectionValid, type GraphConnection } from '../graph/graph-validation'

export type { GraphConnection } from '../graph/graph-validation'
export { wouldCreateCycle } from '../graph/graph-validation'

export interface NodePositionChange {
  id: string
  position: { x: number; y: number }
}

@Injectable({ providedIn: 'root' })
export class GraphEditorStore {
  readonly nodes = signal<EditorNode[]>([])
  readonly edges = signal<EditorEdge[]>([])
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

  private dragging = false

  setManifests(manifests: NodeManifest[]): void {
    this.manifests.set(manifests)
  }

  checkpoint(): void {
    this.past.set([...this.past(), this.snapshot()].slice(-50))
    this.future.set([])
  }

  beginDrag(): void {
    if (this.dragging) return
    this.dragging = true
    this.checkpoint()
  }

  endDrag(): void {
    this.dragging = false
  }

  moveNodes(changes: readonly NodePositionChange[]): void {
    if (!changes.length) return
    const positions = new Map(changes.map((change) => [change.id, change.position]))
    this.nodes.update((nodes) => nodes.map((node) => {
      const position = positions.get(node.id)
      return position ? { ...node, position: { ...position } } : node
    }))
  }

  addNode(manifest: NodeManifest, position: { x: number; y: number }): string {
    const node: EditorNode = {
      id: newId(manifest.id.replaceAll('.', '-')),
      type: 'modelNode',
      position: snapPosition(position),
      data: {
        manifest,
        parameters: Object.fromEntries(manifest.parameters.map((item) => [item.id, structuredClone(item.default)])),
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
    return isGraphConnectionValid(this.nodes(), this.edges(), connection)
  }

  connect(connection: GraphConnection): boolean {
    if (!this.isConnectionValid(connection)) return false
    this.checkpoint()
    this.edges.update((edges) => [...edges, {
      ...connection,
      id: newId('edge'),
      type: 'smoothstep',
    }])
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

  updateNode(nodeId: string, patch: { label?: string; parameters?: Record<string, unknown> }): void {
    this.checkpoint()
    this.nodes.update((nodes) => nodes.map((node) => node.id === nodeId ? {
      ...node,
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
      type: 'smoothstep',
    })))
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
      schema_version: 3,
      nodes: this.nodes().map((node) => ({
        id: node.id,
        definition_id: node.data.manifest.id,
        definition_version: node.data.manifest.version,
        label: node.data.label,
        position: { ...node.position },
        parameters: structuredClone(node.data.parameters),
      })),
      edges: this.edges().map((edge) => ({
        id: edge.id,
        source_node_id: edge.source,
        source_port_id: edge.sourceHandle ?? 'output',
        target_node_id: edge.target,
        target_port_id: edge.targetHandle ?? 'input',
      })),
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
    this.nodes.set(nodes.map((node) => ({ ...node, position: positions.get(node.id) ?? { ...node.position } })))
    this.edges.set(edges.map((edge) => ({ ...edge })))
  }

  private snapshot(): EditorHistorySnapshot {
    return {
      nodes: this.nodes().map((node) => ({ ...node, position: { ...node.position }, data: { ...node.data, parameters: structuredClone(node.data.parameters) } })),
      edges: this.edges().map((edge) => ({ ...edge })),
      modelConfiguration: structuredClone(this.modelConfiguration()),
      exportSettings: structuredClone(this.exportSettings()),
      futureTrainingConfiguration: structuredClone(this.futureTrainingConfiguration()),
    }
  }

  private restoreSnapshot(snapshot: EditorHistorySnapshot): void {
    this.nodes.set(snapshot.nodes.map((node) => ({ ...node, position: { ...node.position }, data: { ...node.data, parameters: structuredClone(node.data.parameters) } })))
    this.edges.set(snapshot.edges.map((edge) => ({ ...edge })))
    this.modelConfiguration.set(structuredClone(snapshot.modelConfiguration))
    this.exportSettings.set(structuredClone(snapshot.exportSettings))
    this.futureTrainingConfiguration.set(structuredClone(snapshot.futureTrainingConfiguration))
  }
}

function newId(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`
}

function snapPosition(position: { x: number; y: number }): { x: number; y: number } {
  return { x: Math.round(position.x / 16) * 16, y: Math.round(position.y / 16) * 16 }
}
