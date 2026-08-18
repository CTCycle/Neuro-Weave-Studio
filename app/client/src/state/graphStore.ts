import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type EdgeChange,
  type NodeChange,
} from '@xyflow/react'
import { create } from 'zustand'
import type {
  EditorEdge,
  EditorNode,
  ModelConfiguration,
  ModelGraph,
  NodeManifest,
  ProjectDocument,
  ShapeSummary,
  ValidationDiagnostic,
} from '../domain/types'
import { editorNodeFromGraph, emptyModelConfiguration } from '../domain/types'

interface Snapshot {
  nodes: EditorNode[]
  edges: EditorEdge[]
  modelConfiguration: ModelConfiguration
  exportSettings: ModelGraph['export_settings']
  futureTrainingConfiguration: ModelGraph['future_training_configuration']
}

interface GraphState {
  nodes: EditorNode[]
  edges: EditorEdge[]
  manifests: NodeManifest[]
  modelConfiguration: ModelConfiguration
  exportSettings: ModelGraph['export_settings']
  futureTrainingConfiguration: ModelGraph['future_training_configuration']
  selectedNodeId: string | null
  selectedEdgeId: string | null
  copiedNodeIds: string[]
  past: Snapshot[]
  future: Snapshot[]
  setManifests: (manifests: NodeManifest[]) => void
  checkpoint: () => void
  onNodesChange: (changes: NodeChange<EditorNode>[]) => void
  onEdgesChange: (changes: EdgeChange<EditorEdge>[]) => void
  addNode: (manifest: NodeManifest, position: { x: number; y: number }) => void
  connect: (connection: Connection) => boolean
  isConnectionValid: (connection: Connection) => boolean
  selectNode: (nodeId: string | null) => void
  selectEdge: (edgeId: string | null) => void
  updateNode: (nodeId: string, patch: { label?: string; parameters?: Record<string, unknown> }) => void
  deleteSelected: () => void
  copySelected: () => void
  pasteCopied: () => void
  undo: () => void
  redo: () => void
  replaceGraph: (project: ProjectDocument) => void
  reset: () => void
  toGraph: () => ModelGraph
  applyDiagnostics: (diagnostics: ValidationDiagnostic[], summaries: ShapeSummary[]) => void
  updateConfiguration: (configuration: Partial<ModelConfiguration>) => void
  autoLayout: (nodes: EditorNode[], edges: EditorEdge[]) => void
}

const newId = (prefix: string) => `${prefix}-${crypto.randomUUID()}`
const cloneNodes = (nodes: EditorNode[]) => nodes.map((node) => ({ ...node, position: { ...node.position }, data: { ...node.data, parameters: { ...node.data.parameters }, diagnostics: [...node.data.diagnostics] } }))
const cloneEdges = (edges: EditorEdge[]) => edges.map((edge) => ({ ...edge }))
const defaultExportSettings = (): ModelGraph['export_settings'] => ({
  include_comments: true,
  include_shape_assertions: true,
  preferred_binary_format: 'keras',
})

const snapshot = (state: Pick<GraphState, 'nodes' | 'edges' | 'modelConfiguration' | 'exportSettings' | 'futureTrainingConfiguration'>): Snapshot => ({
  nodes: cloneNodes(state.nodes),
  edges: cloneEdges(state.edges),
  modelConfiguration: structuredClone(state.modelConfiguration),
  exportSettings: structuredClone(state.exportSettings),
  futureTrainingConfiguration: structuredClone(state.futureTrainingConfiguration),
})

function wouldCreateCycle(nodes: EditorNode[], edges: EditorEdge[], source: string, target: string): boolean {
  if (source === target) return true
  const adjacency = new Map<string, string[]>()
  for (const node of nodes) adjacency.set(node.id, [])
  for (const edge of edges) adjacency.get(edge.source)?.push(edge.target)
  const stack = [target]
  const visited = new Set<string>()
  while (stack.length) {
    const current = stack.pop()!
    if (current === source) return true
    if (visited.has(current)) continue
    visited.add(current)
    stack.push(...(adjacency.get(current) ?? []))
  }
  return false
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],
  manifests: [],
  modelConfiguration: emptyModelConfiguration(),
  exportSettings: defaultExportSettings(),
  futureTrainingConfiguration: {},
  selectedNodeId: null,
  selectedEdgeId: null,
  copiedNodeIds: [],
  past: [],
  future: [],
  setManifests: (manifests) => set({ manifests }),
  checkpoint: () => set((state) => ({ past: [...state.past, snapshot(state)].slice(-50), future: [] })),
  onNodesChange: (changes) => set((state) => ({ nodes: applyNodeChanges(changes, state.nodes) })),
  onEdgesChange: (changes) => set((state) => ({ edges: applyEdgeChanges(changes, state.edges) })),
  addNode: (manifest, position) => set((state) => {
    const parameters = Object.fromEntries(manifest.parameters.map((item) => [item.id, structuredClone(item.default)]))
    const node: EditorNode = {
      id: newId(manifest.id.replaceAll('.', '-')),
      type: 'modelNode',
      position,
      data: { manifest, parameters, label: manifest.display_name, diagnostics: [] },
    }
    return { past: [...state.past, snapshot(state)].slice(-50), future: [], nodes: [...state.nodes, node], selectedNodeId: node.id, selectedEdgeId: null }
  }),
  isConnectionValid: (connection) => {
    if (!connection.source || !connection.target || !connection.sourceHandle || !connection.targetHandle) return false
    const state = get()
    if (wouldCreateCycle(state.nodes, state.edges, connection.source, connection.target)) return false
    const sourceNode = state.nodes.find((node) => node.id === connection.source)
    const targetNode = state.nodes.find((node) => node.id === connection.target)
    const sourcePort = sourceNode?.data.manifest.outputs.find((port) => port.id === connection.sourceHandle)
    const targetPort = targetNode?.data.manifest.inputs.find((port) => port.id === connection.targetHandle)
    if (!sourcePort || !targetPort) return false
    const expectedKind = targetPort.kind === 'tensor_list' ? targetPort.element_kind ?? 'tensor' : targetPort.kind
    if (sourcePort.kind !== expectedKind) return false
    if (!targetPort.accepts_multiple && state.edges.some((edge) => edge.target === connection.target && edge.targetHandle === connection.targetHandle)) return false
    return true
  },
  connect: (connection) => {
    if (!get().isConnectionValid(connection)) return false
    set((state) => ({
      past: [...state.past, snapshot(state)].slice(-50),
      future: [],
      edges: addEdge({ ...connection, id: newId('edge'), type: 'smoothstep' }, state.edges),
    }))
    return true
  },
  selectNode: (selectedNodeId) => set({ selectedNodeId, selectedEdgeId: null }),
  selectEdge: (selectedEdgeId) => set({ selectedEdgeId, selectedNodeId: null }),
  updateNode: (nodeId, patch) => set((state) => ({
    past: [...state.past, snapshot(state)].slice(-50),
    future: [],
    nodes: state.nodes.map((node) => node.id === nodeId ? {
      ...node,
      data: {
        ...node.data,
        label: patch.label ?? node.data.label,
        parameters: patch.parameters ?? node.data.parameters,
      },
    } : node),
  })),
  deleteSelected: () => set((state) => {
    if (!state.selectedNodeId && !state.selectedEdgeId) return state
    if (state.selectedEdgeId) {
      return {
        past: [...state.past, snapshot(state)].slice(-50),
        future: [],
        edges: state.edges.filter((edge) => edge.id !== state.selectedEdgeId),
        selectedEdgeId: null,
      }
    }
    const id = state.selectedNodeId!
    return {
      past: [...state.past, snapshot(state)].slice(-50),
      future: [],
      nodes: state.nodes.filter((node) => node.id !== id),
      edges: state.edges.filter((edge) => edge.source !== id && edge.target !== id),
      selectedNodeId: null,
    }
  }),
  copySelected: () => set((state) => ({ copiedNodeIds: state.selectedNodeId ? [state.selectedNodeId] : [] })),
  pasteCopied: () => set((state) => {
    const selected = state.nodes.filter((node) => state.copiedNodeIds.includes(node.id))
    if (!selected.length) return state
    const copies = selected.map((node) => ({
      ...structuredClone(node),
      id: newId(node.data.manifest.id.replaceAll('.', '-')),
      position: { x: node.position.x + 32, y: node.position.y + 32 },
      selected: false,
    }))
    return { past: [...state.past, snapshot(state)].slice(-50), future: [], nodes: [...state.nodes, ...copies], selectedNodeId: copies[0].id, selectedEdgeId: null }
  }),
  undo: () => set((state) => {
    const previous = state.past.at(-1)
    if (!previous) return state
    return { ...previous, past: state.past.slice(0, -1), future: [snapshot(state), ...state.future].slice(0, 50), selectedNodeId: null, selectedEdgeId: null }
  }),
  redo: () => set((state) => {
    const next = state.future[0]
    if (!next) return state
    return { ...next, past: [...state.past, snapshot(state)].slice(-50), future: state.future.slice(1), selectedNodeId: null, selectedEdgeId: null }
  }),
  replaceGraph: (project) => set((state) => {
    const manifests = new Map(state.manifests.map((manifest) => [`${manifest.id}@${manifest.version}`, manifest]))
    const nodes = project.graph.nodes.map((node) => {
      const manifest = manifests.get(`${node.definition_id}@${node.definition_version}`)
      if (!manifest) throw new Error(`Missing node manifest ${node.definition_id}@${node.definition_version}`)
      return editorNodeFromGraph(node, manifest)
    })
    const edges: EditorEdge[] = project.graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source_node_id,
      sourceHandle: edge.source_port_id,
      target: edge.target_node_id,
      targetHandle: edge.target_port_id,
      type: 'smoothstep',
    }))
    return {
      nodes,
      edges,
      modelConfiguration: project.graph.model_configuration,
      exportSettings: structuredClone(project.graph.export_settings),
      futureTrainingConfiguration: structuredClone(project.graph.future_training_configuration),
      selectedNodeId: null,
      selectedEdgeId: null,
      past: [],
      future: [],
    }
  }),
  reset: () => set({ nodes: [], edges: [], modelConfiguration: emptyModelConfiguration(), exportSettings: defaultExportSettings(), futureTrainingConfiguration: {}, selectedNodeId: null, selectedEdgeId: null, past: [], future: [] }),
  toGraph: () => {
    const state = get()
    return {
      schema_version: 2,
      nodes: state.nodes.map((node) => ({
        id: node.id,
        definition_id: node.data.manifest.id,
        definition_version: node.data.manifest.version,
        label: node.data.label,
        position: node.position,
        parameters: node.data.parameters,
      })),
      edges: state.edges.map((edge) => ({
        id: edge.id,
        source_node_id: edge.source,
        source_port_id: edge.sourceHandle ?? 'output',
        target_node_id: edge.target,
        target_port_id: edge.targetHandle ?? 'input',
      })),
      model_configuration: state.modelConfiguration,
      export_settings: structuredClone(state.exportSettings),
      future_training_configuration: structuredClone(state.futureTrainingConfiguration),
    }
  },
  applyDiagnostics: (diagnostics, summaries) => set((state) => {
    const byNode = new Map<string, ValidationDiagnostic[]>()
    diagnostics.forEach((diagnostic) => {
      if (diagnostic.node_id) byNode.set(diagnostic.node_id, [...(byNode.get(diagnostic.node_id) ?? []), diagnostic])
    })
    const shapeByNode = new Map(summaries.map((summary) => [summary.node_id, summary]))
    return {
      nodes: state.nodes.map((node) => ({
        ...node,
        data: { ...node.data, diagnostics: byNode.get(node.id) ?? [], shapeSummary: shapeByNode.get(node.id) },
      })),
    }
  }),
  updateConfiguration: (configuration) => set((state) => ({
    past: [...state.past, snapshot(state)].slice(-50),
    future: [],
    modelConfiguration: { ...state.modelConfiguration, ...configuration },
  })),
  autoLayout: (nodes, edges) => set((state) => ({ nodes, edges, past: [...state.past, snapshot(state)].slice(-50), future: [] })),
}))

export { wouldCreateCycle }
