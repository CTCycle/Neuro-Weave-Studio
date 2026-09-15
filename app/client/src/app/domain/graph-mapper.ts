import type { EditorEdge, EditorNode, EditorNodeData } from './editor.types'
import type { GraphNodeInstance, ModelGraph, NodeManifest, OutputBinding, ProjectDocument } from './types'

export interface GraphEditorSession {
  nodes: EditorNode[]
  edges: EditorEdge[]
  outputBindings: OutputBinding[]
  modelConfiguration: ModelGraph['model_configuration']
  exportSettings: ModelGraph['export_settings']
  futureTrainingConfiguration: ModelGraph['future_training_configuration']
  selectedNodeId: string | null
  selectedEdgeId: string | null
  copiedNodeIds: string[]
  past: EditorHistorySnapshot[]
  future: EditorHistorySnapshot[]
}

export interface EditorHistorySnapshot {
  nodes: EditorNode[]
  edges: EditorEdge[]
  outputBindings: OutputBinding[]
  modelConfiguration: ModelGraph['model_configuration']
  exportSettings: ModelGraph['export_settings']
  futureTrainingConfiguration: ModelGraph['future_training_configuration']
}

export const defaultExportSettings = (): ModelGraph['export_settings'] => ({
  include_comments: true,
  include_shape_assertions: true,
  preferred_binary_format: 'keras',
})

export function editorNodeFromGraph(node: GraphNodeInstance, manifest: NodeManifest): EditorNode {
  return {
    id: node.id,
    type: 'modelNode',
    position: { ...node.position },
    locked: node.locked,
    data: {
      manifest,
      parameters: structuredClone(node.parameters),
      label: node.label ?? manifest.display_name,
    },
  }
}

export function editorEdgeFromGraph(edge: ModelGraph['edges'][number]): EditorEdge {
  return {
    id: edge.id,
    source: edge.source_node_id,
    sourceHandle: edge.source_port_id,
    target: edge.target_node_id,
    targetHandle: edge.target_port_id,
    type: 'bezier',
  }
}

export function graphEditorSessionFromGraph(
  graph: ModelGraph,
  manifests: readonly NodeManifest[],
): GraphEditorSession {
  const manifestMap = new Map(manifests.map((manifest) => [`${manifest.id}@${manifest.version}`, manifest]))
  const nodes = graph.nodes.map((node) => {
    const manifest = manifestMap.get(`${node.definition_id}@${node.definition_version}`)
    if (!manifest) throw new Error(`Missing node manifest ${node.definition_id}@${node.definition_version}`)
    return editorNodeFromGraph(node, manifest)
  })
  return {
    nodes,
    edges: graph.edges.map(editorEdgeFromGraph),
    outputBindings: structuredClone(graph.output_bindings),
    modelConfiguration: structuredClone(graph.model_configuration),
    exportSettings: structuredClone(graph.export_settings),
    futureTrainingConfiguration: structuredClone(graph.future_training_configuration),
    selectedNodeId: null,
    selectedEdgeId: null,
    copiedNodeIds: [],
    past: [],
    future: [],
  }
}

export function projectDocumentToSession(
  project: ProjectDocument,
  manifests: readonly NodeManifest[],
): GraphEditorSession {
  return graphEditorSessionFromGraph(project.graph, manifests)
}

export function sessionToGraph(session: GraphEditorSession): ModelGraph {
  return {
    schema_version: 5,
    nodes: session.nodes.map((node) => ({
      id: node.id,
      definition_id: node.data.manifest.id,
      definition_version: node.data.manifest.version,
      label: node.data.label,
      position: { ...node.position },
      parameters: structuredClone(node.data.parameters),
      locked: node.locked,
    })),
    output_bindings: structuredClone(session.outputBindings),
    edges: session.edges.map((edge) => ({
      id: edge.id,
      source_node_id: edge.source,
      source_port_id: edge.sourceHandle ?? 'output',
      target_node_id: edge.target,
      target_port_id: edge.targetHandle ?? 'input',
    })),
    model_configuration: structuredClone(session.modelConfiguration),
    export_settings: structuredClone(session.exportSettings),
    future_training_configuration: structuredClone(session.futureTrainingConfiguration),
  }
}

export function cloneEditorNode(node: EditorNode): EditorNode {
  const data: EditorNodeData = {
    manifest: node.data.manifest,
    label: node.data.label,
    parameters: structuredClone(node.data.parameters),
  }
  return { ...node, position: { ...node.position }, data }
}

export function cloneEditorEdge(edge: EditorEdge): EditorEdge {
  return { ...edge }
}

export function cloneHistory(history: readonly EditorHistorySnapshot[]): EditorHistorySnapshot[] {
  return history.map((entry) => ({
    nodes: entry.nodes.map(cloneEditorNode),
    edges: entry.edges.map(cloneEditorEdge),
    outputBindings: structuredClone(entry.outputBindings),
    modelConfiguration: structuredClone(entry.modelConfiguration),
    exportSettings: structuredClone(entry.exportSettings),
    futureTrainingConfiguration: structuredClone(entry.futureTrainingConfiguration),
  }))
}

export function cloneEditorSession(session: GraphEditorSession): GraphEditorSession {
  return {
    nodes: session.nodes.map(cloneEditorNode),
    edges: session.edges.map(cloneEditorEdge),
    outputBindings: structuredClone(session.outputBindings),
    modelConfiguration: structuredClone(session.modelConfiguration),
    exportSettings: structuredClone(session.exportSettings),
    futureTrainingConfiguration: structuredClone(session.futureTrainingConfiguration),
    selectedNodeId: session.selectedNodeId,
    selectedEdgeId: session.selectedEdgeId,
    copiedNodeIds: [...session.copiedNodeIds],
    past: cloneHistory(session.past),
    future: cloneHistory(session.future),
  }
}
