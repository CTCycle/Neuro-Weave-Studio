import { useEffect, useMemo } from 'react'
import { Background, BackgroundVariant, Controls, MiniMap, ReactFlow, ReactFlowProvider, useReactFlow } from '@xyflow/react'
import type { ModelGraph, NodeManifest } from '../domain/types'
import { editorNodeFromGraph } from '../domain/types'
import { ModelNode } from './ModelNode'

const nodeTypes = { modelNode: ModelNode }

function PreviewInner({ graph, manifests }: { graph: ModelGraph; manifests: NodeManifest[] }) {
  const { fitView } = useReactFlow()
  const manifestMap = useMemo(() => new Map(manifests.map((manifest) => [`${manifest.id}@${manifest.version}`, manifest])), [manifests])
  const nodes = useMemo(() => graph.nodes.flatMap((node) => {
    const manifest = manifestMap.get(`${node.definition_id}@${node.definition_version}`)
    return manifest ? [editorNodeFromGraph(node, manifest)] : []
  }), [graph.nodes, manifestMap])
  const edges = useMemo(() => graph.edges.map((edge) => ({ id: edge.id, source: edge.source_node_id, sourceHandle: edge.source_port_id, target: edge.target_node_id, targetHandle: edge.target_port_id, type: 'smoothstep' })), [graph.edges])
  useEffect(() => { requestAnimationFrame(() => fitView({ padding: 0.18, duration: 0 })) }, [fitView, nodes.length, edges.length])
  return <div className="canvas-shell model-preview"><ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} nodesDraggable={false} nodesConnectable={false} elementsSelectable={false} zoomOnDoubleClick={false} fitView minZoom={0.2} maxZoom={2.5} proOptions={{ hideAttribution: true }}><Background variant={BackgroundVariant.Dots} gap={24} size={1.2} /><MiniMap zoomable={false} pannable={false} nodeColor="#315866" nodeStrokeColor="#82d3d9" nodeStrokeWidth={2} maskColor="rgba(5, 10, 14, .72)" /><Controls showInteractive={false} /></ReactFlow>{nodes.length === 0 && <div className="preview-empty">No renderable nodes in this revision.</div>}</div>
}

export function ReadOnlyGraphPreview({ graph, manifests }: { graph: ModelGraph; manifests: NodeManifest[] }) {
  return <ReactFlowProvider><PreviewInner graph={graph} manifests={manifests} /></ReactFlowProvider>
}
