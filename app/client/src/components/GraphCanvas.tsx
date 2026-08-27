import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react'
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Connection,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { Map as MapIcon, Maximize2, Minimize2, Save } from 'lucide-react'
import { ModelNode } from './ModelNode'
import type { EditorEdge } from '../domain/types'
import { useGraphStore } from '../state/graphStore'

const nodeTypes = { modelNode: ModelNode }

interface Props {
  fullCanvas: boolean
  modelName: string
  dirty: boolean
  busy: boolean
  onEnterFocus: () => void
  onExitFocus: () => void
  onSave: () => void
}

function CanvasInner({ fullCanvas, modelName, dirty, busy, onEnterFocus, onExitFocus, onSave }: Props) {
  const wrapper = useRef<HTMLDivElement>(null)
  const { screenToFlowPosition, fitView } = useReactFlow()
  const [minimapVisible, setMinimapVisible] = useState(true)
  const nodes = useGraphStore((state) => state.nodes)
  const edges = useGraphStore((state) => state.edges)
  const manifests = useGraphStore((state) => state.manifests)
  const onNodesChange = useGraphStore((state) => state.onNodesChange)
  const onEdgesChange = useGraphStore((state) => state.onEdgesChange)
  const addNode = useGraphStore((state) => state.addNode)
  const connect = useGraphStore((state) => state.connect)
  const isConnectionValid = useGraphStore((state) => state.isConnectionValid)
  const selectNode = useGraphStore((state) => state.selectNode)
  const selectEdge = useGraphStore((state) => state.selectEdge)
  const applyLayout = useGraphStore((state) => state.autoLayout)
  const checkpoint = useGraphStore((state) => state.checkpoint)
  const flowConnectionValidation = useCallback(
    (connection: Connection | EditorEdge) => isConnectionValid({
      source: connection.source,
      target: connection.target,
      sourceHandle: connection.sourceHandle ?? null,
      targetHandle: connection.targetHandle ?? null,
    }),
    [isConnectionValid],
  )

  const onDrop = useCallback((event: DragEvent) => {
    event.preventDefault()
    const definitionId = event.dataTransfer.getData('application/neuro-weave-node')
    const manifest = manifests.find((item) => item.id === definitionId)
    if (!manifest) return
    addNode(manifest, screenToFlowPosition({ x: event.clientX, y: event.clientY }))
  }, [addNode, manifests, screenToFlowPosition])

  const autoLayout = useCallback(() => {
    const incoming = new Map(nodes.map((node) => [node.id, 0]))
    const outgoing = new Map(nodes.map((node) => [node.id, [] as string[]]))
    edges.forEach((edge) => {
      incoming.set(edge.target, (incoming.get(edge.target) ?? 0) + 1)
      outgoing.get(edge.source)?.push(edge.target)
    })
    const queue = nodes.filter((node) => incoming.get(node.id) === 0).map((node) => node.id)
    const layer = new Map(queue.map((id) => [id, 0]))
    while (queue.length) {
      const source = queue.shift()!
      for (const target of outgoing.get(source) ?? []) {
        layer.set(target, Math.max(layer.get(target) ?? 0, (layer.get(source) ?? 0) + 1))
        incoming.set(target, (incoming.get(target) ?? 1) - 1)
        if (incoming.get(target) === 0) queue.push(target)
      }
    }
    const rows = new Map<number, number>()
    const laidOut = nodes.map((node) => {
      const column = layer.get(node.id) ?? 0
      const row = rows.get(column) ?? 0
      rows.set(column, row + 1)
      return { ...node, position: { x: column * 360, y: row * 180 } }
    })
    applyLayout(laidOut, edges)
    requestAnimationFrame(() => fitView({ padding: 0.2, duration: 300 }))
  }, [applyLayout, edges, fitView, nodes])

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const state = useGraphStore.getState()
      const command = event.ctrlKey || event.metaKey
      if (command && event.key.toLowerCase() === 'z') {
        event.preventDefault()
        if (event.shiftKey) state.redo()
        else state.undo()
      }
      if (command && event.key.toLowerCase() === 'y') {
        event.preventDefault()
        state.redo()
      }
      if (command && event.key.toLowerCase() === 'c') state.copySelected()
      if (command && event.key.toLowerCase() === 'v') {
        event.preventDefault()
        state.pasteCopied()
      }
      if (
        (event.key === 'Delete' || event.key === 'Backspace') &&
        !(event.target instanceof HTMLInputElement) &&
        !(event.target instanceof HTMLTextAreaElement)
      ) {
        state.deleteSelected()
      }
      if (command && event.key.toLowerCase() === 'l') {
        event.preventDefault()
        autoLayout()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [autoLayout])

  return (
    <div ref={wrapper} className="canvas-shell" onDrop={onDrop} onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={(connection: Connection) => connect(connection)}
        isValidConnection={flowConnectionValidation}
        onNodeClick={(_, node) => selectNode(node.id)}
        onNodeDragStart={checkpoint}
        onEdgeClick={(_, edge) => selectEdge(edge.id)}
        onPaneClick={() => { selectNode(null); selectEdge(null) }}
        fitView
        snapToGrid
        snapGrid={[16, 16]}
        deleteKeyCode={null}
        minZoom={0.2}
        maxZoom={2.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={24} size={1.2} />
        {minimapVisible && <MiniMap zoomable pannable nodeColor="#315866" nodeStrokeColor="#82d3d9" nodeStrokeWidth={2} maskColor="rgba(5, 10, 14, .72)" />}
        <Controls showInteractive={false} />
        <div className={`canvas-editor-tools ${fullCanvas ? 'is-focus-mode' : ''}`} role="toolbar" aria-label="Graph editor controls">
          <button type="button" className="canvas-tool" onClick={() => setMinimapVisible((visible) => !visible)} aria-label={minimapVisible ? 'Hide minimap' : 'Show minimap'} title={minimapVisible ? 'Hide minimap' : 'Show minimap'}><MapIcon size={14} /><span>{minimapVisible ? 'Hide map' : 'Show map'}</span></button>
          {fullCanvas ? <div className="canvas-focus-bar"><span className="canvas-focus-model">{modelName}</span><span className={`canvas-focus-draft ${dirty ? 'is-dirty' : ''}`}>{busy ? 'Saving…' : dirty ? 'Draft' : 'Saved'}</span><button type="button" className="canvas-tool canvas-focus-save" onClick={onSave} disabled={busy} title="Save model"><Save size={14} /><span>Save</span></button><button type="button" className="canvas-tool canvas-focus-exit" onClick={onExitFocus} title="Exit full canvas"><Minimize2 size={14} /><span>Exit full canvas</span></button></div> : <button type="button" className="canvas-tool" onClick={onEnterFocus} aria-label="Enter full canvas" title="Enter full canvas"><Maximize2 size={14} /><span>Focus</span></button>}
        </div>
        <div className="canvas-hint">Drop a node to build the model • Ctrl+L auto-layout</div>
      </ReactFlow>
    </div>
  )
}

export function GraphCanvas(props: Props) {
  return <ReactFlowProvider><CanvasInner {...props} /></ReactFlowProvider>
}
