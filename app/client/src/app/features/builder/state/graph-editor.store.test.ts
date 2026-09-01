import { afterEach, describe, expect, it } from 'vitest'
import { emptyModelConfiguration } from '../../../domain/types'
import type { EditorEdge, EditorNode } from '../../../domain/editor.types'
import { GraphEditorStore, wouldCreateCycle } from './graph-editor.store'

const manifest = {
  id: 'layer.dense',
  version: 1,
  display_name: 'Dense',
  category: 'core',
  description: '',
  semantic_role: 'operation' as const,
  inputs: [],
  outputs: [{ id: 'output', display_name: 'Output', kind: 'tensor' as const, required: true, accepts_multiple: false, description: '' }],
  parameters: [{ id: 'units', display_name: 'Units', type: 'integer' as const, default: 32, required: false, constraints: {}, options: [], description: '' }],
  shape_rule: { kind: 'dense', config: {} },
  implementations: { keras: { supported: true } },
  ui: { icon: 'Box', accent: 'violet', width: 250, height: 120 },
  deprecated: false,
  documentation_ref: '',
}

const node = (id: string): EditorNode => ({ id, type: 'modelNode', position: { x: 0, y: 0 }, locked: false, data: { manifest, parameters: {}, label: id } })

describe('neutral graph editor store', () => {
  let store: GraphEditorStore

  afterEach(() => store?.reset())

  it('rejects an edge that closes a cycle', () => {
    const nodes = [node('a'), node('b'), node('c')]
    const edges: EditorEdge[] = [{ id: 'ab', source: 'a', target: 'b' }, { id: 'bc', source: 'b', target: 'c' }]
    expect(wouldCreateCycle(nodes, edges, 'c', 'a')).toBe(true)
    expect(wouldCreateCycle(nodes, edges, 'a', 'c')).toBe(false)
  })

  it('preserves export and future-training metadata across undo and redo', () => {
    store = new GraphEditorStore()
    store.restoreEditorSession({ nodes: [], edges: [], outputBindings: [], modelConfiguration: emptyModelConfiguration(), exportSettings: { include_comments: false, include_shape_assertions: false, preferred_binary_format: 'keras' }, futureTrainingConfiguration: { dataset: 'deferred' }, selectedNodeId: null, selectedEdgeId: null, copiedNodeIds: [], past: [], future: [] })
    store.updateConfiguration({ ...emptyModelConfiguration(), keras: { ...emptyModelConfiguration().keras, optimizer: 'sgd' } })
    expect(store.toGraph().export_settings.include_comments).toBe(false)
    expect(store.toGraph().future_training_configuration).toEqual({ dataset: 'deferred' })
    store.undo()
    expect(store.toGraph().model_configuration.keras.optimizer).toBe('adam')
    store.redo()
    expect(store.toGraph().model_configuration.keras.optimizer).toBe('sgd')
  })

  it('captures and restores a detached session with selection and history', () => {
    store = new GraphEditorStore()
    store.setManifests([manifest])
    store.addNode(manifest, { x: 40, y: 60 })
    const nodeId = store.nodes()[0].id
    store.selectNode(nodeId)
    const captured = store.captureEditorSession()
    store.reset()
    store.restoreEditorSession(captured)
    captured.nodes[0].data.label = 'mutated outside the store'
    expect(store.nodes()[0].data.label).toBe('Dense')
    expect(store.selectedNodeId()).toBe(nodeId)
    expect(store.past()).toHaveLength(1)
  })

  it('deletes a selected edge without deleting its endpoint nodes', () => {
    store = new GraphEditorStore()
    store.restoreEditorSession({
      nodes: [node('source'), node('target')],
      edges: [{ id: 'source-target', source: 'source', sourceHandle: 'output', target: 'target', targetHandle: 'input', type: 'smoothstep' }],
      outputBindings: [],
      modelConfiguration: emptyModelConfiguration(),
      exportSettings: { include_comments: true, include_shape_assertions: true, preferred_binary_format: 'keras' },
      futureTrainingConfiguration: {},
      selectedNodeId: null,
      selectedEdgeId: null,
      copiedNodeIds: [],
      past: [],
      future: [],
    })

    store.selectEdge('source-target')
    store.deleteSelected()

    expect(store.edges()).toEqual([])
    expect(store.nodes().map((item) => item.id)).toEqual(['source', 'target'])
    expect(store.selectedEdgeId()).toBeNull()
    store.undo()
    expect(store.edges().map((edge) => edge.id)).toEqual(['source-target'])
  })

  it('duplicates exact configuration and creates a detached default copy', () => {
    store = new GraphEditorStore()
    store.setManifests([manifest])
    store.addNode(manifest, { x: 40, y: 60 })
    const originalId = store.nodes()[0].id
    store.updateNode(originalId, { label: 'Configured', parameters: { units: 128 } })

    const exactId = store.duplicateExact(originalId)
    const defaultsId = store.duplicateWithDefaults(originalId)

    expect(exactId).not.toBe(originalId)
    expect(defaultsId).not.toBe(originalId)
    expect(defaultsId).not.toBe(exactId)
    expect(store.nodes().find((node) => node.id === exactId)?.data.parameters).toEqual({ units: 128 })
    expect(store.nodes().find((node) => node.id === exactId)?.data.label).toBe('Configured')
    expect(store.nodes().find((node) => node.id === defaultsId)?.data.parameters).toEqual({ units: 32 })
    expect(store.edges()).toEqual([])
  })

  it('locks a node against movement while preserving configuration and output state', () => {
    store = new GraphEditorStore()
    store.setManifests([manifest])
    store.addNode(manifest, { x: 40, y: 60 })
    const nodeId = store.nodes()[0].id
    store.toggleNodeLocked(nodeId)
    store.moveNodes([{ id: nodeId, position: { x: 240, y: 260 } }])
    expect(store.nodes()[0].locked).toBe(true)
    expect(store.nodes()[0].position).toEqual({ x: 48, y: 64 })

    store.toggleNodeOutput(nodeId)
    expect(store.isOutput(nodeId)).toBe(true)
    expect(store.isEffectiveOutput(nodeId)).toBe(true)
    expect(store.toGraph().output_bindings).toEqual([{ node_id: nodeId, port_id: 'output' }])
    expect(store.toGraph().nodes[0].locked).toBe(true)

    store.deleteSelected()
    store.selectNode(nodeId)
    store.deleteSelected()
    expect(store.toGraph().output_bindings).toEqual([])
  })
})
