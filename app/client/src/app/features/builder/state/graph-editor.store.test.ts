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
  inputs: [],
  outputs: [],
  parameters: [],
  shape_rule: { kind: 'dense', config: {} },
  implementations: { keras: { supported: true } },
  ui: { icon: 'Box', accent: 'violet', width: 250, height: 120 },
  deprecated: false,
  documentation_ref: '',
}

const node = (id: string): EditorNode => ({ id, type: 'modelNode', position: { x: 0, y: 0 }, data: { manifest, parameters: {}, label: id } })

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
    store.restoreEditorSession({ nodes: [], edges: [], modelConfiguration: emptyModelConfiguration(), exportSettings: { include_comments: false, include_shape_assertions: false, preferred_binary_format: 'keras' }, futureTrainingConfiguration: { dataset: 'deferred' }, selectedNodeId: null, selectedEdgeId: null, copiedNodeIds: [], past: [], future: [] })
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
})
