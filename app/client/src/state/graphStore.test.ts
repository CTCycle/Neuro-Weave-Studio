import { afterEach, describe, expect, it } from 'vitest'
import { useGraphStore, wouldCreateCycle } from './graphStore'
import { emptyModelConfiguration } from '../domain/types'
import type { EditorEdge, EditorNode, ProjectDocument } from '../domain/types'

const node = (id: string): EditorNode => ({
  id,
  type: 'modelNode',
  position: { x: 0, y: 0 },
  data: {
    manifest: { id: 'layer.dense', version: 1, display_name: 'Dense', category: 'core', description: '', inputs: [], outputs: [], parameters: [], shape_rule: { kind: 'dense', config: {} }, implementations: { keras: { supported: true }, tensorflow: { supported: true }, pytorch: { supported: true }, jax: { supported: true } }, ui: { icon: 'Box', accent: 'violet', width: 250, height: 120 }, deprecated: false, documentation_ref: '' },
    parameters: {}, label: id,
  },
})

describe('wouldCreateCycle', () => {
  it('rejects an edge that closes a cycle', () => {
    const nodes = [node('a'), node('b'), node('c')]
    const edges: EditorEdge[] = [{ id: 'ab', source: 'a', target: 'b' }, { id: 'bc', source: 'b', target: 'c' }]
    expect(wouldCreateCycle(nodes, edges, 'c', 'a')).toBe(true)
    expect(wouldCreateCycle(nodes, edges, 'a', 'c')).toBe(false)
  })
})

describe('graph metadata persistence', () => {
  afterEach(() => useGraphStore.getState().reset())

  it('preserves export and future-training metadata through edit saves', () => {
    const project = {
      graph: {
        schema_version: 2,
        nodes: [],
        edges: [],
        model_configuration: emptyModelConfiguration(),
        export_settings: {
          include_comments: false,
          include_shape_assertions: false,
          preferred_binary_format: 'keras',
        },
        future_training_configuration: { dataset: 'deferred' },
      },
    } as unknown as ProjectDocument

    useGraphStore.getState().replaceGraph(project)
    useGraphStore.getState().updateConfiguration({ optimizer: 'sgd' })

    expect(useGraphStore.getState().toGraph()).toMatchObject({
      export_settings: project.graph.export_settings,
      future_training_configuration: project.graph.future_training_configuration,
    })
    useGraphStore.getState().undo()
    expect(useGraphStore.getState().toGraph()).toMatchObject({
      export_settings: project.graph.export_settings,
      future_training_configuration: project.graph.future_training_configuration,
    })
    useGraphStore.getState().redo()
    expect(useGraphStore.getState().toGraph()).toMatchObject({
      export_settings: project.graph.export_settings,
      future_training_configuration: project.graph.future_training_configuration,
      model_configuration: { optimizer: 'sgd' },
    })
  })
})

describe('editor session persistence', () => {
  afterEach(() => useGraphStore.getState().reset())

  it('captures and restores graph history and selection without sharing mutable nodes', () => {
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
    useGraphStore.getState().setManifests([manifest])
    useGraphStore.getState().addNode(manifest, { x: 40, y: 60 })
    const nodeId = useGraphStore.getState().nodes[0].id
    useGraphStore.getState().selectNode(nodeId)
    const captured = useGraphStore.getState().captureEditorSession()

    useGraphStore.getState().reset()
    useGraphStore.getState().restoreEditorSession(captured)
    captured.nodes[0].data.label = 'mutated outside the store'

    expect(useGraphStore.getState().nodes[0].data.label).toBe('Dense')
    expect(useGraphStore.getState().selectedNodeId).toBe(nodeId)
    expect(useGraphStore.getState().past).toHaveLength(1)
  })
})
