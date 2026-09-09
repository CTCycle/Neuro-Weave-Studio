import { afterEach, describe, expect, it } from 'vitest'
import { emptyModelConfiguration } from '../../../domain/types'
import type { EditorEdge, EditorNode } from '../../../domain/editor.types'
import { isGraphConnectionValid, validateGraphConnection } from '../graph/graph-validation'
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
      edges: [{ id: 'source-target', source: 'source', sourceHandle: 'output', target: 'target', targetHandle: 'input', type: 'bezier' }],
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

  it('uses framework defaults for reset and default duplication', () => {
    store = new GraphEditorStore()
    const configuredManifest = {
      ...manifest,
      parameters: [{ ...manifest.parameters[0], framework_default: 16, editor_default: 32 }],
    }
    store.setManifests([configuredManifest])
    store.addNode(configuredManifest, { x: 40, y: 60 })
    const nodeId = store.nodes()[0].id
    store.updateNode(nodeId, { parameters: { units: 128 } })

    const copyId = store.duplicateWithDefaults(nodeId)
    expect(store.nodes().find((node) => node.id === copyId)?.data.parameters).toEqual({ units: 16 })

    store.resetParameters(nodeId)
    expect(store.nodes().find((node) => node.id === nodeId)?.data.parameters).toEqual({ units: 16 })
  })

  it('uses the selected implementation capability default when one is declared', () => {
    store = new GraphEditorStore()
    const capabilityManifest = {
      ...manifest,
      implementations: {
        keras: {
          ...manifest.implementations.keras,
          parameters: { units: { supported: true, constraints: {}, default: 12 } },
        },
      },
    }
    store.setManifests([capabilityManifest])
    store.setExecutionProfile('pytorch', true)
    store.addNode(capabilityManifest, { x: 40, y: 60 })
    const nodeId = store.nodes()[0].id

    expect(store.duplicateWithDefaults(nodeId)).not.toBeNull()
    expect(store.nodes()[1].data.parameters).toEqual({ units: 12 })
    store.resetParameters(nodeId)
    expect(store.nodes()[0].data.parameters).toEqual({ units: 12 })
  })

  it('rejects invalid and duplicate port connections before they reach the graph', () => {
    const input = {
      ...manifest,
      id: 'input.tensor',
      semantic_role: 'input' as const,
      inputs: [],
      outputs: [{ ...manifest.outputs[0], kind: 'tensor' as const }],
    }
    const target = {
      ...manifest,
      inputs: [{ id: 'input', display_name: 'Input', kind: 'tensor' as const, required: true, accepts_multiple: false, description: '' }],
    }
    const disabled = {
      ...target,
      inputs: [{ ...target.inputs[0], enabled_when: { mode: 'enabled' } }],
    }
    const source: EditorNode = { id: 'source', type: 'modelNode', position: { x: 0, y: 0 }, locked: false, data: { manifest: input, parameters: {}, label: 'Source' } }
    const destination: EditorNode = { id: 'destination', type: 'modelNode', position: { x: 0, y: 0 }, locked: false, data: { manifest: target, parameters: {}, label: 'Destination' } }
    const blockedDestination: EditorNode = { id: 'blocked', type: 'modelNode', position: { x: 0, y: 0 }, locked: false, data: { manifest: disabled, parameters: {}, label: 'Blocked' } }
    const connection = { source: 'source', sourceHandle: 'output', target: 'destination', targetHandle: 'input' }

    expect(isGraphConnectionValid([source, destination], [], connection)).toBe(true)
    expect(isGraphConnectionValid([source, destination], [{ id: 'existing', ...connection }], connection)).toBe(false)
    expect(isGraphConnectionValid([source, blockedDestination], [], { ...connection, target: 'blocked' })).toBe(false)
  })

  it('allows output fan-out and supported multi-input connections', () => {
    const sourceManifest = { ...manifest, id: 'source.tensor', inputs: [], outputs: [{ ...manifest.outputs[0], accepts_multiple: false }] }
    const targetManifest = {
      ...manifest,
      id: 'merge.tensor',
      inputs: [{ id: 'input', display_name: 'Input', kind: 'tensor' as const, required: true, accepts_multiple: true, description: '' }],
      outputs: [{ ...manifest.outputs[0], accepts_multiple: false }],
    }
    const source = { ...node('source'), data: { ...node('source').data, manifest: sourceManifest } }
    const sourceTwo = { ...node('source-two'), data: { ...node('source-two').data, manifest: sourceManifest } }
    const firstTarget = { ...node('target-one'), data: { ...node('target-one').data, manifest: targetManifest } }
    const secondTarget = { ...node('target-two'), data: { ...node('target-two').data, manifest: targetManifest } }
    const first = { source: 'source', sourceHandle: 'output', target: 'target-one', targetHandle: 'input' }
    const second = { source: 'source', sourceHandle: 'output', target: 'target-two', targetHandle: 'input' }
    const mergeOne = { source: 'source', sourceHandle: 'output', target: 'target-one', targetHandle: 'input' }
    const mergeTwo = { source: 'source-two', sourceHandle: 'output', target: 'target-one', targetHandle: 'input' }

    expect(validateGraphConnection([source, firstTarget, secondTarget], [], first).valid).toBe(true)
    expect(validateGraphConnection([source, firstTarget, secondTarget], [
      { id: 'first', ...first },
    ], second).valid).toBe(true)
    expect(validateGraphConnection([source, sourceTwo, firstTarget], [
      { id: 'merge-one', ...mergeOne },
    ], mergeTwo).valid).toBe(true)
  })

  it('returns typed cycle feedback without adding history or an edge', () => {
    const cycleManifest = {
      ...manifest,
      inputs: [{ id: 'input', display_name: 'Input', kind: 'tensor' as const, required: true, accepts_multiple: true, description: '' }],
    }
    const nodes = ['a', 'b', 'c'].map((id) => ({ ...node(id), data: { ...node(id).data, manifest: cycleManifest } }))
    const edges: EditorEdge[] = [
      { id: 'ab', source: 'a', sourceHandle: 'output', target: 'b', targetHandle: 'input' },
      { id: 'bc', source: 'b', sourceHandle: 'output', target: 'c', targetHandle: 'input' },
    ]
    const connection = { source: 'c', sourceHandle: 'output', target: 'a', targetHandle: 'input' }
    expect(validateGraphConnection(nodes, edges, connection)).toMatchObject({ valid: false, reason: 'cycle' })

    store = new GraphEditorStore()
    store.restoreEditorSession({ nodes, edges, outputBindings: [], modelConfiguration: emptyModelConfiguration(), exportSettings: { include_comments: true, include_shape_assertions: true, preferred_binary_format: 'keras' }, futureTrainingConfiguration: {}, selectedNodeId: null, selectedEdgeId: null, copiedNodeIds: [], past: [], future: [] })
    const pastLength = store.past().length
    const result = store.connect(connection)
    expect(result).toMatchObject({ valid: false, reason: 'cycle' })
    expect(store.edges()).toHaveLength(2)
    expect(store.past()).toHaveLength(pastLength)
  })

  it('does not mark input nodes as explicit outputs', () => {
    store = new GraphEditorStore()
    const inputManifest = { ...manifest, id: 'input.tensor', semantic_role: 'input' as const }
    store.setManifests([inputManifest])
    store.addNode(inputManifest, { x: 40, y: 60 })
    store.toggleNodeOutput(store.nodes()[0].id)
    expect(store.outputBindings()).toEqual([])
  })

  it('guards inferred terminal outputs and updates availability across explicit transitions', () => {
    store = new GraphEditorStore()
    const source = node('source')
    const terminal = node('terminal')
    store.restoreEditorSession({
      nodes: [source, terminal],
      edges: [{ id: 'source-terminal', source: 'source', sourceHandle: 'output', target: 'terminal', targetHandle: 'input', type: 'bezier' }],
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

    expect(store.isInferredOutput('terminal')).toBe(true)
    expect(store.canToggleNodeOutput('terminal')).toBe(false)
    const pastLength = store.past().length
    store.toggleNodeOutput('terminal')
    expect(store.outputBindings()).toEqual([])
    expect(store.past()).toHaveLength(pastLength)

    store.toggleNodeOutput('source')
    expect(store.outputBindings()).toEqual([{ node_id: 'source', port_id: 'output' }])
    expect(store.isInferredOutput('terminal')).toBe(false)
    expect(store.canToggleNodeOutput('terminal')).toBe(true)

    store.toggleNodeOutput('terminal')
    expect(store.outputBindings()).toEqual([
      { node_id: 'source', port_id: 'output' },
      { node_id: 'terminal', port_id: 'output' },
    ])
    store.toggleNodeOutput('source')
    expect(store.outputBindings()).toEqual([{ node_id: 'terminal', port_id: 'output' }])
    store.toggleNodeOutput('terminal')
    expect(store.outputBindings()).toEqual([])
    expect(store.isInferredOutput('terminal')).toBe(true)
    expect(store.canToggleNodeOutput('terminal')).toBe(false)
  })

  it('deduplicates restored bindings and preserves independent multi-output bindings', () => {
    store = new GraphEditorStore()
    const multiOutputManifest = {
      ...manifest,
      id: 'layer.multi-output',
      outputs: [
        manifest.outputs[0],
        { id: 'auxiliary', display_name: 'Auxiliary', kind: 'tensor' as const, required: true, accepts_multiple: true, description: '' },
      ],
    }
    const multiNode: EditorNode = { ...node('multi'), data: { ...node('multi').data, manifest: multiOutputManifest } }
    store.restoreEditorSession({
      nodes: [multiNode],
      edges: [],
      outputBindings: [
        { node_id: 'multi', port_id: 'output' },
        { node_id: 'multi', port_id: 'output' },
      ],
      modelConfiguration: emptyModelConfiguration(),
      exportSettings: { include_comments: true, include_shape_assertions: true, preferred_binary_format: 'keras' },
      futureTrainingConfiguration: {},
      selectedNodeId: null,
      selectedEdgeId: null,
      copiedNodeIds: [],
      past: [],
      future: [],
    })

    expect(store.outputBindings()).toEqual([{ node_id: 'multi', port_id: 'output' }])
    store.toggleNodeOutput('multi', 'auxiliary')
    expect(store.outputBindings()).toEqual([
      { node_id: 'multi', port_id: 'output' },
      { node_id: 'multi', port_id: 'auxiliary' },
    ])
    store.toggleNodeOutput('multi', 'output')
    expect(store.outputBindings()).toEqual([{ node_id: 'multi', port_id: 'auxiliary' }])
    store.toggleNodeOutput('multi')
    expect(store.outputBindings()).toEqual([])
  })

  it('locks a node against movement while preserving configuration and output state', () => {
    store = new GraphEditorStore()
    const source = node('source')
    const target = node('target')
    store.restoreEditorSession({
      nodes: [source, target],
      edges: [{ id: 'source-target', source: 'source', sourceHandle: 'output', target: 'target', targetHandle: 'input', type: 'bezier' }],
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
    const nodeId = 'source'
    store.toggleNodeLocked(nodeId)
    store.moveNodes([{ id: nodeId, position: { x: 240, y: 260 } }])
    expect(store.nodes()[0].locked).toBe(true)
    expect(store.nodes()[0].position).toEqual({ x: 0, y: 0 })

    store.toggleNodeOutput(nodeId)
    expect(store.isOutput(nodeId)).toBe(true)
    expect(store.isEffectiveOutput(nodeId)).toBe(true)
    expect(store.toGraph().output_bindings).toEqual([{ node_id: nodeId, port_id: 'output' }])
    expect(store.toGraph().nodes[0].locked).toBe(true)

    store.selectNode(nodeId)
    store.deleteSelected()
    expect(store.toGraph().output_bindings).toEqual([])
  })
})
