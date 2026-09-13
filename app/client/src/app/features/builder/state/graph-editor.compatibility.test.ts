import { afterEach, describe, expect, it } from 'vitest'
import type { EditorEdge, EditorNode } from '../../../domain/editor.types'
import { emptyModelConfiguration } from '../../../domain/types'
import type { ImplementationMapping, NodeManifest, ParameterCapability, ParameterDefinition } from '../../../domain/types'
import { GraphEditorStore } from './graph-editor.store'

const outputPort = {
  id: 'output',
  display_name: 'Output',
  kind: 'tensor' as const,
  required: true,
  accepts_multiple: false,
  description: '',
}

const inputPort = {
  id: 'input',
  display_name: 'Input',
  kind: 'tensor' as const,
  required: true,
  accepts_multiple: false,
  description: '',
}

const units: ParameterDefinition = {
  id: 'units',
  display_name: 'Units',
  type: 'integer',
  default: 32,
  required: false,
  constraints: { min: 1 },
  options: [],
  description: '',
}

const kernelInitializer: ParameterDefinition = {
  id: 'kernel_initializer',
  display_name: 'Kernel initializer',
  type: 'initializer',
  default: 'glorot_uniform',
  required: false,
  constraints: {},
  options: ['glorot_uniform', 'he_normal'],
  description: '',
}

const unitForgetBias: ParameterDefinition = {
  id: 'unit_forget_bias',
  display_name: 'Unit forget bias',
  type: 'boolean',
  default: true,
  required: false,
  constraints: {},
  options: [],
  description: '',
}

const capability = (overrides: Partial<ParameterCapability> = {}): ParameterCapability => ({
  supported: true,
  status: 'verified',
  constraints: {},
  ...overrides,
})

const mapping = (parameters: Record<string, ParameterCapability> = {}): ImplementationMapping => ({
  supported: true,
  status: 'verified',
  parameters,
})

const manifest = (id = 'layer.dense', overrides: Partial<NodeManifest> = {}): NodeManifest => ({
  id,
  version: 1,
  display_name: 'Dense',
  category: 'core',
  description: '',
  semantic_role: 'operation',
  inputs: [inputPort],
  outputs: [outputPort],
  parameters: [units, kernelInitializer, unitForgetBias],
  shape_rule: { kind: 'dense', config: {} },
  implementations: {
    keras: mapping(),
    tensorflow: mapping({
      kernel_initializer: capability({ supported: false }),
      unit_forget_bias: capability({ allowed_values: [true] }),
    }),
    pytorch: mapping({
      kernel_initializer: capability({ supported: false }),
      unit_forget_bias: capability({ allowed_values: [true, false] }),
    }),
    jax: mapping({
      kernel_initializer: capability({ supported: false }),
      unit_forget_bias: capability({ allowed_values: [true] }),
    }),
  },
  ui: { icon: 'Box', accent: 'violet', width: 250, height: 120 },
  deprecated: false,
  documentation_ref: '',
  ...overrides,
})

const editorNode = (
  id: string,
  nodeManifest: NodeManifest,
  parameters: Record<string, unknown> = {},
): EditorNode => ({
  id,
  type: 'modelNode',
  position: { x: 0, y: 0 },
  locked: false,
  data: { manifest: nodeManifest, parameters, label: nodeManifest.display_name },
})

const session = (nodes: EditorNode[], edges: EditorEdge[] = []) => ({
  nodes,
  edges,
  outputBindings: [],
  modelConfiguration: emptyModelConfiguration(),
  exportSettings: { include_comments: true, include_shape_assertions: true, preferred_binary_format: 'keras' as const },
  futureTrainingConfiguration: {},
  selectedNodeId: null,
  selectedEdgeId: null,
  copiedNodeIds: [],
  past: [],
  future: [],
})

describe('GraphEditorStore backend compatibility guards', () => {
  let store: GraphEditorStore

  afterEach(() => store?.reset())

  it.each([
    ['tensorflow', true],
    ['pytorch', true],
    ['jax', true],
    ['tensorflow', false],
    ['pytorch', false],
    ['jax', false],
  ] as const)('adds a verified common node for the %s %s profile', (backend, kerasEnabled) => {
    store = new GraphEditorStore()
    const nodeManifest = manifest()
    store.setManifests([nodeManifest])
    store.setExecutionProfile(backend, kerasEnabled)

    expect(store.addNode(nodeManifest, { x: 40, y: 40 })).not.toBeNull()
    expect(store.nodes()).toHaveLength(1)
  })

  it('rejects a missing, unsupported, or unverified selected node mapping before mutation', () => {
    store = new GraphEditorStore()
    const nodeManifest = manifest('layer.unavailable', {
      implementations: {
        keras: mapping(),
        pytorch: mapping(),
      },
    })
    store.setManifests([nodeManifest])
    store.setExecutionProfile('jax', false)

    const revision = store.graphRevision()
    expect(store.nodeCompatibility(nodeManifest)).toMatchObject({
      available: false,
      reasonCode: 'missing_implementation',
    })
    expect(store.addNode(nodeManifest, { x: 40, y: 40 })).toBeNull()
    expect(store.nodes()).toEqual([])
    expect(store.graphRevision()).toBe(revision)

    const unverified = manifest('layer.unverified', {
      implementations: {
        keras: mapping(),
        pytorch: mapping(),
        jax: mapping(),
        tensorflow: mapping(),
      },
    })
    unverified.implementations['jax'].status = 'unverified'
    expect(store.nodeCompatibility(unverified)).toMatchObject({
      available: false,
      reasonCode: 'unverified_implementation',
    })
  })

  it.each([
    ['tensorflow', true],
    ['pytorch', true],
    ['jax', true],
    ['tensorflow', false],
    ['pytorch', false],
    ['jax', false],
  ] as const)('commits a compatible connection under the %s %s profile', (backend, kerasEnabled) => {
    store = new GraphEditorStore()
    const sourceManifest = manifest('input.tensor', { semantic_role: 'input', inputs: [], parameters: [] })
    const targetManifest = manifest()
    const source = editorNode('source', sourceManifest)
    const target = editorNode('target', targetManifest)
    store.setExecutionProfile(backend, kerasEnabled)
    store.restoreEditorSession(session([source, target]))

    expect(store.connect({ source: 'source', sourceHandle: 'output', target: 'target', targetHandle: 'input' })).toMatchObject({ valid: true })
    expect(store.edges()).toHaveLength(1)
  })

  it('rejects direct connection commits for an unavailable endpoint without partial history', () => {
    store = new GraphEditorStore()
    const sourceManifest = manifest('input.tensor', { semantic_role: 'input', inputs: [], parameters: [] })
    const unavailableTarget = manifest('layer.unavailable', {
      implementations: {
        keras: mapping(),
        pytorch: mapping({}),
        tensorflow: mapping({}),
        jax: { ...mapping(), supported: false },
      },
    })
    store.setExecutionProfile('jax', false)
    store.restoreEditorSession(session([editorNode('source', sourceManifest), editorNode('target', unavailableTarget)]))
    const historyLength = store.past().length

    expect(store.commitConnection({ source: 'source', sourceHandle: 'output', target: 'target', targetHandle: 'input' }, store.graphRevision())).toBe(false)
    expect(store.edges()).toEqual([])
    expect(store.past()).toHaveLength(historyLength)
  })

  it('retains invalid values for repair while guarding update, duplicate, and paste paths', () => {
    store = new GraphEditorStore()
    const nodeManifest = manifest()
    const node = editorNode('dense', nodeManifest, { units: 32, kernel_initializer: 'he_normal' })
    store.setExecutionProfile('pytorch', false)
    store.restoreEditorSession(session([node]))

    expect(store.updateNode('dense', { parameters: { units: 64, kernel_initializer: 'he_normal' } })).toBe(true)
    expect(store.nodes()[0].data.parameters).toEqual({ units: 64, kernel_initializer: 'he_normal' })

    const beforeRejectedUpdate = structuredClone(store.nodes()[0].data.parameters)
    expect(store.updateNode('dense', { parameters: { units: 64, kernel_initializer: 'he_uniform' } })).toBe(false)
    expect(store.nodes()[0].data.parameters).toEqual(beforeRejectedUpdate)

    store.selectNode('dense')
    store.copySelected()
    expect(store.duplicateExact('dense')).toBeNull()
    expect(store.duplicateWithDefaults('dense')).toBeNull()
    expect(store.pasteCopied()).toBe(false)
    expect(store.nodes()).toHaveLength(1)

    store.resetParameters('dense')
    expect(store.nodes()[0].data.parameters).toEqual({ units: 32, kernel_initializer: 'glorot_uniform', unit_forget_bias: true })
    expect(store.duplicateExact('dense')).not.toBeNull()
  })

  it('rejects malformed parameter mutations without changing the stored graph', () => {
    store = new GraphEditorStore()
    const nodeManifest = manifest()
    const node = editorNode('dense', nodeManifest, { units: 32, kernel_initializer: 'glorot_uniform', unit_forget_bias: true })
    store.setExecutionProfile('pytorch', true)
    store.restoreEditorSession(session([node]))
    const before = structuredClone(store.nodes()[0].data.parameters)
    const revision = store.graphRevision()

    expect(store.parameterCompatibility(store.nodes()[0], units, 0)).toMatchObject({ valueCompatible: false })
    expect(store.updateNode('dense', { parameters: { ...before, units: 0 } })).toBe(false)
    expect(store.nodes()[0].data.parameters).toEqual(before)
    expect(store.graphRevision()).toBe(revision)
  })

  it('retains parameter values and re-resolves their diagnostics after profile changes', () => {
    store = new GraphEditorStore()
    const nodeManifest = manifest()
    const node = editorNode('lstm', nodeManifest, { unit_forget_bias: false })
    store.setExecutionProfile('pytorch', true)
    store.restoreEditorSession(session([node]))

    expect(store.nodes()[0].data.parameters).toEqual({ unit_forget_bias: false })
    expect(store.parameterCompatibility(store.nodes()[0], unitForgetBias, false)).toMatchObject({
      editable: true,
      valueCompatible: true,
    })

    store.setExecutionProfile('jax', false)
    expect(store.nodes()[0].data.parameters).toEqual({ unit_forget_bias: false })
    expect(store.parameterCompatibility(store.nodes()[0], unitForgetBias, false)).toMatchObject({
      editable: false,
      valueCompatible: false,
      reasonCode: 'unsupported_parameter_option',
    })

    store.resetParameters('lstm')
    expect(store.nodes()[0].data.parameters['unit_forget_bias']).toBe(true)
    expect(store.parameterCompatibility(store.nodes()[0], unitForgetBias, true).valueCompatible).toBe(true)
  })
})
