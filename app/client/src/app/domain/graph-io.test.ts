import { describe, expect, it } from 'vitest'
import { emptyModelConfiguration, type ModelGraph, type NodeManifest } from './types'
import { outputSelectors, summarizeGraphIo } from './graph-io'

const port = (id = 'output') => ({ id, display_name: id, kind: 'tensor' as const, required: true, accepts_multiple: false, description: '' })

const manifest = (id: string, semantic_role: NodeManifest['semantic_role'], outputs = [port()], task_hint?: string): NodeManifest => ({
  id,
  version: 1,
  display_name: id,
  category: semantic_role === 'input' ? 'inputs' : 'core',
  description: '',
  semantic_role,
  task_hint: task_hint ?? null,
  inputs: semantic_role === 'input' ? [] : [{ id: 'input', display_name: 'Input', kind: 'tensor', required: true, accepts_multiple: false, description: '' }],
  outputs,
  parameters: [],
  shape_rule: { kind: 'preserve', config: {} },
  implementations: { keras: { supported: true, status: 'verified' } },
  ui: { icon: 'Box', accent: 'cyan', width: 250, height: 120 },
  deprecated: false,
  documentation_ref: '',
})

const manifests = [
  manifest('input.tensor', 'input'),
  manifest('layer.head', 'operation', [port()], 'binary_classification'),
  manifest('layer.side', 'operation'),
]

const classificationManifests = [
  manifest('input.tensor', 'input'),
  manifest('layer.dense', 'operation'),
  manifest('output.classification', 'output_adapter', [port()], 'classification'),
]

const graph = (output_bindings: ModelGraph['output_bindings'] = []): ModelGraph => ({
  schema_version: 6,
  nodes: [
    { id: 'features', definition_id: 'input.tensor', definition_version: 1, label: 'Features', position: { x: 0, y: 0 }, parameters: { shape: [8] }, locked: false },
    { id: 'head', definition_id: 'layer.head', definition_version: 1, label: 'Primary', position: { x: 200, y: 0 }, parameters: {}, locked: false },
    { id: 'side', definition_id: 'layer.side', definition_version: 1, label: 'Auxiliary', position: { x: 200, y: 120 }, parameters: {}, locked: false },
  ],
  edges: [
    { id: 'e1', source_node_id: 'features', source_port_id: 'output', target_node_id: 'head', target_port_id: 'input' },
  ],
  output_bindings,
  model_configuration: emptyModelConfiguration(),
  export_settings: { include_comments: true, include_shape_assertions: true, preferred_binary_format: 'keras' },
  future_training_configuration: {},
})

const classificationGraph = (parameters: Record<string, unknown>): ModelGraph => ({
  ...graph(),
  nodes: [
    { id: 'features', definition_id: 'input.tensor', definition_version: 1, label: 'Features', position: { x: 0, y: 0 }, parameters: { shape: [8] }, locked: false },
    { id: 'head', definition_id: 'layer.dense', definition_version: 1, label: 'Head', position: { x: 200, y: 0 }, parameters, locked: false },
    { id: 'output', definition_id: 'output.classification', definition_version: 1, label: 'Prediction', position: { x: 400, y: 0 }, parameters: {}, locked: false },
  ],
  edges: [
    { id: 'e1', source_node_id: 'features', source_port_id: 'output', target_node_id: 'head', target_port_id: 'input' },
    { id: 'e2', source_node_id: 'head', source_port_id: 'output', target_node_id: 'output', target_port_id: 'input' },
  ],
})

describe('graph I/O summary', () => {
  it('counts semantic input nodes and infers every unconsumed value output', () => {
    const summary = summarizeGraphIo(graph(), manifests)

    expect(summary.inputCount).toBe(1)
    expect(summary.outputs.map((item) => item.node_id)).toEqual(['head', 'side'])
    expect(summary.outputs.every((item) => item.explicit === false)).toBe(true)
    expect(summary.outputs[0].task_hint).toBe('binary_classification')
  })

  it('uses explicit output bindings as the complete output boundary', () => {
    const model = graph([{ node_id: 'side', port_id: 'output' }])
    const summary = summarizeGraphIo(model, manifests)

    expect(summary.outputCount).toBe(1)
    expect(summary.outputs[0]).toMatchObject({ node_id: 'side', port_id: 'output', name: 'Auxiliary', explicit: true })
    expect(outputSelectors(summary.outputs[0], model)).toEqual(['Auxiliary', 'side', 'side:output'])
  })

  it('resolves explicit bindings against materialized dynamic output ports', () => {
    const split = manifest('merge.split', 'operation', [{
      ...port(),
      id: 'output',
      cardinality: {
        kind: 'parameter' as const,
        parameter_id: 'sections',
        minimum: 1,
        maximum: 4,
        id_pattern: 'output_{index}',
        display_name_pattern: 'Output {index}',
      },
    }])
    split.parameters = [{ id: 'sections', display_name: 'Sections', type: 'integer', default: 2, required: true, constraints: {}, options: [], description: '' }]
    const model = graph([{ node_id: 'side', port_id: 'output_1' }])
    model.nodes[2] = { ...model.nodes[2], definition_id: split.id, parameters: { sections: 2 } }

    const summary = summarizeGraphIo(model, [...manifests, split])

    expect(summary.outputs).toEqual([{ node_id: 'side', port_id: 'output_1', name: 'Auxiliary_output_1', task_hint: null, explicit: true }])
  })

  it('derives classification task metadata from the terminal output width', () => {
    const binary = summarizeGraphIo(
      classificationGraph({ units: 1, activation: 'sigmoid' }),
      classificationManifests,
    )
    const multiclass = summarizeGraphIo(
      classificationGraph({ units: 3, activation: 'softmax' }),
      classificationManifests,
    )

    expect(binary.outputs[0].task_hint).toBe('binary_classification')
    expect(multiclass.outputs[0].task_hint).toBe('multiclass_classification')
  })

  it('leaves classification task metadata unknown when output width is unknown', () => {
    const summary = summarizeGraphIo(
      classificationGraph({ activation: 'sigmoid' }),
      classificationManifests,
    )

    expect(summary.outputs[0].task_hint).toBeNull()
  })
})
