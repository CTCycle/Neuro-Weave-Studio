// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.
import { describe, expect, it } from 'vitest'
import { DATASET_CAPABILITIES, arrayFormatFromPath, datasetCapability, formatFromPath, sourceExtensions, sourceMode, tableFormatFromPath } from './dataset-capabilities'
import { emptyModelConfiguration, conditionMatches, parameterValueMatchesDefinition, resolveImplementationCompatibility, resolveParameterCompatibility, type ModelGraph, type NodeManifest, type PortDefinition, type ImplementationMapping, type ParameterDefinition } from './types'
import { outputSelectors, summarizeGraphIo } from './graph-io'
import { defaultExportSettings, graphEditorSessionFromGraph, sessionToGraph } from './graph-mapper'
import { materializePorts, materializedOutputs } from './ports'

describe('dataset-capabilities behavior', () => {


describe('dataset capability registry', () => {
  it('exposes the supported organizations and keeps Auto Detect disabled', () => {
    expect(DATASET_CAPABILITIES.filter((item) => item.enabled).map((item) => item.id)).toEqual([
      'tabular', 'images', 'text', 'time_series', 'array', 'composed',
    ])
    const autoDetect = DATASET_CAPABILITIES.find((item) => item.id === 'auto_detect')
    expect(autoDetect?.enabled).toBe(false)
    expect(autoDetect?.sourceLabel).toContain('Coming later')
  })

  it('centralizes source layout behavior for composed and text sources', () => {
    expect(sourceMode('composed', 0)).toBe('directory')
    expect(sourceMode('composed', 1)).toBe('file')
    expect(sourceMode('text', 0)).toBe('file')
    expect(datasetCapability('time_series').formats).toEqual(['csv', 'tsv', 'parquet'])
  })

  it('derives unambiguous file formats from the selected path', () => {
    expect(tableFormatFromPath('C:\\data\\events.PARQUET')).toBe('parquet')
    expect(tableFormatFromPath('/tmp/events.tsv')).toBe('tsv')
    expect(arrayFormatFromPath('/tmp/features.NPZ')).toBe('npz')
    expect(formatFromPath('array', '/tmp/features.npy')).toBe('npy')
    expect(formatFromPath('images', '/tmp/images')).toBeNull()
  })

  it('derives browser filters from the source capability', () => {
    expect(sourceExtensions('tabular')).toEqual(['csv', 'tsv', 'parquet'])
    expect(sourceExtensions('array')).toEqual(['npy', 'npz'])
    expect(sourceExtensions('array', 2)).toEqual(['npy', 'npz'])
    expect(sourceExtensions('text', 0, { textTable: true })).toEqual(['csv', 'tsv', 'parquet'])
    expect(sourceExtensions('images')).toEqual([])
    expect(sourceMode('text', 0, { textPathKind: 'folder' })).toBe('directory')
  })
})
})

describe('graph-io behavior', () => {


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
})

describe('graph-mapper behavior', () => {


const inputManifest: NodeManifest = {
  id: 'input.tabular',
  version: 1,
  display_name: 'Tabular input',
  category: 'inputs',
  description: '',
  semantic_role: 'input',
  task_hint: null,
  inputs: [],
  outputs: [{ id: 'output', display_name: 'Output', kind: 'tensor', required: true, accepts_multiple: true, description: '' }],
  parameters: [{ id: 'shape', display_name: 'Shape', type: 'shape', default: [8], required: false, constraints: {}, options: [], description: '' }],
  shape_rule: { kind: 'input', config: {} },
  implementations: { keras: { supported: true, status: 'verified' } },
  ui: { icon: 'Database', accent: 'cyan', width: 220, height: 120 },
  deprecated: false,
  documentation_ref: '',
}

const currentGraph: ModelGraph = {
  schema_version: 6,
  nodes: [{
    id: 'input',
    definition_id: 'input.tabular',
    definition_version: 1,
    label: 'Features',
    position: { x: 0, y: 0 },
    parameters: { shape: [8] },
    locked: false,
  }],
  edges: [],
  output_bindings: [],
  model_configuration: emptyModelConfiguration(),
  export_settings: defaultExportSettings(),
  future_training_configuration: {},
}

describe('current graph mapping', () => {
  it('round-trips the current graph schema without compatibility rewriting', () => {
    const session = graphEditorSessionFromGraph(currentGraph, [inputManifest])

    expect(session.nodes[0].data.parameters).toEqual({ shape: [8] })
    expect(sessionToGraph(session)).toEqual(currentGraph)
    expect(currentGraph.schema_version).toBe(6)
    expect(currentGraph.nodes[0].parameters.shape).toEqual([8])
  })
})
})

describe('ports behavior', () => {


const dynamicOutput: PortDefinition = {
  id: 'output',
  display_name: 'Output',
  kind: 'tensor',
  required: true,
  accepts_multiple: false,
  description: 'Split output',
  cardinality: {
    kind: 'parameter',
    parameter_id: 'sections',
    minimum: 1,
    maximum: 4,
    id_pattern: 'output_{index}',
    display_name_pattern: 'Output {index}',
  },
}

const splitManifest = {
  inputs: [],
  outputs: [dynamicOutput],
} as NodeManifest

describe('declarative port materialization', () => {
  it('expands parameter-driven outputs into stable canonical ports', () => {
    expect(materializedOutputs(splitManifest, { sections: 3 }).map((port) => ({ id: port.id, label: port.display_name }))).toEqual([
      { id: 'output_0', label: 'Output 0' },
      { id: 'output_1', label: 'Output 1' },
      { id: 'output_2', label: 'Output 2' },
    ])
  })

  it('rejects values outside the declared cardinality contract', () => {
    expect(() => materializePorts([dynamicOutput], { sections: 0 })).toThrow(/between 1 and 4/)
    expect(() => materializePorts([dynamicOutput], { sections: 1.5 })).toThrow(/integer/)
    expect(() => materializePorts([dynamicOutput], { sections: 5 })).toThrow(/between 1 and 4/)
  })
})
})

describe('types.compatibility behavior', () => {


const parameter: ParameterDefinition = {
  id: 'mode',
  display_name: 'Mode',
  type: 'enum',
  default: 'auto',
  required: false,
  constraints: {},
  options: ['auto', 'fast'],
  description: '',
}

const manifest = (mapping: ImplementationMapping | undefined): NodeManifest => ({
  id: 'layer.test',
  version: 1,
  display_name: 'Test layer',
  category: 'test',
  description: '',
  semantic_role: 'operation',
  inputs: [],
  outputs: [],
  parameters: [parameter],
  shape_rule: { kind: 'operation', config: {} },
  implementations: mapping ? { pytorch: mapping } : {},
  ui: { icon: 'Box', accent: 'cyan', width: 250, height: 120 },
  deprecated: false,
  documentation_ref: '',
})

const verifiedMapping = (overrides: Partial<ImplementationMapping> = {}): ImplementationMapping => ({
  supported: true,
  status: 'verified',
  ...overrides,
})

describe('profile compatibility resolver', () => {
  it.each([
    ['missing_implementation', undefined, undefined],
    ['unsupported_implementation', false, 'verified'],
    ['unsupported_implementation', true, 'unsupported'],
    ['unverified_implementation', true, 'unverified'],
  ] as const)('treats %s mappings as unavailable', (reasonCode, supported, status) => {
    const mapping = supported === undefined
      ? undefined
      : verifiedMapping({ supported, status: status ?? 'verified' })
    const result = resolveImplementationCompatibility(manifest(mapping), 'pytorch', false)

    expect(result.available).toBe(false)
    expect(result.reasonCode).toBe(reasonCode)
    expect(result.reason).toContain('Test layer')
  })

  it('treats a Keras execution-backend exclusion as unavailable', () => {
    const result = resolveImplementationCompatibility(
      {
        ...manifest(undefined),
        implementations: { keras: verifiedMapping({ execution_backends: { pytorch: false } }) },
      },
      'pytorch',
      true,
    )

    expect(result).toMatchObject({
      available: false,
      reasonCode: 'backend_excluded',
    })
  })

  it('keeps an unsupported parameter default visible but blocks non-default values', () => {
    const node = manifest(verifiedMapping({
      parameters: {
        mode: {
          supported: false,
          status: 'verified',
          constraints: {},
        },
      },
    }))

    const defaultValue = resolveParameterCompatibility(node, parameter, 'auto', { mode: 'auto' }, 'pytorch', false)
    const nonDefaultValue = resolveParameterCompatibility(node, parameter, 'fast', { mode: 'fast' }, 'pytorch', false)

    expect(defaultValue).toMatchObject({ editable: false, valueCompatible: true, reasonCode: 'unsupported_parameter' })
    expect(nonDefaultValue).toMatchObject({ editable: false, valueCompatible: false, reasonCode: 'unsupported_parameter' })
  })

  it('keeps parameters editable when the verified node has no target-specific capability entry', () => {
    const result = resolveParameterCompatibility(manifest(verifiedMapping()), parameter, 'auto', { mode: 'auto' }, 'pytorch', false)

    expect(result).toMatchObject({ editable: true, valueCompatible: true, reasonCode: null })
  })

  it('treats an unverified parameter mapping as unavailable for every value', () => {
    const node = manifest(verifiedMapping({
      parameters: {
        mode: {
          supported: true,
          status: 'unverified',
          constraints: {},
        },
      },
    }))
    const result = resolveParameterCompatibility(node, parameter, 'auto', { mode: 'auto' }, 'pytorch', false)

    expect(result).toMatchObject({
      editable: false,
      valueCompatible: false,
      reasonCode: 'unverified_parameter',
    })
  })

  it('keeps a supported parameter editable while exposing a fixed-value restriction', () => {
    const node = manifest(verifiedMapping({
      parameters: {
        mode: {
          supported: true,
          status: 'verified',
          allowed_values: ['auto'],
          constraints: {},
        },
      },
    }))

    expect(resolveParameterCompatibility(node, parameter, 'auto', { mode: 'auto' }, 'pytorch', false)).toMatchObject({
      editable: false,
      valueCompatible: true,
      reasonCode: 'parameter_fixed',
    })
    expect(resolveParameterCompatibility(node, parameter, 'fast', { mode: 'fast' }, 'pytorch', false)).toMatchObject({
      editable: false,
      valueCompatible: false,
      reasonCode: 'unsupported_parameter_option',
    })
  })

  it('reports malformed stored values without making repair controls unavailable', () => {
    const node = manifest(verifiedMapping())
    const result = resolveParameterCompatibility(node, parameter, 'unknown', { mode: 'unknown' }, 'pytorch', false)

    expect(parameterValueMatchesDefinition(parameter, 'unknown')).toBe(false)
    expect(result).toMatchObject({
      editable: true,
      valueCompatible: false,
      reasonCode: 'invalid_parameter_value',
    })
  })

  it('evaluates all supported conditional rule forms consistently', () => {
    expect(conditionMatches({ mode: { equals: 'fast' } }, { mode: 'fast' })).toBe(true)
    expect(conditionMatches({ mode: { not_equals: 'auto' } }, { mode: 'fast' })).toBe(true)
    expect(conditionMatches({ mode: { in: ['fast', 'safe'] } }, { mode: 'safe' })).toBe(true)
    expect(conditionMatches({ mode: { not_in: ['auto', 'safe'] } }, { mode: 'fast' })).toBe(true)
    expect(conditionMatches({ mode: ['fast', 'safe'] }, { mode: 'auto' })).toBe(false)
  })
})
})
