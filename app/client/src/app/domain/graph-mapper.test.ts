import { describe, expect, it } from 'vitest'
import { defaultExportSettings, graphEditorSessionFromGraph, sessionToGraph } from './graph-mapper'
import type { ModelGraph, NodeManifest } from './types'
import { emptyModelConfiguration } from './types'

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
