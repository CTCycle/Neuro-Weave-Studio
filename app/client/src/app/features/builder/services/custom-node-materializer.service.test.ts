import { afterEach, describe, expect, it } from 'vitest'
import type { CustomNodeDefinition } from '../../../domain/custom-node.types'
import type { NodeManifest } from '../../../domain/types'
import { GraphEditorStore } from '../state/graph-editor.store'
import { CustomNodeMaterializerService } from './custom-node-materializer.service'

const port = {
  id: 'tensor',
  display_name: 'Tensor',
  kind: 'tensor' as const,
  required: true,
  accepts_multiple: false,
  description: '',
}

const manifest = (id: string, displayName: string): NodeManifest => ({
  id,
  version: 1,
  display_name: displayName,
  category: 'test',
  description: '',
  semantic_role: 'operation',
  inputs: [{ ...port }],
  outputs: [{ ...port }],
  parameters: [],
  shape_rule: { kind: 'operation', config: {} },
  implementations: { keras: { supported: true, status: 'verified' } },
  ui: { icon: 'Box', accent: 'cyan', width: 250, height: 120 },
  deprecated: false,
  documentation_ref: '',
})

const expressionManifest: NodeManifest = {
  ...manifest('math.expression', 'Safe Mathematical Expression'),
  parameters: [
    {
      id: 'expression',
      display_name: 'Expression',
      type: 'expression',
      default: 'x',
      required: true,
      constraints: {},
      options: [],
      description: '',
    },
  ],
  operation: {
    family: 'unary',
    shape_behavior: 'expression',
    output_dtype: 'same',
    arity: 1,
    variadic: false,
    arguments: [],
    keywords: [],
  },
}

const common = {
  version: 2,
  category: 'custom',
  description: 'Reusable template',
  origin: 'custom' as const,
  locked: false as const,
  can_edit: true as const,
  can_delete: true as const,
  created_at: '2026-09-03T12:00:00Z',
  updated_at: '2026-09-03T12:30:00Z',
}

describe('custom node materializer', () => {
  let store: GraphEditorStore
  let materializer: CustomNodeMaterializerService

  afterEach(() => store?.reset())

  it('materializes a mathematical template as the canonical safe-expression manifest', () => {
    store = new GraphEditorStore()
    store.setManifests([expressionManifest])
    materializer = new CustomNodeMaterializerService(store)
    const definition: CustomNodeDefinition = {
      ...common,
      id: 'custom.node.math',
      display_name: 'Scaled residual',
      kind: 'mathematical',
      executable_template: true,
      spec: {
        type: 'mathematical',
        formula: 'x^2',
        expression: 'x**2',
        input_symbols: ['x'],
      },
    }

    materializer.materialize(definition, { x: 60, y: 80 })

    expect(store.nodes()).toHaveLength(1)
    expect(store.nodes()[0].data.manifest.id).toBe('math.expression')
    expect(store.nodes()[0].data.label).toBe('Scaled residual')
    expect(store.nodes()[0].data.parameters['expression']).toBe('x**2')
  })

  it('expands a sequential template into built-in nodes and a canonical edge', () => {
    store = new GraphEditorStore()
    const first = manifest('layer.first', 'First')
    const second = manifest('layer.second', 'Second')
    store.setManifests([first, second])
    materializer = new CustomNodeMaterializerService(store)
    const definition: CustomNodeDefinition = {
      ...common,
      id: 'custom.node.block',
      display_name: 'Reusable block',
      kind: 'composite',
      executable_template: true,
      spec: {
        type: 'composite',
        connection_mode: 'sequential',
        components: [
          {
            definition_id: first.id,
            definition_version: first.version,
            label: 'Projection',
            parameters: {},
          },
          {
            definition_id: second.id,
            definition_version: second.version,
            label: 'Activation',
            parameters: {},
          },
        ],
      },
    }

    materializer.materialize(definition, { x: 40, y: 40 })

    expect(store.nodes().map((node) => node.data.manifest.id)).toEqual([
      'layer.first',
      'layer.second',
    ])
    expect(store.nodes().map((node) => node.data.label)).toEqual(['Projection', 'Activation'])
    expect(store.edges()).toHaveLength(1)
    expect(store.edges()[0]).toMatchObject({
      source: store.nodes()[0].id,
      sourceHandle: 'tensor',
      target: store.nodes()[1].id,
      targetHandle: 'tensor',
    })
  })

  it('preflights every component and leaves the graph unchanged when one is unavailable', () => {
    store = new GraphEditorStore()
    const first = {
      ...manifest('layer.first', 'First'),
      implementations: {
        keras: { supported: true, status: 'verified' as const },
        pytorch: { supported: true, status: 'verified' as const },
      },
    }
    const second = {
      ...manifest('layer.second', 'Second'),
      implementations: {
        keras: { supported: true, status: 'verified' as const },
        pytorch: { supported: false, status: 'verified' as const },
      },
    }
    store.setManifests([first, second])
    store.setExecutionProfile('pytorch', false)
    materializer = new CustomNodeMaterializerService(store)
    const definition: CustomNodeDefinition = {
      ...common,
      id: 'custom.node.block.invalid-profile',
      display_name: 'Unavailable block',
      kind: 'composite',
      executable_template: true,
      spec: {
        type: 'composite',
        connection_mode: 'sequential',
        components: [
          { definition_id: first.id, definition_version: first.version, parameters: {} },
          { definition_id: second.id, definition_version: second.version, parameters: {} },
        ],
      },
    }

    expect(materializer.compatibility(definition)).toMatchObject({
      available: false,
      reason: expect.stringContaining('not supported'),
    })
    expect(() => materializer.materialize(definition, { x: 40, y: 40 })).toThrow(/not supported/)
    expect(store.nodes()).toEqual([])
    expect(store.edges()).toEqual([])
  })

  it('does not materialize repository references into the canonical graph', () => {
    store = new GraphEditorStore()
    store.setManifests([expressionManifest])
    materializer = new CustomNodeMaterializerService(store)
    const definition: CustomNodeDefinition = {
      ...common,
      id: 'custom.node.repository',
      display_name: 'Pinned encoder',
      kind: 'repository',
      executable_template: false,
      spec: {
        type: 'repository',
        provider: 'huggingface',
        repository_id: 'openai-community/gpt2',
        revision: 'main',
        artifact_type: 'model',
        resolved_commit: 'a'.repeat(40),
      },
    }

    expect(() => materializer.materialize(definition, { x: 0, y: 0 })).toThrow(
      /trusted execution adapter/,
    )
    expect(store.nodes()).toEqual([])
    expect(store.edges()).toEqual([])
  })
})
