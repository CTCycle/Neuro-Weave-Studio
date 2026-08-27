import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { ReactFlowProvider, type Node } from '@xyflow/react'
import { ModelNode } from './ModelNode'
import { PropertiesInspector } from './PropertiesInspector'
import { useGraphStore } from '../state/graphStore'
import type { EditorNodeData, NodeManifest } from '../domain/types'

const manifest: NodeManifest = {
  id: 'layer.dense',
  version: 1,
  display_name: 'Dense',
  category: 'core',
  description: 'Fully connected layer',
  inputs: [{ id: 'input', display_name: 'Input', kind: 'tensor', required: true, accepts_multiple: false, description: '' }],
  outputs: [{ id: 'output', display_name: 'Output', kind: 'tensor', required: true, accepts_multiple: false, description: '' }],
  parameters: [],
  shape_rule: { kind: 'dense', config: {} },
  implementations: { keras: { supported: true }, tensorflow: { supported: true }, pytorch: { supported: true }, jax: { supported: true } },
  ui: { icon: 'Box', accent: 'violet', width: 250, height: 120 },
  deprecated: false,
  documentation_ref: '',
}

const data: EditorNodeData = {
  manifest,
  parameters: {},
  label: 'Dense',
  shapeSummary: { node_id: 'node', outputs: { output: { shape: [null, 4, 4], dtype: 'float32' } } },
}

const node = { id: 'node', type: 'modelNode', position: { x: 0, y: 0 }, data } as Node<EditorNodeData, 'modelNode'>

describe('shape presentation', () => {
  afterEach(() => {
    cleanup()
    useGraphStore.getState().reset()
  })

  it('keeps inferred shape on the node footer without duplicating it in the inspector', () => {
    useGraphStore.setState({ nodes: [node], selectedNodeId: node.id })
    render(<PropertiesInspector />)
    expect(screen.queryByText(/Inferred output/i)).not.toBeInTheDocument()

    cleanup()
    render(<ReactFlowProvider><ModelNode data={data} /></ReactFlowProvider>)
    expect(screen.getByText('? × 4 × 4')).toBeInTheDocument()
  })
})
