import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { NodeCatalog } from './NodeCatalog'
import type { NodeManifest } from '../domain/types'

const manifests: NodeManifest[] = [
  {
    id: 'layer.dense',
    version: 1,
    display_name: 'Dense',
    category: 'core',
    description: 'Fully connected layer',
    inputs: [],
    outputs: [],
    parameters: [],
    shape_rule: { kind: 'dense', config: {} },
    implementations: {
      keras: { supported: true },
      tensorflow: { supported: true },
      pytorch: { supported: true },
      jax: { supported: true },
    },
    ui: { icon: 'Box', accent: 'violet', width: 250, height: 120 },
    deprecated: false,
    documentation_ref: '',
  },
]

describe('NodeCatalog', () => {
  afterEach(cleanup)

  it('uses the drag payload consumed by the model canvas', () => {
    render(<NodeCatalog manifests={manifests} executionBackend="pytorch" kerasEnabled />)
    fireEvent.click(screen.getByRole('button', { name: /Core/ }))

    const payload = new Map<string, string>()
    const item = screen.getByText('Dense').closest('[draggable="true"]')
    expect(item).not.toBeNull()

    fireEvent.dragStart(item!, {
      dataTransfer: {
        setData: (type: string, value: string) => payload.set(type, value),
        effectAllowed: '',
      },
    })

    expect(payload.get('application/neuro-weave-node')).toBe('layer.dense')
  })
})
