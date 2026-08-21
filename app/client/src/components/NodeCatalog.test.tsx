import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import { NodeCatalog } from './NodeCatalog'
import type { NodeManifest } from '../domain/types'

const manifests: NodeManifest[] = [
  { id: 'layer.dense', version: 1, display_name: 'Dense', category: 'core', description: 'Fully connected layer', inputs: [], outputs: [], parameters: [], shape_rule: { kind: 'dense', config: {} }, implementations: { keras: { supported: true }, tensorflow: { supported: true }, pytorch: { supported: true }, jax: { supported: true } }, ui: { icon: 'Box', accent: 'violet', width: 250, height: 120 }, deprecated: false, documentation_ref: '' },
  { id: 'conv.conv2d', version: 1, display_name: 'Conv2D', category: 'convolutional', description: 'Spatial convolution', inputs: [], outputs: [], parameters: [], shape_rule: { kind: 'conv_nd', config: {} }, implementations: { keras: { supported: true }, tensorflow: { supported: true }, pytorch: { supported: true }, jax: { supported: true } }, ui: { icon: 'Box', accent: 'blue', width: 250, height: 120 }, deprecated: false, documentation_ref: '' },
]

describe('NodeCatalog', () => {
  afterEach(cleanup)

  it('filters nodes by text', async () => {
    render(<NodeCatalog manifests={manifests} executionBackend="pytorch" kerasEnabled />)
    await userEvent.type(screen.getByLabelText('Search nodes'), 'dense')
    expect(screen.getByText('Dense')).toBeInTheDocument()
    expect(screen.queryByText('Conv2D')).not.toBeInTheDocument()
  })

  it('groups nodes into collapsible categories', async () => {
    render(<NodeCatalog manifests={manifests} executionBackend="pytorch" kerasEnabled />)
    expect(screen.getByText('Dense')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Core/ }))
    expect(screen.queryByText('Dense')).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Core/ }))
    expect(screen.getByText('Dense')).toBeInTheDocument()
  })

  it('keeps the node drag payload stable', () => {
    render(<NodeCatalog manifests={manifests} executionBackend="pytorch" kerasEnabled />)
    const data = new Map<string, string>()
    const item = screen.getByText('Dense').closest('[draggable="true"]')
    expect(item).not.toBeNull()
    fireEvent.dragStart(item!, { dataTransfer: { setData: (type: string, value: string) => data.set(type, value), effectAllowed: '' } })
    expect(data.get('application/neuro-weave-node')).toBe('layer.dense')
  })
})
