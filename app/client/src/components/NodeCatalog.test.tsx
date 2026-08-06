import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { NodeCatalog } from './NodeCatalog'
import type { NodeManifest } from '../domain/types'

const manifests: NodeManifest[] = [
  { id: 'layer.dense', version: 1, display_name: 'Dense', category: 'core', description: 'Fully connected layer', inputs: [], outputs: [], parameters: [], shape_rule: { kind: 'dense', config: {} }, frameworks: { keras: { supported: true }, tensorflow: { supported: true }, pytorch: { supported: true }, jax: { supported: true } }, ui: { icon: 'Box', accent: 'violet', width: 250, height: 120 }, deprecated: false, documentation_ref: '' },
  { id: 'conv.conv2d', version: 1, display_name: 'Conv2D', category: 'convolutional', description: 'Spatial convolution', inputs: [], outputs: [], parameters: [], shape_rule: { kind: 'conv_nd', config: {} }, frameworks: { keras: { supported: true }, tensorflow: { supported: true }, pytorch: { supported: true }, jax: { supported: true } }, ui: { icon: 'Box', accent: 'blue', width: 250, height: 120 }, deprecated: false, documentation_ref: '' },
]

describe('NodeCatalog', () => {
  it('filters nodes by text', async () => {
    render(<NodeCatalog manifests={manifests} codeTarget="keras" />)
    await userEvent.type(screen.getByLabelText('Search nodes'), 'dense')
    expect(screen.getByText('Dense')).toBeInTheDocument()
    expect(screen.queryByText('Conv2D')).not.toBeInTheDocument()
  })
})
