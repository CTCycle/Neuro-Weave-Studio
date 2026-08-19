import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import type { ValidationResult } from '../domain/types'
import { useGraphStore } from '../state/graphStore'
import { ValidationPanel } from './ValidationPanel'

const result: ValidationResult = {
  valid: false,
  diagnostics: [
    { code: 'invalid_shape', severity: 'error', message: 'Output shape is incompatible.', node_id: 'node-1', details: {} },
    { code: 'disconnected', severity: 'warning', message: 'Connection is incomplete.', edge_id: 'edge-1', details: {} },
  ],
  shape_summaries: [{ node_id: 'node-1', outputs: {}, parameter_count: 8 }],
  total_parameters: 8,
}

describe('ValidationPanel', () => {
  afterEach(() => {
    cleanup()
    useGraphStore.getState().reset()
  })

  it('opens blocking findings and selects their graph target', async () => {
    render(<ValidationPanel result={result} />)
    expect(await screen.findByText('Output shape is incompatible.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Output shape is incompatible/ }))
    expect(useGraphStore.getState().selectedNodeId).toBe('node-1')
    await userEvent.click(screen.getByRole('button', { name: /Connection is incomplete/ }))
    expect(useGraphStore.getState().selectedEdgeId).toBe('edge-1')
  })

  it('keeps a successful validation summary collapsed until opened', async () => {
    const validResult = { ...result, valid: true, diagnostics: [] }
    render(<ValidationPanel result={validResult} />)
    expect(screen.queryByText('No validation findings.')).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Validation/ }))
    expect(screen.getByText('No validation findings.')).toBeInTheDocument()
  })
})
