import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ModelSmokeTestDrawer } from './ModelSmokeTestDrawer'

const status = { id: 'pytorch' as const, available: true, accelerator: 'cuda', device_name: 'Test GPU' }

describe('ModelSmokeTestDrawer', () => {
  afterEach(cleanup)

  it('makes the dry-run boundary and ready action explicit', () => {
    render(<ModelSmokeTestDrawer backend="pytorch" backendStatus={status} state={{ status: 'ready' }} onRun={vi.fn()} onClose={vi.fn()} />)
    expect(screen.getByRole('dialog', { name: 'Model smoke test' })).toBeInTheDocument()
    expect(screen.getByText(/no training or weight updates/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /run dummy inference/i })).toBeEnabled()
  })

  it('shows successful tensor metadata', () => {
    render(<ModelSmokeTestDrawer backend="pytorch" backendStatus={status} state={{ status: 'succeeded', result: { status: 'succeeded', execution_backend: 'pytorch', accelerator: 'cuda', device_name: 'Test GPU', inputs: [{ node_id: 'input', shape: [1, 4], dtype: 'float32' }], outputs: [{ node_id: 'output', shape: [1, 1], dtype: 'float32' }], parameter_count: 5, duration_ms: 3.2 } }} onRun={vi.fn()} onClose={vi.fn()} />)
    expect(screen.getByText('Execution succeeded')).toBeInTheDocument()
    expect(screen.getByText('[1, 4]')).toBeInTheDocument()
    expect(screen.getByText('5')).toBeInTheDocument()
  })

  it('shows actionable validation errors', () => {
    render(<ModelSmokeTestDrawer backend="pytorch" backendStatus={status} state={{ status: 'error', error: { code: 'model_validation_failed', message: 'Shape dimensions cannot be zero', diagnostics: [{ code: 'invalid_shape_dimension', severity: 'error', message: 'Shape dimensions cannot be zero', details: {} }] } }} onRun={vi.fn()} onClose={vi.fn()} />)
    expect(screen.getByText('Execution blocked')).toBeInTheDocument()
    expect(screen.getAllByText('Shape dimensions cannot be zero')).toHaveLength(2)
  })
})
