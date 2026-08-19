import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ExecutionBackendStatus } from '../domain/types'
import { useAppResources } from '../state/useAppResources'
import { BackendStatusPanel } from './BackendStatusPanel'

vi.mock('../state/useAppResources', () => ({ useAppResources: vi.fn() }))

const statuses: ExecutionBackendStatus[] = [
  { id: 'tensorflow', installed: true, available: true, version: '2.18', accelerator: 'cpu', device_name: 'CPU' },
  { id: 'pytorch', installed: true, available: false, version: '2.7', reason_code: 'worker_failed' },
  { id: 'jax', installed: false, available: false, reason_code: 'dependency_missing' },
]

describe('BackendStatusPanel', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('shows every detected backend and lets the user select only an operational one', async () => {
    const setSelectedExecutionBackend = vi.fn()
    vi.mocked(useAppResources).mockReturnValue({
      manifests: [], projects: [], backendStatuses: statuses, runtimeApis: [{ id: 'keras', available: true }, { id: 'native', available: false, reason_code: 'not_implemented' }], runtimeSelection: null, selectedExecutionBackend: null,
      loading: false, error: null, refreshProjects: vi.fn(), clearError: vi.fn(), setSelectedExecutionBackend, setRuntimeSelection: vi.fn(),
    })
    render(<BackendStatusPanel />)
    expect(screen.getByRole('button', { name: /TensorFlow/ })).toBeEnabled()
    expect(screen.getByRole('button', { name: /PyTorch/ })).toBeDisabled()
    expect(screen.getByRole('button', { name: /JAX/ })).toBeDisabled()
    await userEvent.click(screen.getByRole('button', { name: /TensorFlow/ }))
    expect(setSelectedExecutionBackend).toHaveBeenCalledWith('tensorflow')
  })
})
