import { describe, expect, it } from 'vitest'
import type { ExecutionBackendStatus } from '../../domain/types'
import {
  backendSupportsApi,
  defaultRuntimePreference,
  preferredBackend,
  preferredBackendForApi,
  preferredRuntimeApi,
} from './runtime-selection'

const status = (
  id: ExecutionBackendStatus['id'],
  available: boolean,
  runtimeApis: ExecutionBackendStatus['runtime_apis'],
): ExecutionBackendStatus => ({
  id,
  installed: available,
  available,
  runtime_apis: runtimeApis,
  devices: [{
    device_id: 'cpu',
    lease_key: 'cpu',
    kind: 'cpu',
    name: 'CPU',
    accelerator: 'cpu',
    available: true,
    supported_memory_policies: ['default'],
  }],
})

describe('runtime selection', () => {
  it('uses native when Keras is unavailable for an otherwise usable backend', () => {
    const nativeOnly = status('jax', true, [
      { id: 'keras', available: false, state: 'unavailable' },
      { id: 'native', available: true, state: 'verified' },
    ])

    expect(preferredRuntimeApi(nativeOnly)).toBe('native')
    expect(defaultRuntimePreference(nativeOnly)?.api).toBe('native')
  })

  it('prefers Keras when both APIs are available', () => {
    const both = status('tensorflow', true, [
      { id: 'native', available: true, state: 'verified' },
      { id: 'keras', available: true, state: 'verified' },
    ])

    expect(preferredRuntimeApi(both)).toBe('keras')
    expect(defaultRuntimePreference(both)?.api).toBe('keras')
  })

  it('selects only a backend that exposes the requested API', () => {
    const statuses = [
      status('pytorch', true, [{ id: 'keras', available: false }, { id: 'native', available: true }]),
      status('tensorflow', true, [{ id: 'keras', available: true }, { id: 'native', available: false }]),
    ]

    expect(preferredBackendForApi(statuses, 'keras')).toBe('tensorflow')
    expect(preferredBackendForApi(statuses, 'native')).toBe('pytorch')
    expect(backendSupportsApi(statuses[0], 'keras')).toBe(false)
  })

  it('retains the legacy Keras-only interpretation when per-backend API metadata is absent', () => {
    const legacy = status('pytorch', true, undefined)

    expect(backendSupportsApi(legacy, 'keras')).toBe(true)
    expect(backendSupportsApi(legacy, 'native')).toBe(false)
    expect(preferredBackend([legacy])).toBe('pytorch')
  })
})
