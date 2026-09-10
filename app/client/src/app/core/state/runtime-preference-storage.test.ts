import { beforeEach, describe, expect, it } from 'vitest'
import type { WorkerConfiguration } from '../../domain/types'
import { RUNTIME_PREFERENCES_STORAGE_KEY, readRuntimePreferences, storedPreferenceBackend, writeRuntimePreferences } from './runtime-preference-storage'

const preference: WorkerConfiguration = {
  api: 'native',
  backend: 'pytorch',
  device_id: 'cuda:0',
  memory_policy: { kind: 'pytorch_fraction', fraction: 0.25 },
  precision: 'float32',
}

describe('runtime preference storage', () => {
  beforeEach(() => localStorage.clear())

  it('round-trips browser-local worker preferences without a server write', () => {
    writeRuntimePreferences({ pytorch: preference })

    expect(readRuntimePreferences()).toEqual({ pytorch: preference })
    expect(storedPreferenceBackend(readRuntimePreferences())).toBe('pytorch')
    expect(localStorage.getItem(RUNTIME_PREFERENCES_STORAGE_KEY)).toContain('cuda:0')
  })

  it('ignores malformed or unrelated stored entries', () => {
    localStorage.setItem(RUNTIME_PREFERENCES_STORAGE_KEY, JSON.stringify({ unknown: { device_id: 'bad' }, tensorflow: 'bad' }))

    expect(readRuntimePreferences()).toEqual({})
  })
})
