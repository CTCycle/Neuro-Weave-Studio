import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { WorkerConfiguration } from '../../domain/types'
import {
  RUNTIME_PREFERENCE_BACKEND_STORAGE_KEY,
  RUNTIME_PREFERENCES_STORAGE_KEY,
  readRuntimePreferenceBackend,
  readRuntimePreferences,
  storedPreferenceBackend,
  writeRuntimePreferenceBackend,
  writeRuntimePreferences,
} from './runtime-preference-storage'

const preference: WorkerConfiguration = {
  api: 'native',
  backend: 'pytorch',
  device_id: 'cuda:0',
  memory_policy: { kind: 'pytorch_fraction', fraction: 0.25 },
  precision: 'float32',
}

describe('runtime preference storage', () => {
  beforeEach(() => localStorage.clear())
  afterEach(() => vi.restoreAllMocks())

  it('round-trips browser-local worker preferences without a server write', () => {
    writeRuntimePreferences({ pytorch: preference })

    expect(readRuntimePreferences()).toEqual({ pytorch: preference })
    expect(storedPreferenceBackend(readRuntimePreferences())).toBe('pytorch')
    expect(localStorage.getItem(RUNTIME_PREFERENCES_STORAGE_KEY)).toContain('cuda:0')
  })

  it('ignores malformed or unrelated stored entries', () => {
    localStorage.setItem(RUNTIME_PREFERENCES_STORAGE_KEY, JSON.stringify({
      unknown: { device_id: 'bad' },
      tensorflow: 'bad',
      pytorch: { ...preference, memory_policy: null },
    }))

    expect(readRuntimePreferences()).toEqual({})
  })

  it('persists and validates the last active backend independently', () => {
    writeRuntimePreferenceBackend('tensorflow')

    expect(localStorage.getItem(RUNTIME_PREFERENCE_BACKEND_STORAGE_KEY)).toBe('tensorflow')
    expect(readRuntimePreferenceBackend()).toBe('tensorflow')

    localStorage.setItem(RUNTIME_PREFERENCE_BACKEND_STORAGE_KEY, 'cuda:0')
    expect(readRuntimePreferenceBackend()).toBeNull()
  })

  it('treats inaccessible storage reads and writes as unavailable', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('Storage is disabled.', 'SecurityError')
    })

    expect(readRuntimePreferences()).toEqual({})
    expect(readRuntimePreferenceBackend()).toBeNull()

    vi.restoreAllMocks()
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('Storage is disabled.', 'SecurityError')
    })

    expect(() => writeRuntimePreferences({ pytorch: preference })).not.toThrow()
    expect(() => writeRuntimePreferenceBackend('pytorch')).not.toThrow()
  })
})
