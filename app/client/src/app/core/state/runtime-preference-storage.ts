import type { ExecutionBackend, WorkerConfiguration } from '../../domain/types'

export const RUNTIME_PREFERENCES_STORAGE_KEY = 'neuroweave.runtime-preferences.v1'

type StoredPreferences = Partial<Record<ExecutionBackend, WorkerConfiguration>>

export function readRuntimePreferences(): StoredPreferences {
  const storage = browserStorage()
  if (!storage) return {}
  try {
    const parsed: unknown = JSON.parse(storage.getItem(RUNTIME_PREFERENCES_STORAGE_KEY) || '{}')
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return {}
    return Object.fromEntries(
      Object.entries(parsed).filter(([backend, value]) => isExecutionBackend(backend) && typeof value === 'object' && value !== null),
    ) as StoredPreferences
  } catch {
    return {}
  }
}

export function writeRuntimePreferences(preferences: StoredPreferences): void {
  const storage = browserStorage()
  if (!storage) return
  try {
    storage.setItem(RUNTIME_PREFERENCES_STORAGE_KEY, JSON.stringify(preferences))
  } catch {
    // Storage can be unavailable in privacy-restricted browser contexts.
  }
}

export function storedPreferenceBackend(preferences: StoredPreferences): ExecutionBackend | null {
  for (const backend of ['pytorch', 'tensorflow', 'jax'] as const) if (preferences[backend]) return backend
  return null
}

function browserStorage(): Storage | null {
  try {
    return typeof window === 'undefined' ? null : window.localStorage
  } catch {
    return null
  }
}

function isExecutionBackend(value: string): value is ExecutionBackend {
  return value === 'tensorflow' || value === 'pytorch' || value === 'jax'
}
