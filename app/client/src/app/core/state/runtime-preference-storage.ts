import type { ExecutionBackend, WorkerConfiguration } from '../../domain/types'
import { isWorkerConfiguration } from './runtime-selection'

export const RUNTIME_PREFERENCES_STORAGE_KEY = 'neuroweave.runtime-preferences.v1'
export const RUNTIME_PREFERENCE_BACKEND_STORAGE_KEY = 'neuroweave.runtime-preference-backend.v1'

type StoredPreferences = Partial<Record<ExecutionBackend, WorkerConfiguration>>

export function readRuntimePreferences(): StoredPreferences {
  const storage = browserStorage()
  if (!storage) return {}
  try {
    const parsed: unknown = JSON.parse(storage.getItem(RUNTIME_PREFERENCES_STORAGE_KEY) || '{}')
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return {}
    return Object.fromEntries(
      Object.entries(parsed).filter(([backend, value]) => isExecutionBackend(backend) && isWorkerConfiguration(value) && value.backend === backend),
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

export function readRuntimePreferenceBackend(): ExecutionBackend | null {
  const storage = browserStorage()
  if (!storage) return null
  try {
    const backend = storage.getItem(RUNTIME_PREFERENCE_BACKEND_STORAGE_KEY)
    return backend && isExecutionBackend(backend) ? backend : null
  } catch {
    return null
  }
}

export function writeRuntimePreferenceBackend(backend: ExecutionBackend | null): void {
  const storage = browserStorage()
  if (!storage) return
  try {
    if (backend) storage.setItem(RUNTIME_PREFERENCE_BACKEND_STORAGE_KEY, backend)
    else storage.removeItem(RUNTIME_PREFERENCE_BACKEND_STORAGE_KEY)
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
