import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { ExecutionBackend, ExecutionBackendStatus, NodeManifest, ProjectSummary, RuntimeApiStatus, RuntimeDevice, WorkerConfiguration } from '../domain/types'
import { AppResourcesContext, type AppResources } from './appResourcesContext'

function defaultPreference(status: ExecutionBackendStatus): WorkerConfiguration {
  const device: RuntimeDevice = status.devices?.[0] ?? {
    device_id: 'cpu',
    lease_key: 'cpu',
    kind: 'cpu',
    name: 'CPU',
    accelerator: 'cpu',
    available: true,
    supported_memory_policies: ['default'],
  }
  return {
    api: 'keras',
    backend: status.id,
    device_id: device.device_id,
    memory_policy: { kind: 'default' },
    precision: 'float32',
  }
}

function preferredBackend(statuses: ExecutionBackendStatus[]): ExecutionBackend | null {
  const available = statuses.filter((item) => item.available)
  return (available.length === 1 ? available[0] : available.find((item) => item.id === 'pytorch'))?.id ?? null
}

export function AppResourcesProvider({ children }: { children: ReactNode }) {
  const [manifests, setManifests] = useState<NodeManifest[]>([])
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [backendStatuses, setBackendStatuses] = useState<ExecutionBackendStatus[]>([])
  const [runtimeApis, setRuntimeApis] = useState<RuntimeApiStatus[]>([])
  const [runtimePreferences, setRuntimePreferences] = useState<Partial<Record<ExecutionBackend, WorkerConfiguration>>>({})
  const [runtimePreferenceBackend, setRuntimePreferenceBackend] = useState<ExecutionBackend | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    Promise.allSettled([api.catalog(), api.listProjects(), api.metadata()]).then((results) => {
      if (!active) return
      const [catalogResult, projectsResult, metadataResult] = results
      const errors: string[] = []
      if (catalogResult.status === 'fulfilled') {
        setManifests(catalogResult.value)
      } else errors.push('Node catalog unavailable.')
      if (projectsResult.status === 'fulfilled') setProjects(projectsResult.value)
      else errors.push('Model catalog unavailable.')
      if (metadataResult.status === 'fulfilled') {
        const statuses = metadataResult.value.execution_backends
        setBackendStatuses(statuses)
        setRuntimeApis(metadataResult.value.runtime_apis ?? [{ id: 'keras', available: true }, { id: 'native', available: false, reason_code: 'not_implemented' }])
        setRuntimePreferences((current) => {
          const next = { ...current }
          for (const status of statuses.filter((item) => item.available)) {
            if (!next[status.id]) next[status.id] = defaultPreference(status)
          }
          return next
        })
        setRuntimePreferenceBackend((current) => current && statuses.some((item) => item.id === current && item.available) ? current : preferredBackend(statuses))
      } else errors.push('Runtime metadata unavailable.')
      setError(errors.length ? errors.join(' ') : null)
      setLoading(false)
    })
    return () => { active = false }
  }, [])

  const selectRuntimePreferenceBackend = useCallback((backend: ExecutionBackend | null) => {
    setRuntimePreferenceBackend(backend)
    if (!backend) return
    const status = backendStatuses.find((item) => item.id === backend)
    if (!status) return
    setRuntimePreferences((current) => current[backend] ? current : { ...current, [backend]: defaultPreference(status) })
  }, [backendStatuses])

  const setRuntimePreference = useCallback((selection: WorkerConfiguration) => {
    setRuntimePreferences((current) => ({ ...current, [selection.backend]: selection }))
    setRuntimePreferenceBackend(selection.backend)
  }, [])

  const value = useMemo<AppResources>(() => ({
    manifests,
    projects,
    backendStatuses,
    runtimeApis,
    runtimePreferences,
    runtimePreferenceBackend,
    loading,
    error,
    refreshProjects: async () => {
      const next = await api.listProjects()
      setProjects(next)
      return next
    },
    clearError: () => setError(null),
    setRuntimePreferenceBackend: selectRuntimePreferenceBackend,
    setRuntimePreference,
  }), [backendStatuses, error, loading, manifests, projects, runtimeApis, runtimePreferences, runtimePreferenceBackend, selectRuntimePreferenceBackend, setRuntimePreference])

  return <AppResourcesContext.Provider value={value}>{children}</AppResourcesContext.Provider>
}
