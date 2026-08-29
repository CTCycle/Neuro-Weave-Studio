import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { CompilationCapabilities, ExecutionBackend, ExecutionBackendStatus, NodeManifest, ProjectSummary, RuntimeApiStatus, RuntimeDevice, WorkerConfiguration } from '../domain/types'
import { AppResourcesContext, type AppResources, type ResourceReadiness } from './appResourcesContext'

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
  const [compilationCapabilities, setCompilationCapabilities] = useState<CompilationCapabilities | null>(null)
  const [runtimePreferences, setRuntimePreferences] = useState<Partial<Record<ExecutionBackend, WorkerConfiguration>>>({})
  const [runtimePreferenceBackend, setRuntimePreferenceBackend] = useState<ExecutionBackend | null>(null)
  const [readiness, setReadiness] = useState<ResourceReadiness>({ catalog: false, projects: false, runtime: false })
  const [resourceErrors, setResourceErrors] = useState<Record<string, string>>({})

  useEffect(() => {
    let active = true
    const markReady = (resource: keyof ResourceReadiness) => {
      if (active) setReadiness((current) => ({ ...current, [resource]: true }))
    }
    const recordError = (resource: keyof ResourceReadiness, message: string) => {
      if (active) setResourceErrors((current) => ({ ...current, [resource]: message }))
    }

    void api.catalog().then((value) => {
      if (active) setManifests(value)
    }).catch(() => recordError('catalog', 'Node catalog unavailable.')).finally(() => markReady('catalog'))

    void api.listProjects().then((value) => {
      if (active) setProjects(value)
    }).catch(() => recordError('projects', 'Model catalog unavailable.')).finally(() => markReady('projects'))

    void api.metadata().then((value) => {
      if (!active) return
      const statuses = value.execution_backends
      setBackendStatuses(statuses)
      setCompilationCapabilities(value.configuration.compilation_capabilities)
      setRuntimeApis(value.runtime_apis ?? [{ id: 'keras', available: true }, { id: 'native', available: false, reason_code: 'runtime_unavailable' }])
      setRuntimePreferences((current) => {
        const next = { ...current }
        for (const status of statuses.filter((item) => item.available)) {
          if (!next[status.id]) next[status.id] = defaultPreference(status)
        }
        return next
      })
      setRuntimePreferenceBackend((current) => current && statuses.some((item) => item.id === current && item.available) ? current : preferredBackend(statuses))
    }).catch(() => recordError('runtime', 'Runtime metadata unavailable.')).finally(() => markReady('runtime'))

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

  const error = Object.values(resourceErrors).join(' ') || null
  const value = useMemo<AppResources>(() => ({
    manifests,
    projects,
    backendStatuses,
    runtimeApis,
    compilationCapabilities,
    runtimePreferences,
    runtimePreferenceBackend,
    readiness,
    error,
    refreshProjects: async () => {
      const next = await api.listProjects()
      setProjects(next)
      return next
    },
    clearError: () => setResourceErrors({}),
    setRuntimePreferenceBackend: selectRuntimePreferenceBackend,
    setRuntimePreference,
  }), [backendStatuses, compilationCapabilities, error, manifests, projects, readiness, runtimeApis, runtimePreferences, runtimePreferenceBackend, selectRuntimePreferenceBackend, setRuntimePreference])

  return <AppResourcesContext.Provider value={value}>{children}</AppResourcesContext.Provider>
}
