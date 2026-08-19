import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { ExecutionBackend, ExecutionBackendStatus, NodeManifest, ProjectSummary, RuntimeApiStatus, WorkerConfiguration } from '../domain/types'
import { AppResourcesContext, type AppResources } from './appResourcesContext'

export function AppResourcesProvider({ children }: { children: ReactNode }) {
  const [manifests, setManifests] = useState<NodeManifest[]>([])
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [backendStatuses, setBackendStatuses] = useState<ExecutionBackendStatus[]>([])
  const [runtimeApis, setRuntimeApis] = useState<RuntimeApiStatus[]>([])
  const [runtimeSelection, setRuntimeSelection] = useState<WorkerConfiguration | null>(null)
  const [selectedExecutionBackend, setSelectedExecutionBackend] = useState<ExecutionBackend | null>(null)
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
        setRuntimeSelection((current) => {
          const available = statuses.filter((item) => item.available)
          if (current && available.some((item) => item.id === current.backend)) return current
          const first = available.length === 1 ? available[0] : available.find((item) => item.id === 'pytorch')
          const device = first?.devices?.[0]
          return first && device ? {
            api: 'keras',
            backend: first.id,
            device_id: device.device_id,
            memory_policy: { kind: 'default' },
            precision: 'float32',
          } : null
        })
        setSelectedExecutionBackend((current) => current && statuses.some((item) => item.id === current && item.available) ? current : null)
      }
      else errors.push('Runtime metadata unavailable.')
      setError(errors.length ? errors.join(' ') : null)
      setLoading(false)
    })
    return () => { active = false }
  }, [])

  const selectBackend = useCallback((backend: ExecutionBackend | null) => {
    setSelectedExecutionBackend(backend)
    if (!backend) {
      setRuntimeSelection(null)
      return
    }
    const status = backendStatuses.find((item) => item.id === backend)
    const device = status?.devices?.[0]
    if (status && device) {
      setRuntimeSelection((current) => ({
        api: current?.api ?? 'keras',
        backend,
        device_id: device.device_id,
        memory_policy: { kind: 'default' },
        precision: current?.precision ?? 'float32',
      }))
    }
  }, [backendStatuses])

  const value = useMemo<AppResources>(() => ({
    manifests,
    projects,
    backendStatuses,
    runtimeApis,
    runtimeSelection,
    selectedExecutionBackend,
    loading,
    error,
    refreshProjects: async () => {
      const next = await api.listProjects()
      setProjects(next)
      return next
    },
    clearError: () => setError(null),
    setSelectedExecutionBackend: selectBackend,
    setRuntimeSelection,
  }), [backendStatuses, error, loading, manifests, projects, runtimeApis, runtimeSelection, selectedExecutionBackend, selectBackend])

  return <AppResourcesContext.Provider value={value}>{children}</AppResourcesContext.Provider>
}
