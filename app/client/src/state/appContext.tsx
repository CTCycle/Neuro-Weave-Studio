import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { ExecutionBackend, ExecutionBackendStatus, NodeManifest, ProjectSummary } from '../domain/types'
import { AppResourcesContext, type AppResources } from './appResourcesContext'

export function AppResourcesProvider({ children }: { children: ReactNode }) {
  const [manifests, setManifests] = useState<NodeManifest[]>([])
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [backendStatuses, setBackendStatuses] = useState<ExecutionBackendStatus[]>([])
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
        setSelectedExecutionBackend((current) => {
          const available = statuses.filter((item) => item.available)
          if (current && available.some((item) => item.id === current)) return current
          return available.length === 1 ? available[0].id : null
        })
      }
      else errors.push('Runtime metadata unavailable.')
      setError(errors.length ? errors.join(' ') : null)
      setLoading(false)
    })
    return () => { active = false }
  }, [])

  const value = useMemo<AppResources>(() => ({
    manifests,
    projects,
    backendStatuses,
    selectedExecutionBackend,
    loading,
    error,
    refreshProjects: async () => {
      const next = await api.listProjects()
      setProjects(next)
      return next
    },
    clearError: () => setError(null),
    setSelectedExecutionBackend,
  }), [backendStatuses, error, loading, manifests, projects, selectedExecutionBackend])

  return <AppResourcesContext.Provider value={value}>{children}</AppResourcesContext.Provider>
}
