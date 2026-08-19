import { createContext } from 'react'
import type { ExecutionBackend, ExecutionBackendStatus, NodeManifest, ProjectSummary, RuntimeApiStatus, WorkerConfiguration } from '../domain/types'

export interface AppResources {
  manifests: NodeManifest[]
  projects: ProjectSummary[]
  backendStatuses: ExecutionBackendStatus[]
  runtimeApis: RuntimeApiStatus[]
  runtimeSelection: WorkerConfiguration | null
  selectedExecutionBackend: ExecutionBackend | null
  loading: boolean
  error: string | null
  setSelectedExecutionBackend: (backend: ExecutionBackend | null) => void
  setRuntimeSelection: (selection: WorkerConfiguration | null) => void
  refreshProjects: () => Promise<ProjectSummary[]>
  clearError: () => void
}

export const AppResourcesContext = createContext<AppResources | null>(null)
