import { createContext } from 'react'
import type { CompilationCatalog, ExecutionBackend, ExecutionBackendStatus, NodeManifest, ProjectSummary, RuntimeApiStatus, WorkerConfiguration } from '../domain/types'

export interface AppResources {
  manifests: NodeManifest[]
  projects: ProjectSummary[]
  backendStatuses: ExecutionBackendStatus[]
  runtimeApis: RuntimeApiStatus[]
  compilationCatalog?: CompilationCatalog | null
  runtimePreferences: Partial<Record<ExecutionBackend, WorkerConfiguration>>
  runtimePreferenceBackend: ExecutionBackend | null
  loading: boolean
  error: string | null
  setRuntimePreferenceBackend: (backend: ExecutionBackend | null) => void
  setRuntimePreference: (selection: WorkerConfiguration) => void
  refreshProjects: () => Promise<ProjectSummary[]>
  clearError: () => void
}

export const AppResourcesContext = createContext<AppResources | null>(null)
