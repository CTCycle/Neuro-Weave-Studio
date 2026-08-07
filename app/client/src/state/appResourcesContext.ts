import { createContext } from 'react'
import type { ExecutionBackendStatus, NodeManifest, ProjectSummary } from '../domain/types'

export interface AppResources {
  manifests: NodeManifest[]
  projects: ProjectSummary[]
  backendStatuses: ExecutionBackendStatus[]
  loading: boolean
  error: string | null
  refreshProjects: () => Promise<ProjectSummary[]>
  clearError: () => void
}

export const AppResourcesContext = createContext<AppResources | null>(null)
