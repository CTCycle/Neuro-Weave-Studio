import type {
  CompilationResult,
  ModelGraph,
  NodeManifest,
  ProjectDocument,
  ProjectExportBundle,
  ProjectSummary,
  TargetFramework,
  ValidationResult,
} from '../domain/types'

const API_ROOT = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/$/, '')

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }))
    throw new ApiError(response.status, JSON.stringify(payload.detail ?? payload))
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

async function requestBlob(path: string, init?: RequestInit): Promise<Blob> {
  const response = await fetch(`${API_ROOT}${path}`, init)
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }))
    throw new ApiError(response.status, JSON.stringify(payload.detail ?? payload))
  }
  return response.blob()
}

export const api = {
  async catalog(): Promise<NodeManifest[]> {
    const response = await request<{ nodes: NodeManifest[] }>('/nodes/catalog')
    return response.nodes
  },
  async listProjects(): Promise<ProjectSummary[]> {
    const response = await request<{ projects: ProjectSummary[] }>('/projects')
    return response.projects
  },
  getProject(id: string): Promise<ProjectDocument> {
    return request(`/projects/${id}`)
  },
  createProject(payload: {
    name: string
    description: string
    target_framework: TargetFramework
    graph: ModelGraph
  }): Promise<ProjectDocument> {
    return request('/projects', { method: 'POST', body: JSON.stringify(payload) })
  },
  updateProject(id: string, payload: {
    name: string
    description: string
    target_framework: TargetFramework
    graph: ModelGraph
    expected_revision: number
  }): Promise<ProjectDocument> {
    return request(`/projects/${id}`, { method: 'PUT', body: JSON.stringify(payload) })
  },
  duplicateProject(id: string, name?: string): Promise<ProjectDocument> {
    return request(`/projects/${id}/duplicate`, {
      method: 'POST',
      body: JSON.stringify({ name: name ?? null }),
    })
  },
  deleteProject(id: string): Promise<void> {
    return request(`/projects/${id}`, { method: 'DELETE' })
  },
  exportProject(id: string): Promise<ProjectExportBundle> {
    return request(`/projects/${id}/export`)
  },
  importProject(bundle: ProjectExportBundle): Promise<ProjectDocument> {
    return request('/projects/import', { method: 'POST', body: JSON.stringify(bundle) })
  },
  validate(graph: ModelGraph, framework: TargetFramework): Promise<ValidationResult> {
    return request('/validation', {
      method: 'POST',
      body: JSON.stringify({ graph, framework }),
    })
  },
  compile(graph: ModelGraph, framework: TargetFramework): Promise<CompilationResult> {
    return request('/compilation', {
      method: 'POST',
      body: JSON.stringify({ graph, framework }),
    })
  },
  sourceForProject(id: string, framework: TargetFramework): Promise<{ code: string }> {
    return request(`/exports/${id}/source/${framework}`)
  },
  exportKerasModel(id: string): Promise<Blob> {
    return requestBlob(`/exports/${id}/keras-model`, { method: 'POST' })
  },
}
