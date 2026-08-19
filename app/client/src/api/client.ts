import type {
  CompilationResult,
  MetadataResponse,
  ModelGraph,
  NodeManifest,
  ProjectDocument,
  ProjectExportBundle,
  ProjectSummary,
  CodeTarget,
  ValidationResult,
  JobResponse,
  WorkerConfiguration,
} from '../domain/types'

const ABSOLUTE_URL_PATTERN = /^[a-zA-Z][a-zA-Z\d+\-.]*:\/\//

function normalizeApiBase(rawValue: string | undefined): string {
  const candidate = (rawValue || '/api').trim()
  if (!candidate) return '/api'
  if (ABSOLUTE_URL_PATTERN.test(candidate)) {
    throw new Error('VITE_API_BASE_URL must be a relative path (for example /api).')
  }
  const withLeadingSlash = candidate.startsWith('/') ? candidate : `/${candidate}`
  return withLeadingSlash.length > 1 && withLeadingSlash.endsWith('/')
    ? withLeadingSlash.slice(0, -1)
    : withLeadingSlash
}

const API_BASE = normalizeApiBase(import.meta.env.VITE_API_BASE_URL)

export class ApiError extends Error {
  constructor(public status: number, message: string, public detail: unknown = null) {
    super(message)
    this.name = 'ApiError'
  }
}

function errorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === 'object' && detail !== null && 'message' in detail) {
    const message = (detail as { message?: unknown }).message
    if (typeof message === 'string') return message
  }
  return typeof detail === 'string' ? detail : fallback
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }))
    const detail = payload.detail ?? payload
    throw new ApiError(response.status, errorMessage(detail, response.statusText), detail)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

async function requestBlob(path: string, init?: RequestInit): Promise<Blob> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }))
    const detail = payload.detail ?? payload
    throw new ApiError(response.status, errorMessage(detail, response.statusText), detail)
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
    code_target: CodeTarget
    graph: ModelGraph
  }): Promise<ProjectDocument> {
    return request('/projects', { method: 'POST', body: JSON.stringify(payload) })
  },
  updateProject(id: string, payload: {
    name: string
    description: string
    code_target: CodeTarget
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
  validate(graph: ModelGraph, code_target: CodeTarget): Promise<ValidationResult> {
    return request('/validation', {
      method: 'POST',
      body: JSON.stringify({ graph, code_target }),
    })
  },
  compile(graph: ModelGraph, code_target: CodeTarget): Promise<CompilationResult> {
    return request('/compilation', {
      method: 'POST',
      body: JSON.stringify({ graph, code_target }),
    })
  },
  submitSmokeInference(graph: ModelGraph, worker: WorkerConfiguration): Promise<JobResponse> {
    return request('/execution/jobs', {
      method: 'POST',
      body: JSON.stringify({ operation: { type: 'smoke_inference', graph }, worker }),
    })
  },
  submitKerasExport(project_id: string, worker: WorkerConfiguration): Promise<JobResponse> {
    return request('/execution/jobs', {
      method: 'POST',
      body: JSON.stringify({ operation: { type: 'keras_export', project_id }, worker }),
    })
  },
  listJobs(active_only = false): Promise<JobResponse[]> {
    return request(`/execution/jobs?active_only=${active_only ? 'true' : 'false'}`)
  },
  getJob(jobId: string): Promise<JobResponse> {
    return request(`/execution/jobs/${jobId}`)
  },
  stopJob(jobId: string): Promise<JobResponse> {
    return request(`/execution/jobs/${jobId}/stop`, { method: 'POST' })
  },
  jobArtifact(jobId: string): Promise<Blob> {
    return requestBlob(`/execution/jobs/${jobId}/artifact`)
  },
  metadata(): Promise<MetadataResponse> {
    return request('/metadata')
  },
  sourceForProject(id: string, code_target: CodeTarget): Promise<{ code: string }> {
    return request(`/exports/${id}/source/${code_target}`)
  },
}
