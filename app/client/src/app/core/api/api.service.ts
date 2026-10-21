// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http'
import { Injectable, inject } from '@angular/core'
import { firstValueFrom } from 'rxjs'
import { ApiError, apiErrorMessage } from './api-error'
import type { FilesystemBrowseResponse } from '../../domain/filesystem.types'
import type { ApplicationSettings, DatabaseSettings, DatabaseStatus, DatabaseSwitchResult, DatabaseTestResult } from '../../domain/settings.types'
import type {
  DatasetDefinitionMutation,
  DatasetDocument,
  DatasetInspection,
  DatasetInspectionRequest,
  DatasetSummary,
  DatasetValidationResult,
} from '../../domain/dataset.types'
import type {
  CompilationResult,
  ConnectionValidationResult,
  ExecutionBackend,
  ExportCapabilitiesResponse,
  ExportFormat,
  ExportOptions,
  ModelGraph,
  JobResponse,
  KerasArchitectureImportResponse,
  MetadataResponse,
  MetricCatalogResponse,
  MetricDefinition,
  NodeManifest,
  ProjectDocument,
  ProjectExportBundle,
  ProjectSummary,
  RuntimeDiscoveryResponse,
  TensorInspectionNode,
  TensorInspectionSession,
  TensorInspectionViewRequest,
  TensorInspectionViewResponse,
  ValidationResult,
  WorkerConfiguration,
} from '../../domain/types'

const API_BASE = '/api'
const CONNECTION_CANDIDATE_BATCH_SIZE = 256
const DEFAULT_REQUEST_TIMEOUT_MS = 10_000
const GRAPH_VALIDATION_TIMEOUT_MS = 60_000
const DATASET_OPERATION_TIMEOUT_MS = 60_000
const ARCHITECTURE_IMPORT_TIMEOUT_MS = 60_000
const ARTIFACT_DOWNLOAD_TIMEOUT_MS = 60_000

interface RequestOptions {
  timeoutMs?: number
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient)
  private readonly jsonHeaders = new HttpHeaders({ 'Content-Type': 'application/json' })

  private async request<T>(path: string, method = 'GET', body?: unknown, options: RequestOptions = {}): Promise<T> {
    try {
      return await firstValueFrom(this.http.request<T>(method, `${API_BASE}${path}`, {
        body,
        headers: this.jsonHeaders,
        timeout: options.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      }))
    } catch (error) {
      throw this.toApiError(error)
    }
  }

  private async requestBlob(path: string, options: RequestOptions = {}): Promise<Blob> {
    try {
      return await firstValueFrom(this.http.get(`${API_BASE}${path}`, {
        responseType: 'blob',
        timeout: options.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      }))
    } catch (error) {
      throw this.toApiError(error)
    }
  }

  private toApiError(error: unknown): ApiError {
    if (error instanceof ApiError) return error
    if (error instanceof HttpErrorResponse) {
      const payload = error.error as { detail?: unknown } | unknown
      const detail = typeof payload === 'object' && payload !== null && 'detail' in payload
        ? (payload as { detail?: unknown }).detail
        : payload
      return new ApiError(error.status, apiErrorMessage(detail, error.statusText || 'Request failed.'), detail)
    }
    return error instanceof Error ? new ApiError(0, error.message, error) : new ApiError(0, String(error), error)
  }

  async catalog(): Promise<NodeManifest[]> {
    const response = await this.request<{ nodes: NodeManifest[] }>('/nodes/catalog')
    return response.nodes
  }

  getApplicationSettings(): Promise<ApplicationSettings> {
    return this.request<ApplicationSettings>('/settings')
  }

  patchApplicationSettings(settings: Partial<ApplicationSettings>): Promise<ApplicationSettings> {
    return this.request<ApplicationSettings>('/settings', 'PATCH', settings)
  }

  getDatabaseStatus(): Promise<DatabaseStatus> {
    return this.request<DatabaseStatus>('/settings/database')
  }

  testDatabase(settings: DatabaseSettings): Promise<DatabaseTestResult> {
    return this.request<DatabaseTestResult>('/settings/database/test', 'POST', settings, { timeoutMs: 120_000 })
  }

  switchDatabase(settings: DatabaseSettings): Promise<DatabaseSwitchResult> {
    return this.request<DatabaseSwitchResult>('/settings/database', 'PUT', settings, { timeoutMs: 120_000 })
  }

  async listProjects(): Promise<ProjectSummary[]> {
    const response = await this.request<{ projects: ProjectSummary[] }>('/projects')
    return response.projects
  }

  async listDatasets(): Promise<DatasetSummary[]> {
    const response = await this.request<{ datasets: DatasetSummary[] }>('/datasets')
    return response.datasets
  }

  inspectDataset(payload: DatasetInspectionRequest): Promise<DatasetInspection> {
    return this.request('/datasets/inspect', 'POST', payload, { timeoutMs: DATASET_OPERATION_TIMEOUT_MS })
  }

  validateDataset(payload: DatasetDefinitionMutation): Promise<DatasetValidationResult> {
    return this.request('/datasets/validate', 'POST', payload, { timeoutMs: DATASET_OPERATION_TIMEOUT_MS })
  }

  createDataset(payload: DatasetDefinitionMutation): Promise<DatasetDocument> {
    return this.request('/datasets', 'POST', payload, { timeoutMs: DATASET_OPERATION_TIMEOUT_MS })
  }

  getDataset(id: string): Promise<DatasetDocument> {
    return this.request(`/datasets/${encodeURIComponent(id)}`)
  }

  updateDataset(id: string, payload: DatasetDefinitionMutation): Promise<DatasetDocument> {
    return this.request(`/datasets/${encodeURIComponent(id)}`, 'PUT', payload, { timeoutMs: DATASET_OPERATION_TIMEOUT_MS })
  }

  revalidateDataset(id: string): Promise<DatasetDocument> {
    return this.request(`/datasets/${encodeURIComponent(id)}/validate`, 'POST', undefined, { timeoutMs: DATASET_OPERATION_TIMEOUT_MS })
  }

  deleteDataset(id: string): Promise<void> {
    return this.request(`/datasets/${encodeURIComponent(id)}`, 'DELETE')
  }

  getProject(id: string): Promise<ProjectDocument> {
    return this.request(`/projects/${id}`)
  }

  createProject(payload: {
    name: string
    description: string
    execution_backend: ExecutionBackend
    keras_enabled: boolean
    graph: ModelGraph
  }): Promise<ProjectDocument> {
    return this.request('/projects', 'POST', payload)
  }

  updateProject(id: string, payload: {
    name: string
    description: string
    execution_backend: ExecutionBackend
    keras_enabled: boolean
    graph: ModelGraph
    expected_revision: number
  }): Promise<ProjectDocument> {
    return this.request(`/projects/${id}`, 'PUT', payload)
  }

  duplicateProject(id: string, name?: string): Promise<ProjectDocument> {
    return this.request(`/projects/${id}/duplicate`, 'POST', { name: name ?? null })
  }

  deleteProject(id: string): Promise<void> {
    return this.request(`/projects/${id}`, 'DELETE')
  }

  exportProject(id: string): Promise<ProjectExportBundle> {
    return this.request(`/projects/${id}/export`)
  }

  importProject(bundle: ProjectExportBundle): Promise<ProjectDocument> {
    return this.request('/projects/import', 'POST', bundle)
  }

  async importKerasArchitecture(file: Blob, execution_backend: ExecutionBackend, name?: string): Promise<KerasArchitectureImportResponse> {
    const query = new URLSearchParams({ execution_backend })
    if (name) query.set('name', name)
    try {
      return await firstValueFrom(this.http.post<KerasArchitectureImportResponse>(
        `${API_BASE}/imports/keras-architecture?${query.toString()}`,
        file,
        {
          headers: new HttpHeaders({ 'Content-Type': 'application/vnd.keras.zip' }),
          timeout: ARCHITECTURE_IMPORT_TIMEOUT_MS,
        },
      ))
    } catch (error) {
      throw this.toApiError(error)
    }
  }

  validate(graph: ModelGraph, execution_backend: ExecutionBackend, keras_enabled: boolean): Promise<ValidationResult> {
    return this.request('/validation', 'POST', { graph, execution_backend, keras_enabled }, { timeoutMs: GRAPH_VALIDATION_TIMEOUT_MS })
  }

  validateConnection(graph: ModelGraph, connection: ModelGraph['edges'][number], execution_backend: ExecutionBackend, keras_enabled: boolean): Promise<ConnectionValidationResult> {
    return this.request('/validation/connection', 'POST', { graph, connection, execution_backend, keras_enabled })
  }

  async validateConnectionCandidates(graph: ModelGraph, connections: ModelGraph['edges'], execution_backend: ExecutionBackend, keras_enabled: boolean): Promise<{ valid_connection_ids: string[] }> {
    const valid_connection_ids: string[] = []
    for (let offset = 0; offset < connections.length; offset += CONNECTION_CANDIDATE_BATCH_SIZE) {
      const response = await this.request<{ valid_connection_ids: string[] }>('/validation/connection-candidates', 'POST', {
        graph,
        connections: connections.slice(offset, offset + CONNECTION_CANDIDATE_BATCH_SIZE),
        execution_backend,
        keras_enabled,
      })
      valid_connection_ids.push(...response.valid_connection_ids)
    }
    return { valid_connection_ids }
  }

  compile(graph: ModelGraph, execution_backend: ExecutionBackend, keras_enabled: boolean): Promise<CompilationResult> {
    return this.request('/compilation', 'POST', { graph, execution_backend, keras_enabled })
  }

  submitFunctionalValidation(project_id: string, graph: ModelGraph, worker: WorkerConfiguration): Promise<JobResponse> {
    return this.request('/execution/jobs', 'POST', { operation: { type: 'smoke_inference', project_id, graph }, worker })
  }

  submitTensorInspection(project_id: string, project_revision: number, seed: number, worker: WorkerConfiguration): Promise<JobResponse> {
    return this.request('/execution/jobs', 'POST', { operation: { type: 'tensor_inspection', project_id, project_revision, seed }, worker })
  }

  tensorInspection(sessionId: string): Promise<TensorInspectionSession> {
    return this.request(`/tensor-inspections/${encodeURIComponent(sessionId)}`)
  }

  tensorInspectionNode(sessionId: string, nodeId: string): Promise<TensorInspectionNode> {
    return this.request(`/tensor-inspections/${encodeURIComponent(sessionId)}/nodes/${encodeURIComponent(nodeId)}`)
  }

  tensorInspectionView(sessionId: string, payload: TensorInspectionViewRequest): Promise<TensorInspectionViewResponse> {
    return this.request(`/tensor-inspections/${encodeURIComponent(sessionId)}/view`, 'POST', payload)
  }

  deleteTensorInspection(sessionId: string): Promise<void> {
    return this.request(`/tensor-inspections/${encodeURIComponent(sessionId)}`, 'DELETE')
  }

  submitModelExport(project_id: string, format: ExportFormat, filename_stem: string, options: ExportOptions, worker: WorkerConfiguration): Promise<JobResponse> {
    return this.request('/execution/jobs', 'POST', { operation: { type: 'model_export', project_id, format, filename_stem, options }, worker })
  }

  listJobs(active_only = false): Promise<JobResponse[]> {
    return this.request(`/execution/jobs?active_only=${active_only ? 'true' : 'false'}`)
  }

  getJob(jobId: string): Promise<JobResponse> {
    return this.request(`/execution/jobs/${jobId}`)
  }

  stopJob(jobId: string): Promise<JobResponse> {
    return this.request(`/execution/jobs/${jobId}/stop`, 'POST')
  }

  jobArtifact(jobId: string): Promise<Blob> {
    return this.requestBlob(`/execution/jobs/${jobId}/artifact`, { timeoutMs: ARTIFACT_DOWNLOAD_TIMEOUT_MS })
  }

  metadata(): Promise<MetadataResponse> {
    return this.request('/metadata')
  }

  requestRuntimeDiscovery(backend: ExecutionBackend, api: WorkerConfiguration['api'], force_refresh = false): Promise<RuntimeDiscoveryResponse> {
    return this.request('/execution/discovery', 'POST', { backend, api, force_refresh })
  }

  runtimeDiscoveryStatus(backend: ExecutionBackend, api: WorkerConfiguration['api']): Promise<RuntimeDiscoveryResponse> {
    return this.request(`/execution/discovery?backend=${encodeURIComponent(backend)}&api=${encodeURIComponent(api)}`)
  }

  sourceForProject(id: string): Promise<{ code: string }> {
    return this.request(`/exports/${id}/source`)
  }

  exportCapabilities(id: string): Promise<ExportCapabilitiesResponse> {
    return this.request(`/exports/${id}/capabilities`)
  }

  browseFilesystem(path = '', includeFiles = true, extensions: string[] = []): Promise<FilesystemBrowseResponse> {
    const query = new URLSearchParams({
      path,
      include_files: includeFiles ? 'true' : 'false',
      extensions: extensions.join(','),
    })
    return this.request(`/filesystem/browse?${query.toString()}`, 'GET', undefined, { timeoutMs: DATASET_OPERATION_TIMEOUT_MS })
  }

  metrics(): Promise<MetricCatalogResponse> {
    return this.request('/metrics')
  }

  createMetric(payload: Pick<MetricDefinition, 'kind' | 'label' | 'description'> & { formula: string }): Promise<MetricDefinition> {
    return this.request('/metrics', 'POST', payload)
  }

  updateMetric(id: string, payload: Pick<MetricDefinition, 'label' | 'description'> & { formula: string }): Promise<MetricDefinition> {
    return this.request(`/metrics/${encodeURIComponent(id)}`, 'PUT', payload)
  }

  deleteMetric(id: string): Promise<void> {
    return this.request(`/metrics/${encodeURIComponent(id)}`, 'DELETE')
  }
}
