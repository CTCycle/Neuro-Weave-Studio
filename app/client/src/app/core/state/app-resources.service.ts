import { Injectable, computed, inject, signal } from '@angular/core'
import { ApiService } from '../api/api.service'
import type {
  CompilationCapabilities,
  ExecutionBackend,
  ExecutionBackendStatus,
  MetricCatalogResponse,
  MetricDefinition,
  NodeManifest,
  ProjectSummary,
  RuntimeApi,
  RuntimeApiStatus,
  RuntimeDiscoveryResponse,
  WorkerConfiguration,
} from '../../domain/types'
import type { DatasetSummary } from '../../domain/dataset.types'
import {
  defaultRuntimePreference,
  isRuntimePreferenceValid,
  preferredBackend,
  preferredBackendForApi as selectPreferredBackendForApi,
} from './runtime-selection'
import { readRuntimePreferences, storedPreferenceBackend, writeRuntimePreferences } from './runtime-preference-storage'

@Injectable({ providedIn: 'root' })
export class AppResourcesService {
  private readonly api = inject(ApiService)
  private readonly storedPreferences = readRuntimePreferences()

  readonly manifests = signal<NodeManifest[]>([])
  readonly projects = signal<ProjectSummary[]>([])
  readonly datasets = signal<DatasetSummary[]>([])
  readonly backendStatuses = signal<ExecutionBackendStatus[]>([])
  readonly runtimeApis = signal<RuntimeApiStatus[]>([])
  readonly compilationCapabilities = signal<CompilationCapabilities | null>(null)
  readonly metricDefinitions = signal<MetricDefinition[]>([])
  readonly runtimePreferences = signal<Partial<Record<ExecutionBackend, WorkerConfiguration>>>(this.storedPreferences)
  readonly runtimePreferenceBackend = signal<ExecutionBackend | null>(storedPreferenceBackend(this.storedPreferences))
  readonly catalogReady = signal(false)
  readonly projectsReady = signal(false)
  readonly datasetsReady = signal(false)
  readonly runtimeReady = signal(false)
  readonly metricsReady = signal(false)
  readonly errors = signal<Record<string, string>>({})
  readonly error = computed(() => Object.values(this.errors()).join(' ') || null)

  private projectsRequest?: Promise<ProjectSummary[]>
  private datasetsRequest?: Promise<DatasetSummary[]>
  private catalogRequest?: Promise<NodeManifest[]>
  private runtimeRequest?: Promise<void>
  private metricsRequest?: Promise<MetricDefinition[]>
  private readonly discoveryRequests = new Map<string, Promise<RuntimeDiscoveryResponse>>()
  private readonly validatedDiscoveries = new Set<string>()
  readonly discoveryPending = signal<Record<string, boolean>>({})

  loadProjects(): Promise<ProjectSummary[]> {
    if (this.projectsRequest) return this.projectsRequest
    this.projectsRequest = this.api.listProjects().then((value) => {
      this.projects.set(value)
      this.clearResourceError('projects')
      return value
    }).catch((reason) => {
      this.recordError('projects', this.message(reason, 'Model catalog unavailable.'))
      return []
    }).finally(() => {
      this.projectsReady.set(true)
      markResource('projects')
    })
    return this.projectsRequest
  }

  loadDatasets(): Promise<DatasetSummary[]> {
    if (this.datasetsRequest) return this.datasetsRequest
    this.datasetsRequest = this.api.listDatasets().then((value) => {
      this.datasets.set(value)
      this.clearResourceError('datasets')
      return value
    }).catch((reason) => {
      this.recordError('datasets', this.message(reason, 'Dataset catalog unavailable.'))
      return []
    }).finally(() => {
      this.datasetsReady.set(true)
      markResource('datasets')
    })
    return this.datasetsRequest
  }

  loadCatalog(): Promise<NodeManifest[]> {
    if (this.catalogRequest) return this.catalogRequest
    this.catalogRequest = this.api.catalog().then((value) => {
      this.manifests.set(value)
      this.clearResourceError('catalog')
      return value
    }).catch((reason) => {
      this.recordError('catalog', this.message(reason, 'Node catalog unavailable.'))
      return []
    }).finally(() => {
      this.catalogReady.set(true)
      markResource('catalog')
    })
    return this.catalogRequest
  }

  loadRuntimeMetadata(): Promise<void> {
    if (this.runtimeRequest) return this.runtimeRequest
    this.runtimeRequest = this.api.metadata().then((value) => {
      const statuses = value.execution_backends
      this.backendStatuses.set(statuses)
      this.compilationCapabilities.set(value.configuration.compilation_capabilities)
      this.runtimeApis.set(value.runtime_apis ?? [])
      const current = this.runtimePreferences()
      const next = { ...current }
      for (const status of statuses) {
        const existing = next[status.id]
        if (!status.available) {
          delete next[status.id]
          continue
        }
        if (!existing || !isRuntimePreferenceValid(status, existing)) {
          const preference = defaultRuntimePreference(status)
          if (preference) next[status.id] = preference
          else delete next[status.id]
        }
      }
      this.runtimePreferences.set(next)
      writeRuntimePreferences(next)
      const selected = this.runtimePreferenceBackend()
      this.runtimePreferenceBackend.set(
        selected && next[selected] && statuses.some((item) => item.id === selected && item.available)
          ? selected
          : preferredBackend(statuses),
      )
      for (const status of statuses) {
        for (const api of status.runtime_apis ?? []) {
          if (api.state === 'unchecked' || api.state === undefined) this.validatedDiscoveries.delete(discoveryKey(status.id, api.id))
        }
      }
      this.clearResourceError('runtime')
    }).catch((reason) => this.recordError('runtime', this.message(reason, 'Runtime metadata unavailable.'))).finally(() => {
      this.runtimeReady.set(true)
      markResource('runtime')
    })
    return this.runtimeRequest
  }

  loadMetrics(): Promise<MetricDefinition[]> {
    if (this.metricsRequest) return this.metricsRequest
    this.metricsRequest = this.api.metrics().then((value: MetricCatalogResponse) => {
      this.metricDefinitions.set(value.definitions)
      this.clearResourceError('metrics')
      return value.definitions
    }).catch((reason) => {
      this.recordError('metrics', this.message(reason, 'Metric catalog unavailable.'))
      return []
    }).finally(() => {
      this.metricsReady.set(true)
      markResource('metrics')
    })
    return this.metricsRequest
  }

  discoverRuntime(backend: ExecutionBackend, api: WorkerConfiguration['api'], forceRefresh = false, signal?: AbortSignal): Promise<RuntimeDiscoveryResponse> {
    const key = discoveryKey(backend, api)
    let observation = this.discoveryRequests.get(key)
    if (!observation) {
      this.setDiscoveryPending(key, true)
      observation = this.observeRuntime(backend, api, forceRefresh)
      this.discoveryRequests.set(key, observation)
      void observation.then((result) => {
        if (result.state !== 'checking') this.validatedDiscoveries.add(key)
        this.clearDiscoveryObservation(key, observation as Promise<RuntimeDiscoveryResponse>)
      }, () => this.clearDiscoveryObservation(key, observation as Promise<RuntimeDiscoveryResponse>))
    }
    return signal ? observeWithAbort(observation, signal) : observation
  }

  private async observeRuntime(backend: ExecutionBackend, api: WorkerConfiguration['api'], forceRefresh: boolean): Promise<RuntimeDiscoveryResponse> {
    const requested = await this.api.requestRuntimeDiscovery(backend, api, forceRefresh)
    let current = requested
    const deadline = Date.now() + 65_000
    while (current.state === 'checking' && Date.now() < deadline) {
      await waitForDiscoveryPoll()
      current = await this.api.runtimeDiscoveryStatus(backend, api)
    }
    if (current.state === 'checking') throw new Error('Runtime capability discovery timed out after 65 seconds.')
    this.runtimeRequest = undefined
    await this.loadRuntimeMetadata()
    return current
  }

  private clearDiscoveryObservation(key: string, observation: Promise<RuntimeDiscoveryResponse>): void {
    if (this.discoveryRequests.get(key) === observation) this.discoveryRequests.delete(key)
    this.setDiscoveryPending(key, false)
  }

  runtimeCapabilityReady(backend: ExecutionBackend, api: WorkerConfiguration['api']): boolean {
    const key = discoveryKey(backend, api)
    if (this.discoveryPending()[key]) return false
    const status = this.backendStatuses().find((item) => item.id === backend)
    const apiStatus = status?.runtime_apis?.find((item) => item.id === api)
    return Boolean(status?.available && apiStatus?.available && apiStatus.state === 'verified' && this.validatedDiscoveries.has(key))
  }

  private setDiscoveryPending(key: string, pending: boolean): void {
    this.discoveryPending.update((current) => {
      const next = { ...current }
      if (pending) next[key] = true
      else delete next[key]
      return next
    })
  }

  async load(): Promise<void> {
    await Promise.all([this.loadProjects(), this.loadCatalog(), this.loadRuntimeMetadata(), this.loadMetrics()])
  }

  async refreshProjects(): Promise<ProjectSummary[]> {
    this.projectsRequest = undefined
    this.projectsReady.set(false)
    return this.loadProjects()
  }

  async refreshDatasets(): Promise<DatasetSummary[]> {
    this.datasetsRequest = undefined
    this.datasetsReady.set(false)
    return this.loadDatasets()
  }

  retryDatasets(): Promise<DatasetSummary[]> {
    return this.refreshDatasets()
  }

  retryProjects(): Promise<ProjectSummary[]> {
    return this.refreshProjects()
  }

  retryCatalog(): Promise<NodeManifest[]> {
    this.catalogRequest = undefined
    this.catalogReady.set(false)
    return this.loadCatalog()
  }

  retryRuntimeMetadata(): Promise<void> {
    this.runtimeRequest = undefined
    this.runtimeReady.set(false)
    return this.loadRuntimeMetadata()
  }

  async refreshMetrics(): Promise<MetricDefinition[]> {
    this.metricsRequest = undefined
    this.metricsReady.set(false)
    return this.loadMetrics()
  }

  hasError(resource: string): boolean {
    return Boolean(this.errors()[resource])
  }

  preferredBackendForApi(api: RuntimeApi): ExecutionBackend | null {
    return selectPreferredBackendForApi(this.backendStatuses(), api)
  }

  selectRuntimePreferenceBackend(backend: ExecutionBackend | null): void {
    this.runtimePreferenceBackend.set(backend)
    if (!backend) return
    const status = this.backendStatuses().find((item) => item.id === backend)
    if (!status) return
    const preferences = this.runtimePreferences()
    if (!preferences[backend]) {
      const preference = defaultRuntimePreference(status)
      if (preference) {
        const next = { ...preferences, [backend]: preference }
        this.runtimePreferences.set(next)
        writeRuntimePreferences(next)
      }
    }
  }

  setRuntimePreference(selection: WorkerConfiguration): void {
    const next = { ...this.runtimePreferences(), [selection.backend]: selection }
    this.runtimePreferences.set(next)
    writeRuntimePreferences(next)
    this.runtimePreferenceBackend.set(selection.backend)
  }

  clearError(): void {
    this.errors.set({})
  }

  private recordError(resource: string, message: string): void {
    this.errors.set({ ...this.errors(), [resource]: message })
  }

  private clearResourceError(resource: string): void {
    if (!this.hasError(resource)) return
    const next = { ...this.errors() }
    delete next[resource]
    this.errors.set(next)
  }

  private message(reason: unknown, fallback: string): string {
    return reason instanceof Error ? reason.message : fallback
  }
}

function markResource(resource: string): void {
  if (typeof performance !== 'undefined') performance.mark(`nw:resource:${resource}:complete`)
}

function discoveryKey(backend: ExecutionBackend, api: WorkerConfiguration['api']): string {
  return `${backend}:${api}`
}

function waitForDiscoveryPoll(): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, 500))
}

function observeWithAbort<T>(operation: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(abortError())
  return new Promise<T>((resolve, reject) => {
    const onAbort = (): void => reject(abortError())
    signal.addEventListener('abort', onAbort, { once: true })
    void operation.then((value) => {
      signal.removeEventListener('abort', onAbort)
      resolve(value)
    }, (reason) => {
      signal.removeEventListener('abort', onAbort)
      reject(reason)
    })
  })
}

function abortError(): Error {
  const error = new Error('Runtime discovery observation aborted.')
  error.name = 'AbortError'
  return error
}
