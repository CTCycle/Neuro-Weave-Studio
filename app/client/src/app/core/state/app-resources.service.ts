import { Injectable, computed, inject, signal } from '@angular/core'
import { ApiService } from '../api/api.service'
import type {
  CompilationCapabilities,
  ExecutionBackend,
  ExecutionBackendStatus,
  NodeManifest,
  ProjectSummary,
  RuntimeApiStatus,
  RuntimeDevice,
  WorkerConfiguration,
} from '../../domain/types'

@Injectable({ providedIn: 'root' })
export class AppResourcesService {
  private readonly api = inject(ApiService)

  readonly manifests = signal<NodeManifest[]>([])
  readonly projects = signal<ProjectSummary[]>([])
  readonly backendStatuses = signal<ExecutionBackendStatus[]>([])
  readonly runtimeApis = signal<RuntimeApiStatus[]>([])
  readonly compilationCapabilities = signal<CompilationCapabilities | null>(null)
  readonly runtimePreferences = signal<Partial<Record<ExecutionBackend, WorkerConfiguration>>>({})
  readonly runtimePreferenceBackend = signal<ExecutionBackend | null>(null)
  readonly catalogReady = signal(false)
  readonly projectsReady = signal(false)
  readonly runtimeReady = signal(false)
  readonly errors = signal<Record<string, string>>({})
  readonly error = computed(() => Object.values(this.errors()).join(' ') || null)

  constructor() {
    void this.load()
  }

  async load(): Promise<void> {
    void this.api.catalog().then((value) => this.manifests.set(value)).catch(() => this.recordError('catalog', 'Node catalog unavailable.')).finally(() => this.catalogReady.set(true))
    void this.api.listProjects().then((value) => this.projects.set(value)).catch(() => this.recordError('projects', 'Model catalog unavailable.')).finally(() => this.projectsReady.set(true))
    void this.api.metadata().then((value) => {
      const statuses = value.execution_backends
      this.backendStatuses.set(statuses)
      this.compilationCapabilities.set(value.configuration.compilation_capabilities)
      this.runtimeApis.set(value.runtime_apis ?? [{ id: 'keras', available: true }, { id: 'native', available: false, reason_code: 'runtime_unavailable' }])
      const current = this.runtimePreferences()
      const next = { ...current }
      for (const status of statuses.filter((item) => item.available)) {
        if (!next[status.id]) next[status.id] = defaultPreference(status)
      }
      this.runtimePreferences.set(next)
      const selected = this.runtimePreferenceBackend()
      this.runtimePreferenceBackend.set(selected && statuses.some((item) => item.id === selected && item.available) ? selected : preferredBackend(statuses))
    }).catch(() => this.recordError('runtime', 'Runtime metadata unavailable.')).finally(() => this.runtimeReady.set(true))
  }

  async refreshProjects(): Promise<ProjectSummary[]> {
    const projects = await this.api.listProjects()
    this.projects.set(projects)
    return projects
  }

  selectRuntimePreferenceBackend(backend: ExecutionBackend | null): void {
    this.runtimePreferenceBackend.set(backend)
    if (!backend) return
    const status = this.backendStatuses().find((item) => item.id === backend)
    if (!status) return
    const preferences = this.runtimePreferences()
    if (!preferences[backend]) this.runtimePreferences.set({ ...preferences, [backend]: defaultPreference(status) })
  }

  setRuntimePreference(selection: WorkerConfiguration): void {
    this.runtimePreferences.set({ ...this.runtimePreferences(), [selection.backend]: selection })
    this.runtimePreferenceBackend.set(selection.backend)
  }

  clearError(): void {
    this.errors.set({})
  }

  private recordError(resource: string, message: string): void {
    this.errors.set({ ...this.errors(), [resource]: message })
  }
}

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
