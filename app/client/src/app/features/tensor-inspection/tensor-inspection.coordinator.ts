import { Injectable, inject, signal } from '@angular/core'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import type {
  JobResponse,
  ProjectDocument,
  RuntimeDevice,
  TensorInspectionNode,
  TensorInspectionSession,
  TensorInspectionViewResponse,
  WorkerConfiguration,
} from '../../domain/types'
import { pollJobUntilTerminal, terminalJobStates } from '../models/job-polling'

export interface TensorInspectionOpenRequest {
  projectId: string
  revision: number
  nodeId: string
}

@Injectable({ providedIn: 'root' })
export class TensorInspectionCoordinator {
  private readonly api = inject(ApiService)
  private readonly resources = inject(AppResourcesService)
  readonly open = signal(false)
  readonly busy = signal(false)
  readonly error = signal<string | null>(null)
  readonly job = signal<JobResponse | null>(null)
  readonly session = signal<TensorInspectionSession | null>(null)
  readonly node = signal<TensorInspectionNode | null>(null)
  readonly selectedNodeId = signal<string | null>(null)
  readonly view = signal<'values' | 'statistics' | 'histogram' | 'plot' | 'heatmap' | 'scatter'>('values')
  readonly values = signal<TensorInspectionViewResponse | null>(null)
  readonly valuesOffset = signal(0)
  readonly fixedIndices = signal<Record<number, number>>({})
  readonly displayAxes = signal<number[]>([])
  readonly selectedTensorId = signal<string | null>(null)
  private operationToken = 0
  private readonly tensorViewChoices = new Map<string, { displayAxes: number[]; fixedIndices: Record<number, number> }>()

  async start(request: TensorInspectionOpenRequest): Promise<void> {
    if (this.open()) await this.close()
    const token = ++this.operationToken
    this.open.set(true)
    this.busy.set(true)
    this.error.set(null)
    this.session.set(null)
    this.node.set(null)
    this.values.set(null)
    this.valuesOffset.set(0)
    this.displayAxes.set([])
    this.tensorViewChoices.clear()
    try {
      const project = await this.api.getProject(request.projectId)
      if (project.current_revision !== request.revision || project.validation_status !== 'valid') {
        throw new Error('Tensor inspection requires the current saved revision to be successfully validated.')
      }
      const worker = await this.workerForProject(project)
      if (!this.isCurrentOperation(token)) return
      if (!worker) throw new Error('The selected runtime is unavailable. Check Environment and retry.')
      let current = await this.api.submitTensorInspection(project.id, project.current_revision, this.seedFor(project), worker)
      if (!this.isCurrentOperation(token)) {
        await this.api.stopJob(current.id).catch(() => undefined)
        return
      }
      this.job.set(current)
      if (!terminalJobStates.has(current.state)) {
        current = await pollJobUntilTerminal(current, (jobId) => this.api.getJob(jobId), (next) => {
          if (this.isCurrentOperation(token)) this.job.set(next)
        }, { intervalMs: 250, timeoutMs: 120_000, isDestroyed: () => !this.isCurrentOperation(token) })
      }
      if (!this.isCurrentOperation(token)) return
      if (current.state !== 'succeeded') throw new Error(current.error?.message || 'The inspection worker did not complete.')
      const sessionId = String(current.result?.['session_id'] || '')
      if (!sessionId) throw new Error('The inspection worker completed without a session.')
      const session = await this.api.tensorInspection(sessionId)
      if (!this.isCurrentOperation(token)) {
        await this.api.deleteTensorInspection(session.id).catch(() => undefined)
        return
      }
      this.session.set(session)
      const selected = session.nodes.find((item) => item.node_id === request.nodeId) ?? session.nodes[0]
      if (!selected) throw new Error('The inspection session contains no executable nodes.')
      await this.selectNode(selected.node_id)
    } catch (reason) {
      if (this.isCurrentOperation(token)) this.error.set(reason instanceof Error ? reason.message : String(reason))
    } finally {
      if (this.isCurrentOperation(token)) this.busy.set(false)
    }
  }

  async selectNode(nodeId: string): Promise<void> {
    const session = this.session()
    if (!session) return
    const target = session.nodes.find((item) => item.node_id === nodeId)
    if (!target || !target.inspectable) {
      this.error.set(target?.unavailable_reason || 'This node cannot be inspected in the selected runtime.')
      return
    }
    this.error.set(null)
    this.selectedNodeId.set(nodeId)
    this.node.set(await this.api.tensorInspectionNode(session.id, nodeId))
    const first = this.node()?.outputs[0] ?? this.node()?.inputs[0]
    this.selectedTensorId.set(first?.id ?? null)
    this.values.set(null)
    this.valuesOffset.set(0)
    this.restoreTensorViewChoice(first ?? null)
    const currentView = this.view()
    this.view.set(first && this.viewCompatible(currentView, first.shape, first.storage_mode)
      ? currentView
      : this.compatibleDefault(first?.shape ?? [], first?.storage_mode ?? 'summary_only'))
    if (first && this.requiresValues(this.view())) await this.loadValues()
  }

  async loadValues(): Promise<void> {
    const session = this.session()
    const tensorId = this.selectedTensorId()
    if (!session || !tensorId) return
    try {
      const tensor = this.selectedTensor()
      const limit = this.view() === 'heatmap' ? 4096 : this.view() === 'plot' || this.view() === 'scatter' ? 2048 : 256
      this.values.set(await this.api.tensorInspectionView(session.id, {
        tensor_id: tensorId,
        // Sampled payloads are flattened deterministic summaries, so their
        // original tensor coordinates are not available for exact slicing.
        fixed_indices: tensor?.storage_mode === 'full' ? this.fixedIndices() : {},
        display_axes: tensor?.storage_mode === 'full' ? this.displayAxes() : [],
        offset: this.valuesOffset(),
        limit,
      }))
    } catch (reason) {
      this.error.set(reason instanceof Error ? reason.message : String(reason))
    }
  }

  selectTensor(tensorId: string): void {
    this.rememberTensorViewChoice()
    this.selectedTensorId.set(tensorId)
    this.values.set(null)
    this.valuesOffset.set(0)
    const tensor = [...(this.node()?.outputs ?? []), ...(this.node()?.inputs ?? [])].find((item) => item.id === tensorId)
    this.restoreTensorViewChoice(tensor ?? null)
    if (tensor && !this.viewCompatible(this.view(), tensor.shape, tensor.storage_mode)) this.view.set(this.compatibleDefault(tensor.shape, tensor.storage_mode))
    if (this.view() === 'values' || this.view() === 'plot' || this.view() === 'heatmap' || this.view() === 'scatter') void this.loadValues()
  }

  setView(view: 'values' | 'statistics' | 'histogram' | 'plot' | 'heatmap' | 'scatter'): void {
    this.view.set(view)
    if (view !== 'values') this.valuesOffset.set(0)
    if (view === 'values' || view === 'plot' || view === 'heatmap' || view === 'scatter') void this.loadValues()
  }

  setFixedIndex(axis: number, value: number): void {
    this.fixedIndices.update((current) => ({ ...current, [axis]: value }))
    this.valuesOffset.set(0)
    this.rememberTensorViewChoice()
    void this.loadValues()
  }

  setDisplayAxis(slot: number, axis: number): void {
    const tensor = this.selectedTensor()
    if (!tensor || tensor.shape.length < 2 || tensor.storage_mode !== 'full') return
    const next = [...this.displayAxes()]
    next[slot] = axis
    const unique = next.filter((value, index, values) => values.indexOf(value) === index)
    if (unique.length < 2 && tensor.shape.length >= 2) {
      const fallback = [...Array(tensor.shape.length).keys()].reverse().find((value) => value !== axis)
      if (fallback !== undefined) unique[slot === 0 ? 1 : 0] = fallback
    }
    this.displayAxes.set(unique.slice(0, 2).sort((left, right) => left - right))
    this.valuesOffset.set(0)
    this.rememberTensorViewChoice()
    void this.loadValues()
  }

  async close(): Promise<void> {
    this.operationToken += 1
    const currentJob = this.job()
    const currentSession = this.session()
    if (currentJob && !terminalJobStates.has(currentJob.state)) await this.api.stopJob(currentJob.id).catch(() => undefined)
    if (currentSession) await this.api.deleteTensorInspection(currentSession.id).catch(() => undefined)
    this.open.set(false)
    this.busy.set(false)
    this.job.set(null)
    this.session.set(null)
    this.node.set(null)
    this.values.set(null)
    this.displayAxes.set([])
    this.valuesOffset.set(0)
    this.tensorViewChoices.clear()
    this.error.set(null)
  }

  private isCurrentOperation(token: number): boolean { return this.operationToken === token && this.open() }

  previousNode(): void { this.moveNode(-1) }
  nextNode(): void { this.moveNode(1) }

  previousValuesPage(): void {
    this.valuesOffset.set(Math.max(0, this.valuesOffset() - 256))
    void this.loadValues()
  }

  nextValuesPage(): void {
    const total = this.values()?.total_values ?? 0
    if (this.valuesOffset() + 256 >= total) return
    this.valuesOffset.update((offset) => offset + 256)
    void this.loadValues()
  }

  valuesHasPreviousPage(): boolean { return this.valuesOffset() > 0 }
  valuesHasNextPage(): boolean { return this.valuesOffset() + 256 < (this.values()?.total_values ?? 0) }

  private moveNode(delta: number): void {
    const nodes = (this.session()?.nodes ?? []).filter((item) => item.inspectable)
    const index = nodes.findIndex((item) => item.node_id === this.selectedNodeId())
    if (index < 0) return
    const target = nodes[index + delta]
    if (target) void this.selectNode(target.node_id)
  }

  private async workerForProject(project: ProjectDocument): Promise<WorkerConfiguration | null> {
    const api = project.keras_enabled ? 'keras' : 'native'
    await this.resources.loadRuntimeMetadata()
    await this.resources.discoverRuntime(project.execution_backend, api, false)
    const status = this.resources.backendStatuses().find((item) => item.id === project.execution_backend && item.available)
    const preference = this.resources.runtimePreferences()[project.execution_backend]
    const devices = status?.devices?.length ? status.devices : status ? [fallbackDevice()] : []
    const device = devices.find((item) => item.device_id === preference?.device_id && item.available) ?? devices.find((item) => item.available)
    if (!status || !preference || !device || !this.resources.runtimeCapabilityReady(project.execution_backend, api)) return null
    const memoryPolicy = device.supported_memory_policies.includes(preference.memory_policy.kind) ? preference.memory_policy : { kind: 'default' as const }
    return { ...preference, api, device_id: device.device_id, memory_policy: memoryPolicy }
  }

  private seedFor(project: ProjectDocument): number {
    return project.keras_enabled && project.graph.model_configuration.keras.random_seed !== null
      ? Number(project.graph.model_configuration.keras.random_seed)
      : 42
  }

  private compatibleDefault(shape: number[], storageMode: 'full' | 'sampled' | 'summary_only' = 'full'): 'values' | 'statistics' | 'histogram' | 'plot' | 'heatmap' | 'scatter' {
    if (shape.length === 1) return 'plot'
    if (shape.length >= 2 && storageMode === 'full') return 'heatmap'
    return 'values'
  }

  private selectedTensor(): TensorInspectionNode['outputs'][number] | TensorInspectionNode['inputs'][number] | null {
    const tensorId = this.selectedTensorId()
    if (!tensorId) return null
    return [...(this.node()?.outputs ?? []), ...(this.node()?.inputs ?? [])].find((item) => item.id === tensorId) ?? null
  }

  private defaultDisplayAxes(shape: number[]): number[] {
    if (shape.length === 0) return []
    if (shape.length === 1) return [0]
    return [shape.length - 2, shape.length - 1]
  }

  private restoreTensorViewChoice(tensor: TensorInspectionNode['outputs'][number] | TensorInspectionNode['inputs'][number] | null): void {
    if (!tensor) {
      this.displayAxes.set([])
      this.fixedIndices.set({})
      return
    }
    const saved = this.tensorViewChoices.get(tensor.id)
    const axes = saved?.displayAxes?.length ? saved.displayAxes : this.defaultDisplayAxes(tensor.shape)
    this.displayAxes.set(axes.filter((axis) => axis >= 0 && axis < tensor.shape.length).slice(0, 2))
    this.fixedIndices.set({ ...(saved?.fixedIndices ?? {}) })
  }

  private rememberTensorViewChoice(): void {
    const tensorId = this.selectedTensorId()
    if (!tensorId) return
    this.tensorViewChoices.set(tensorId, { displayAxes: [...this.displayAxes()], fixedIndices: { ...this.fixedIndices() } })
  }

  private viewCompatible(view: 'values' | 'statistics' | 'histogram' | 'plot' | 'heatmap' | 'scatter', shape: number[], storageMode: 'full' | 'sampled' | 'summary_only'): boolean {
    if (view === 'plot') return storageMode !== 'summary_only' && shape.length === 1
    if (view === 'heatmap') return storageMode === 'full' && shape.length >= 2
    if (view === 'scatter') return storageMode === 'full' && shape.length === 2 && shape[1] === 2
    return true
  }

  private requiresValues(view: 'values' | 'statistics' | 'histogram' | 'plot' | 'heatmap' | 'scatter'): boolean {
    return view === 'values' || view === 'plot' || view === 'heatmap' || view === 'scatter'
  }
}

function fallbackDevice(): RuntimeDevice { return { device_id: 'cpu', lease_key: 'cpu', kind: 'cpu', name: 'CPU', accelerator: 'cpu', available: true, supported_memory_policies: ['default'] } }
