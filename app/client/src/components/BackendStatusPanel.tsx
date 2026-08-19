import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, CheckCircle2, Cpu, PackageX, Square, Terminal } from 'lucide-react'
import { api } from '../api/client'
import type { ExecutionBackendStatus, JobResponse, MemoryPolicyKind, RuntimeApi, RuntimeApiStatus, RuntimeDevice, WorkerConfiguration } from '../domain/types'
import { useAppResources } from '../state/useAppResources'

const labels: Record<ExecutionBackendStatus['id'], string> = {
  tensorflow: 'TensorFlow',
  pytorch: 'PyTorch',
  jax: 'JAX',
}

const apiLabels: Record<RuntimeApi, string> = { keras: 'Keras', native: 'Native framework' }

export function BackendStatusPanel() {
  const { backendStatuses, runtimeApis, loading, runtimeSelection, setRuntimeSelection, setSelectedExecutionBackend } = useAppResources()
  const [activeJobs, setActiveJobs] = useState<JobResponse[]>([])
  const [pendingSelection, setPendingSelection] = useState<WorkerConfiguration | null>(null)
  const [stopping, setStopping] = useState(false)

  useEffect(() => {
    let mounted = true
    const refreshJobs = async () => {
      try {
        const jobs = await api.listJobs(true)
        if (mounted) setActiveJobs(jobs)
      } catch {
        if (mounted) setActiveJobs([])
      }
    }
    void refreshJobs()
    const timer = window.setInterval(refreshJobs, 1000)
    return () => { mounted = false; window.clearInterval(timer) }
  }, [])

  const operationalCount = backendStatuses.filter((item) => item.available).length
  const currentJobUsesSelection = useMemo(() => activeJobs.some((job) => sameWorker(job.worker, runtimeSelection)), [activeJobs, runtimeSelection])

  if (loading || backendStatuses.length === 0) return null

  const requestSelection = (next: WorkerConfiguration) => {
    if (runtimeSelection && currentJobUsesSelection && !sameWorker(runtimeSelection, next)) {
      setPendingSelection(next)
      return
    }
    applySelection(next)
  }

  const applySelection = (next: WorkerConfiguration) => {
    setSelectedExecutionBackend(next.backend)
    setRuntimeSelection(next)
    setPendingSelection(null)
  }

  const stopAndSwitch = async () => {
    if (!pendingSelection) return
    setStopping(true)
    try {
      await Promise.all(activeJobs.filter((job) => sameWorker(job.worker, runtimeSelection)).map((job) => api.stopJob(job.id)))
      let remaining = activeJobs.filter((job) => sameWorker(job.worker, runtimeSelection))
      for (let attempt = 0; attempt < 30 && remaining.length; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 250))
        remaining = (await api.listJobs(true)).filter((job) => sameWorker(job.worker, runtimeSelection))
      }
      applySelection(pendingSelection)
    } finally {
      setStopping(false)
    }
  }

  const runtimeApiStatuses = runtimeApis.length ? runtimeApis : fallbackRuntimeApis
  return (
    <section className="backend-status-panel" aria-label="Compute environment status">
      <div className="backend-status-heading">
        <div>
          <span className="eyebrow">COMPUTE ENVIRONMENT</span>
          <h2>Execution runtime</h2>
          <p>{operationalCount ? `${operationalCount} operational backend${operationalCount === 1 ? '' : 's'}` : 'No operational backend'} · live worker probes · one lease per physical device</p>
        </div>
        <span className="backend-status-command"><Terminal size={13} />isolated worker controller</span>
      </div>
      <div className="runtime-api-controls" aria-label="Runtime API">
        {runtimeApiStatuses.map((status) => <button key={status.id} type="button" className={`runtime-api-option ${runtimeSelection?.api === status.id ? 'selected' : ''}`} disabled={!status.available || status.id !== 'keras'} onClick={() => runtimeSelection && requestSelection({ ...runtimeSelection, api: status.id })}><strong>{apiLabels[status.id]}</strong><span>{status.available ? 'Operational' : status.reason_code === 'not_implemented' ? 'Deferred' : 'Unavailable'}</span></button>)}
      </div>
      <div className="backend-status-grid">
        {backendStatuses.map((status) => <BackendStatusCard key={status.id} status={status} selected={runtimeSelection?.backend === status.id} runtimeSelection={runtimeSelection} onSelect={requestSelection} />)}
      </div>
      {pendingSelection && <div className="selection-prompt panel" role="alert"><strong>This runtime is used by an active worker.</strong><span>Choose whether to stop and reap it before switching, or keep it unchanged and use the new selection only for subsequent jobs.</span><div className="dialog-actions"><button className="secondary" onClick={() => applySelection(pendingSelection)} disabled={stopping}>Keep running</button><button className="danger-action" onClick={stopAndSwitch} disabled={stopping}>{stopping ? 'Reaping…' : 'Stop and switch'}</button></div></div>}
      <ActiveJobs jobs={activeJobs} />
    </section>
  )
}

function BackendStatusCard({ status, selected, runtimeSelection, onSelect }: { status: ExecutionBackendStatus; selected: boolean; runtimeSelection: WorkerConfiguration | null; onSelect: (selection: WorkerConfiguration) => void }) {
  const state = status.available ? 'usable' : status.installed ? 'unavailable' : 'missing'
  const StateIcon = status.available ? CheckCircle2 : status.installed ? AlertTriangle : PackageX
  const devices = status.devices ?? []
  const selectableDevices = devices.length ? devices : [{ device_id: 'cpu', lease_key: 'cpu', kind: 'cpu', name: 'CPU', accelerator: 'cpu', available: true, supported_memory_policies: ['default'] as MemoryPolicyKind[] }]
  const selectedDevice = selectableDevices.find((device) => device.device_id === runtimeSelection?.device_id) ?? selectableDevices[0]
  const detail = status.available
    ? `${devices.length || 1} device${devices.length === 1 ? '' : 's'}${status.device_name ? ` · ${status.device_name}` : ''}`
    : status.installed ? formatReason(status.reason_code) : 'Not installed'
  const selectedPolicy = runtimeSelection?.backend === status.id ? runtimeSelection.memory_policy : { kind: 'default' as MemoryPolicyKind }
  const selectDevice = (device: RuntimeDevice) => onSelect({ api: 'keras', backend: status.id, device_id: device.device_id, memory_policy: policyFromKind(device.supported_memory_policies[0] ?? 'default'), precision: runtimeSelection?.precision ?? 'float32' })
  const selectPolicy = (kind: MemoryPolicyKind) => onSelect({ api: 'keras', backend: status.id, device_id: selectedDevice?.device_id ?? 'cpu', memory_policy: policyFromKind(kind), precision: runtimeSelection?.precision ?? 'float32' })
  const updatePolicyValue = (value: number) => onSelect({ api: 'keras', backend: status.id, device_id: selectedDevice?.device_id ?? 'cpu', memory_policy: selectedPolicy.kind === 'tensorflow_limit_mb' ? { kind: selectedPolicy.kind, limit_mb: value } : { kind: selectedPolicy.kind, fraction: value }, precision: runtimeSelection?.precision ?? 'float32' })

  return <div className={`backend-status-card-container ${selected ? 'selected' : ''}`}><button type="button" className={`backend-status-card ${state} ${selected ? 'selected' : ''}`} disabled={!status.available} aria-pressed={selected} onClick={() => selectedDevice && selectDevice(selectedDevice)}><div className="backend-status-card-top"><span className="backend-status-name"><Cpu size={14} />{labels[status.id]}</span><span className="backend-status-state"><StateIcon size={13} />{status.available ? 'Usable' : status.installed ? 'Installed' : 'Missing'}</span></div><strong>{status.version ? `v${status.version}` : 'Optional framework'}</strong><span>{detail}</span></button>{status.available && <div className="runtime-fields"><label>Device<select value={selectedDevice?.device_id ?? ''} onChange={(event) => { const device = selectableDevices.find((item) => item.device_id === event.target.value); if (device) selectDevice(device) }}>{selectableDevices.map((device) => <option key={device.device_id} value={device.device_id}>{device.device_id} · {device.name}</option>)}</select></label><label>Memory policy<select value={selectedPolicy.kind} onChange={(event) => selectPolicy(event.target.value as MemoryPolicyKind)}>{(selectedDevice?.supported_memory_policies ?? ['default']).map((policy) => <option key={policy} value={policy}>{policy}</option>)}</select></label>{selectedPolicy.kind === 'tensorflow_limit_mb' && <label>Limit (MB)<input type="number" min="1" value={selectedPolicy.limit_mb ?? 1024} onChange={(event) => updatePolicyValue(Number(event.target.value))} /></label>}{(selectedPolicy.kind === 'jax_preallocate_fraction' || selectedPolicy.kind === 'pytorch_fraction') && <label>Fraction<input type="number" min="0.01" max="1" step="0.01" value={selectedPolicy.fraction ?? 0.5} onChange={(event) => updatePolicyValue(Number(event.target.value))} /></label>}</div>}</div>
}

function ActiveJobs({ jobs }: { jobs: JobResponse[] }) {
  if (!jobs.length) return <div className="active-jobs panel"><div><span className="eyebrow">ACTIVE WORKERS</span><strong>No active workers</strong></div><span className="muted">Jobs continue across navigation and are reaped before their device lease is released.</span></div>
  return <div className="active-jobs panel"><div className="section-heading"><div><span className="eyebrow">ACTIVE WORKERS</span><h3>{jobs.length} active job{jobs.length === 1 ? '' : 's'}</h3></div><Terminal size={16} /></div>{jobs.map((job) => <div className="active-job-row" key={job.id}><div><strong>{job.operation}</strong><span>{job.worker.backend} · {job.worker.device_id} · PID {job.pid ?? 'starting'}</span></div><div><span>{job.state}</span><small>{job.progress.phase} · {job.diagnostics.process_rss_bytes ? `${Math.round(job.diagnostics.process_rss_bytes / 1024 / 1024)} MB RSS` : 'memory pending'}</small></div><button type="button" className="icon-button" title="Stop worker" onClick={() => void api.stopJob(job.id)}><Square size={14} /></button></div>)}</div>
}

function sameWorker(left: WorkerConfiguration | null, right: WorkerConfiguration | null): boolean {
  return Boolean(left && right && left.api === right.api && left.backend === right.backend && left.device_id === right.device_id && left.memory_policy.kind === right.memory_policy.kind && left.precision === right.precision)
}

function policyFromKind(kind: MemoryPolicyKind): WorkerConfiguration['memory_policy'] {
  if (kind === 'tensorflow_limit_mb') return { kind, limit_mb: 1024 }
  if (kind === 'jax_preallocate_fraction' || kind === 'pytorch_fraction') return { kind, fraction: 0.5 }
  return { kind }
}

const fallbackRuntimeApis: RuntimeApiStatus[] = [{ id: 'keras', available: true }, { id: 'native', available: false, reason_code: 'not_implemented' }]

function formatReason(reason: string | null | undefined): string {
  if (reason === 'dependency_missing') return 'Installed package could not be imported'
  if (reason === 'worker_failed') return 'Worker probe failed'
  if (reason === 'backend_import_failed') return 'Backend import failed'
  return reason ? reason.replaceAll('_', ' ') : 'Capability probe unavailable'
}
