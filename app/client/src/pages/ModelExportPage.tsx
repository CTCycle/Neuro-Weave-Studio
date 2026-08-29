import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle2, Download, LoaderCircle, RefreshCw, Square } from 'lucide-react'
import { useOutletContext } from 'react-router-dom'
import { ApiError, api } from '../api/client'
import type {
  ExportCapability,
  ExportCategory,
  ExportFormat,
  ExportOptions,
  JobResponse,
  JobState,
  ProjectDocument,
  RuntimeDevice,
  WorkerConfiguration,
  ExecutionBackendStatus,
} from '../domain/types'
import { useAppResources } from '../state/useAppResources'
import type { ModelWorkspaceContext } from './ModelWorkspacePage'

interface ModelExportPageProps {
  project: ProjectDocument
}

const terminalStates: ReadonlySet<JobState> = new Set(['succeeded', 'terminated', 'failed'])
const categoryOrder: ExportCategory[] = ['project', 'source', 'native', 'interchange', 'checkpoint']
const categoryLabels: Record<ExportCategory, string> = {
  project: 'Project',
  source: 'Source',
  native: 'Native',
  interchange: 'Interchange',
  checkpoint: 'Weights / Checkpoint',
}

const initialOptions: ExportOptions = {
  opset_version: 18,
  dynamic_batch: true,
  external_data: 'auto',
  max_shard_size_gb: null,
}

export function ModelExportPage() {
  const { project } = useOutletContext<ModelWorkspaceContext>()
  return <ModelExportContent project={project} />
}

export function ModelExportContent({ project }: ModelExportPageProps) {
  const { backendStatuses, runtimePreferences } = useAppResources()
  const [capabilities, setCapabilities] = useState<ExportCapability[]>([])
  const [capabilityRevision, setCapabilityRevision] = useState<number | null>(null)
  const [capabilitiesLoading, setCapabilitiesLoading] = useState(true)
  const [capabilitiesError, setCapabilitiesError] = useState<string | null>(null)
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat | null>(null)
  const [filenameStem, setFilenameStem] = useState(project.name)
  const [options, setOptions] = useState<ExportOptions>(initialOptions)
  const [job, setJob] = useState<JobResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const mounted = useRef(true)

  const loadCapabilities = useCallback(async () => {
    setCapabilitiesLoading(true)
    setCapabilitiesError(null)
    try {
      const response = await api.exportCapabilities(project.id)
      if (!mounted.current) return
      setCapabilities(response.formats)
      setCapabilityRevision(response.project_revision)
      setSelectedFormat((current) => {
        if (current && response.formats.some((item) => item.format === current && item.available)) return current
        return response.formats.find((item) => item.available)?.format ?? null
      })
    } catch (reason) {
      if (mounted.current) setCapabilitiesError(messageFor(reason))
    } finally {
      if (mounted.current) setCapabilitiesLoading(false)
    }
  }, [project.id])

  useEffect(() => {
    mounted.current = true
    setFilenameStem(project.name)
    void loadCapabilities()
    return () => { mounted.current = false }
  }, [loadCapabilities, project.name])

  const selectedCapability = useMemo(
    () => capabilities.find((item) => item.format === selectedFormat) ?? null,
    [capabilities, selectedFormat],
  )

  useEffect(() => {
    if (selectedCapability) setOptions(optionsFor(selectedCapability))
  }, [selectedCapability])

  const worker = useMemo(
    () => workerForProject(project, backendStatuses, runtimePreferences),
    [backendStatuses, project, runtimePreferences],
  )
  const isDirect = selectedFormat === 'project_bundle' || selectedFormat === 'source'
  const exportActive = Boolean(job && !terminalStates.has(job.state))
  const canExport = Boolean(selectedCapability?.available && !busy && (isDirect || worker))
  const displayedFilename = `${safeFilename(filenameStem || project.name)}${extensionFor(selectedCapability, options)}`

  const runExport = async () => {
    if (!selectedCapability?.available || !selectedFormat || busy) return
    setBusy(true)
    setNotice(null)
    setJob(null)
    try {
      if (selectedFormat === 'project_bundle') {
        const bundle = await api.exportProject(project.id)
        downloadBlob(new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' }), displayedFilename)
        setNotice(`Downloaded ${displayedFilename}.`)
        return
      }
      if (selectedFormat === 'source') {
        const source = await api.sourceForProject(project.id)
        downloadBlob(new Blob([source.code], { type: 'text/x-python' }), displayedFilename)
        setNotice(`Downloaded ${displayedFilename}.`)
        return
      }
      if (!worker) {
        setNotice('No operational worker is available for this project profile. Configure the backend in Environment and retry.')
        return
      }
      let current = await api.submitModelExport(project.id, selectedFormat, filenameStem.trim() || project.name, options, worker)
      if (mounted.current) setJob(current)
      while (mounted.current && !terminalStates.has(current.state)) {
        await wait(400)
        current = await api.getJob(current.id)
        if (mounted.current) setJob(current)
      }
      if (!mounted.current) return
      if (current.state !== 'succeeded') {
        setNotice(`${selectedCapability.label} ended with ${current.state}: ${current.error?.message ?? 'the worker did not publish an artifact'}`)
        return
      }
      const artifactFilename = current.artifact?.filename ?? displayedFilename
      const blob = await api.jobArtifact(current.id)
      downloadBlob(blob, artifactFilename)
      setNotice(`Downloaded verified ${artifactFilename}.`)
    } catch (reason) {
      if (mounted.current) setNotice(`${selectedCapability.label} failed: ${messageFor(reason)}`)
    } finally {
      if (mounted.current) setBusy(false)
    }
  }

  const stopExport = async () => {
    if (!job || !exportActive || cancelling) return
    setCancelling(true)
    try {
      const stopped = await api.stopJob(job.id)
      if (mounted.current) {
        setJob(stopped)
        setNotice('Cancellation requested. The worker will finish cleanup before the lease is released.')
      }
    } catch (reason) {
      if (mounted.current) setNotice(`Cancellation failed: ${messageFor(reason)}`)
    } finally {
      if (mounted.current) setCancelling(false)
    }
  }

  const groupedCapabilities = categoryOrder.map((category) => ({
    category,
    formats: capabilities.filter((item) => item.category === category),
  })).filter((group) => group.formats.length)

  return <div className="workflow-page export-page">
    <section className="model-export-page panel" aria-labelledby="model-export-title">
      <header className="model-export-header">
        <div>
          <span className="eyebrow">MODEL EXPORT CENTER</span>
          <h2 id="model-export-title">Export {project.name}</h2>
          <p>Revision {capabilityRevision ?? project.current_revision} · {project.execution_backend} · {project.keras_enabled ? 'Keras profile' : 'native profile'}</p>
        </div>
      </header>

      {capabilitiesLoading && <div className="model-export-state"><LoaderCircle className="spin" size={20} /><span>Probing export capabilities…</span></div>}
      {capabilitiesError && <div className="model-export-state export-error" role="alert"><AlertTriangle size={18} /><div><strong>Capabilities unavailable</strong><span>{capabilitiesError}</span><button className="secondary" type="button" onClick={() => void loadCapabilities()}><RefreshCw size={14} />Retry</button></div></div>}

      {!capabilitiesLoading && !capabilitiesError && <>
        <div className="model-export-body">
          <div className="model-export-catalog">
            {groupedCapabilities.map((group) => <section className="export-category" key={group.category} aria-labelledby={`export-category-${group.category}`}>
              <div className="export-category-heading"><div><span className="eyebrow">EXPORT METHODS</span><h3 id={`export-category-${group.category}`}>{categoryLabels[group.category]}</h3></div><span className="muted">{group.formats.filter((item) => item.available).length}/{group.formats.length} available</span></div>
              <div className="export-format-list">
                {group.formats.map((capability) => <button key={capability.format} type="button" className={`export-format-option ${selectedFormat === capability.format ? 'selected' : ''} ${!capability.available ? 'unavailable' : ''}`} disabled={!capability.available || busy} onClick={() => { setSelectedFormat(capability.format); setNotice(null); setJob(null) }} title={capability.available ? capability.compatibility : capability.reason ?? 'Unavailable'}>
                  <span className="export-format-top"><strong>{capability.label}</strong>{capability.available ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}</span>
                  <span className="export-badge-row"><ExportBadges capability={capability} /></span>
                  <span className="export-format-description">{capability.available ? capability.compatibility : capability.reason ?? 'Unavailable for this profile.'}</span>
                </button>)}
              </div>
            </section>)}
            {!capabilities.length && <div className="model-export-state"><AlertTriangle size={18} /><span>No export formats were returned for this project.</span></div>}
          </div>

          <aside className="model-export-config panel">
            <div className="section-heading"><div><span className="eyebrow">EXPORT SETTINGS</span><h3>{selectedCapability?.label ?? 'Select a format'}</h3></div><Download size={17} /></div>
            {selectedCapability ? <>
              <label className="export-field">Filename stem<input value={filenameStem} maxLength={160} onChange={(event) => setFilenameStem(event.target.value)} placeholder={project.name} aria-describedby="export-filename-help" /></label>
              <span id="export-filename-help" className="export-help">The server rejects paths and supplied extensions, then derives the final extension.</span>
              <div className="export-preview-name"><span>Download name</span><strong>{displayedFilename}</strong></div>
              {selectedCapability.format === 'onnx' && <OnnxOptions capability={selectedCapability} options={options} onChange={setOptions} />}
              {selectedCapability.options.some((item) => item.id === 'max_shard_size_gb') && <label className="export-field">Maximum shard size (GB)<input type="number" min="0.01" max="100" step="0.01" value={options.max_shard_size_gb ?? ''} onChange={(event) => setOptions((current) => ({ ...current, max_shard_size_gb: event.target.value ? Number(event.target.value) : null }))} /></label>}
              <div className="export-detail-list"><div><span>Payload</span><strong>{payloadLabel(selectedCapability.payload)}</strong></div><div><span>Conversion</span><strong>{conversionLabel(selectedCapability.conversion)}</strong></div><div><span>Verification</span><strong>Structural · reload · inference</strong></div></div>
              {!isDirect && <div className="export-worker-summary"><span>Worker</span><strong>{worker ? `${worker.api} · ${worker.backend} · ${worker.device_id}` : 'No operational worker'}</strong></div>}
            </> : <p className="muted">Select an available capability to configure an export.</p>}
          </aside>
        </div>

        {job && <div className={`model-export-job ${job.state === 'failed' ? 'failed' : job.state === 'succeeded' ? 'succeeded' : ''}`} role="status"><div><span className="eyebrow">EXPORT JOB · {job.state}</span><strong>{job.progress.message}</strong><small>{job.progress.phase}{job.progress.percent === null || job.progress.percent === undefined ? '' : ` · ${job.progress.percent}%`} {job.artifact ? `· ${job.artifact.filename}` : ''}</small></div>{exportActive && <button className="secondary" type="button" onClick={() => void stopExport()} disabled={cancelling}><Square size={14} />{cancelling ? 'Cancelling…' : 'Cancel export'}</button>}</div>}
        {notice && <div className="model-export-notice" role="alert">{notice}</div>}
      </>}

      <footer className="model-export-footer dialog-actions">
        <span className="model-export-footer-note">{selectedCapability?.limitations[0] ?? 'Unsupported formats remain visible with their reason.'}</span>
        <div><button className="primary" type="button" onClick={() => void runExport()} disabled={!canExport}>{busy ? <LoaderCircle className="spin" size={15} /> : <Download size={15} />}{busy ? 'Exporting…' : job?.state === 'failed' ? 'Retry export' : 'Export artifact'}</button></div>
      </footer>
    </section>
  </div>
}

function ExportBadges({ capability }: { capability: ExportCapability }) {
  const badges = [
    capability.conversion === 'native' ? 'Native' : capability.conversion === 'bridge' ? 'Converted' : null,
    capability.payload === 'full_model' ? 'Full model' : capability.payload === 'weights_only' ? 'Weights-only' : null,
    capability.inference_only ? 'Inference-only' : null,
    capability.training_state ? 'Training state unavailable' : null,
  ].filter((item): item is string => Boolean(item))
  return <>{badges.map((badge) => <span className="export-badge" key={badge}>{badge}</span>)}</>
}

function OnnxOptions({ capability, options, onChange }: { capability: ExportCapability; options: ExportOptions; onChange: (next: ExportOptions | ((current: ExportOptions) => ExportOptions)) => void }) {
  const has = (id: string) => capability.options.some((item) => item.id === id)
  const opset = capability.options.find((item) => item.id === 'opset_version')
  const external = capability.options.find((item) => item.id === 'external_data')
  return <div className="export-option-stack">
    {has('opset_version') && <label className="export-field">{opset?.label ?? 'ONNX opset'}<select value={options.opset_version} onChange={(event) => onChange((current) => ({ ...current, opset_version: Number(event.target.value) }))}>{(opset?.choices.length ? opset.choices : [17, 18, 19, 20]).map((choice) => <option key={String(choice)} value={String(choice)}>{String(choice)}</option>)}</select></label>}
    {has('dynamic_batch') && <label className="export-checkbox"><input type="checkbox" checked={options.dynamic_batch} onChange={(event) => onChange((current) => ({ ...current, dynamic_batch: event.target.checked }))} /><span>Dynamic batch dimensions</span></label>}
    {has('external_data') && <label className="export-field">{external?.label ?? 'External data'}<select value={options.external_data} onChange={(event) => onChange((current) => ({ ...current, external_data: event.target.value as ExportOptions['external_data'] }))}>{(external?.choices.length ? external.choices : ['auto', 'always', 'never']).map((choice) => <option key={String(choice)} value={String(choice)}>{String(choice)}</option>)}</select></label>}
    <span className="export-help">Named inputs and outputs are preserved. External data is packaged as a deterministic ZIP when forced.</span>
  </div>
}

function optionsFor(capability: ExportCapability): ExportOptions {
  const next = { ...initialOptions }
  const value = (id: string) => capability.options.find((item) => item.id === id)?.default
  if (typeof value('opset_version') === 'number') next.opset_version = value('opset_version') as number
  if (typeof value('dynamic_batch') === 'boolean') next.dynamic_batch = value('dynamic_batch') as boolean
  const externalData = value('external_data')
  if (externalData === 'auto' || externalData === 'always' || externalData === 'never') next.external_data = externalData
  return next
}

function workerForProject(project: ProjectDocument, statuses: ExecutionBackendStatus[], preferences: Partial<Record<ProjectDocument['execution_backend'], WorkerConfiguration>>): WorkerConfiguration | null {
  const status = statuses.find((item) => item.id === project.execution_backend)
  if (!status?.available) return null
  const api: WorkerConfiguration['api'] = project.keras_enabled ? 'keras' : 'native'
  const apiStatus = status.runtime_apis?.find((item) => item.id === api)
  if (apiStatus && !apiStatus.available) return null
  const preference = preferences[project.execution_backend]
  const devices = status.devices?.length ? status.devices : [fallbackDevice()]
  const preferredDevice = preference ? devices.find((item) => item.device_id === preference.device_id && item.available) : null
  const device = preferredDevice ?? devices.find((item) => item.available) ?? null
  if (!device) return null
  const preferredPolicy = preference?.memory_policy
  const memoryPolicy = preferredPolicy && device.supported_memory_policies.includes(preferredPolicy.kind) ? preferredPolicy : policyForDevice(device)
  return {
    api,
    backend: project.execution_backend,
    device_id: device.device_id,
    memory_policy: memoryPolicy,
    precision: preference?.precision ?? 'float32',
  }
}

function fallbackDevice(): RuntimeDevice {
  return { device_id: 'cpu', lease_key: 'cpu', kind: 'cpu', name: 'CPU', accelerator: 'cpu', available: true, supported_memory_policies: ['default'] }
}

function policyForDevice(device: RuntimeDevice): WorkerConfiguration['memory_policy'] {
  const kind = device.supported_memory_policies[0] ?? 'default'
  if (kind === 'tensorflow_limit_mb') return { kind, limit_mb: 1024 }
  if (kind === 'jax_preallocate_fraction' || kind === 'pytorch_fraction') return { kind, fraction: 0.5 }
  return { kind }
}

function extensionFor(capability: ExportCapability | null, options: ExportOptions): string {
  if (!capability) return ''
  if (capability.format === 'onnx' && options.external_data === 'always') return '.onnx.zip'
  return capability.extension
}

function payloadLabel(payload: ExportCapability['payload']): string {
  return payload === 'full_model' ? 'Full model' : payload === 'weights_only' ? 'Weights only' : payload === 'source' ? 'Generated source' : 'Project graph'
}

function conversionLabel(conversion: ExportCapability['conversion']): string {
  return conversion === 'native' ? 'Native' : conversion === 'bridge' ? 'Verified bridge' : conversion === 'keras' ? 'Keras adapter' : 'Canonical'
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

function safeFilename(value: string): string {
  return value.replaceAll(/[^a-zA-Z0-9_-]/g, '_') || 'model'
}

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

function messageFor(error: unknown): string {
  return error instanceof ApiError ? error.message : error instanceof Error ? error.message : String(error)
}
