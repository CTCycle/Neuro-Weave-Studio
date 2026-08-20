import { useEffect, useRef, useState } from 'react'
import { Box, ClipboardCheck, Download, LoaderCircle, RefreshCw, X } from 'lucide-react'
import { Link, useOutletContext } from 'react-router-dom'
import { api } from '../api/client'
import { CodePreview } from '../components/CodePreview'
import { CompilationStatusBadge } from '../components/ModelStatusBadge'
import type { JobResponse } from '../domain/types'
import { codeTargetLabels, codeTargetShortLabels } from '../domain/types'
import { useAppResources } from '../state/useAppResources'
import type { ModelWorkspaceContext } from './ModelWorkspacePage'

export function ModelCompilationPage() {
  const { project, refresh } = useOutletContext<ModelWorkspaceContext>()
  const { backendStatuses, runtimeSelection } = useAppResources()
  const [code, setCode] = useState(project.generated_code_snapshots[project.code_target] ?? null)
  const [sourceBusy, setSourceBusy] = useState(false)
  const [exportBusy, setExportBusy] = useState(false)
  const [exportJob, setExportJob] = useState<JobResponse | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const mounted = useRef(true)
  const compiled = Boolean(project.generated_code_snapshots[project.code_target])
  const selectedBackendStatus = backendStatuses.find((item) => item.id === runtimeSelection?.backend)

  useEffect(() => {
    setCode(project.generated_code_snapshots[project.code_target] ?? null)
  }, [project])

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false }
  }, [])

  const compile = async () => {
    setSourceBusy(true)
    setNotice(null)
    try {
      const result = await api.sourceForProject(project.id, project.code_target)
      setCode(result.code)
      await refresh()
      setNotice(`Compiled ${codeTargetShortLabels[project.code_target]} source for revision ${project.current_revision}.`)
    } catch (error) {
      setNotice(`Compilation failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally { setSourceBusy(false) }
  }

  const exportKeras = async () => {
    if (!runtimeSelection) return
    setExportBusy(true)
    setNotice(null)
    try {
      let job = await api.submitKerasExport(project.id, runtimeSelection)
      if (mounted.current) setExportJob(job)
      while (mounted.current && !['succeeded', 'terminated', 'failed'].includes(job.state)) {
        await new Promise((resolve) => window.setTimeout(resolve, 300))
        job = await api.getJob(job.id)
        if (mounted.current) setExportJob(job)
      }
      if (!mounted.current) return
      if (job.state !== 'succeeded') {
        setNotice(`Keras export failed: ${job.error?.message ?? `worker ${job.state}`}`)
        return
      }
      const blob = await api.jobArtifact(job.id)
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${safeFilename(project.name)}.keras`
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(url)
      setNotice(`Exported a verified Keras model with ${runtimeSelection.backend} on ${runtimeSelection.device_id}.`)
    } catch (error) {
      setNotice(`Keras export failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      if (mounted.current) setExportBusy(false)
    }
  }

  const stopExport = async () => {
    if (!exportJob || ['succeeded', 'terminated', 'failed'].includes(exportJob.state)) return
    try {
      setExportJob(await api.stopJob(exportJob.id))
      setNotice('Stop requested. The worker will be reaped before the device lease is released.')
    } catch (error) {
      setNotice(`Stop request failed: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  const exportActive = Boolean(exportJob && !['succeeded', 'terminated', 'failed'].includes(exportJob.state))
  return <div className="workflow-page compilation-page"><div className="workflow-intro"><div><span className="eyebrow">COMPILATION</span><h2>Generate a model artifact</h2><p>Compilation creates source for the current saved revision. It does not train weights.</p></div><CompilationStatusBadge compiled={compiled} target={codeTargetShortLabels[project.code_target]} /></div><div className="workflow-grid"><section className="workflow-card panel"><div className="section-heading"><div><span className="eyebrow">SOURCE TARGET</span><h3>{codeTargetLabels[project.code_target]}</h3></div><ClipboardCheck size={18} /></div><p className="muted">Revision {project.current_revision} is the saved graph used for this source snapshot.</p><div className="workflow-stat-row"><div><span>Target</span><strong>{codeTargetShortLabels[project.code_target]}</strong></div><div><span>Revision</span><strong>r{project.current_revision}</strong></div><div><span>Source</span><strong>{compiled ? 'Available' : 'Not generated'}</strong></div></div><button className="primary" onClick={compile} disabled={sourceBusy || exportActive || exportBusy}>{sourceBusy ? <LoaderCircle className="spin" size={16} /> : compiled ? <RefreshCw size={16} /> : <ClipboardCheck size={16} />}{sourceBusy ? 'Compiling…' : compiled ? 'Recompile source' : 'Compile source'}</button></section><section className="workflow-card panel"><div className="section-heading"><div><span className="eyebrow">KERAS EXPORT</span><h3>Verified Keras artifact</h3></div><Box size={18} /></div><p className="muted">The isolated Keras worker verifies reload and inference before publishing the artifact. Native framework inference is unavailable.</p><div className="backend-selection-summary"><span>Session runtime</span>{runtimeSelection && selectedBackendStatus ? <strong>{runtimeSelection.api} · {selectedBackendStatus.id} · {runtimeSelection.device_id} · {runtimeSelection.memory_policy.kind}</strong> : <Link to="/environment">Choose an operational runtime in Environment</Link>}</div>{exportJob && <div className="job-summary"><span>{exportJob.progress.phase}</span><strong>{exportJob.progress.message}</strong><small>PID {exportJob.pid ?? 'pending'} · {exportJob.state} · {exportJob.diagnostics.process_rss_bytes ? `${Math.round(exportJob.diagnostics.process_rss_bytes / 1024 / 1024)} MB RSS` : 'memory pending'}</small></div>}{exportActive ? <button className="secondary" onClick={stopExport}><X size={16} />Stop export</button> : <button className="secondary" onClick={exportKeras} disabled={!runtimeSelection || sourceBusy || exportBusy}><Download size={16} />Export .keras</button>}</section></div>{code ? <CodePreview code={code} embedded /> : <section className="generated-code-placeholder panel"><FileCodeGlyph /><strong>No source snapshot yet</strong><span>Compile this saved revision to inspect generated code here.</span></section>}{notice && <button className="notice workflow-notice" onClick={() => setNotice(null)}>{notice}</button>}</div>
}

function FileCodeGlyph() { return <span className="placeholder-glyph"><ClipboardCheck size={22} /></span> }
function safeFilename(value: string): string { return value.replaceAll(/[^a-zA-Z0-9_-]/g, '_') || 'model' }
