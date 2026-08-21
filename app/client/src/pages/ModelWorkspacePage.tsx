import { useEffect, useState } from 'react'
import { ArrowLeft, Box, Copy, Download, FileCode2, GitBranch, Info, Play, Settings2, Trash2, UploadCloud } from 'lucide-react'
import { Link, NavLink, Outlet, useNavigate, useOutletContext, useParams } from 'react-router-dom'
import { ApiError, api } from '../api/client'
import { CompilationStatusBadge, ValidationStatusBadge } from '../components/ModelStatusBadge'
import { ReadOnlyGraphPreview } from '../components/ReadOnlyGraphPreview'
import type { ProjectDocument } from '../domain/types'
import { implementationLabel } from '../domain/types'
import { useAppResources } from '../state/useAppResources'

export interface ModelWorkspaceContext {
  project: ProjectDocument
  refresh: () => Promise<ProjectDocument>
}

export function ModelWorkspacePage() {
  const { modelId } = useParams<{ modelId: string }>()
  const { refreshProjects } = useAppResources()
  const [project, setProject] = useState<ProjectDocument | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [deleteBusy, setDeleteBusy] = useState(false)
  const navigate = useNavigate()

  const refresh = async () => {
    if (!modelId) throw new Error('Missing model id')
    const next = await api.getProject(modelId)
    setProject(next)
    await refreshProjects()
    return next
  }

  useEffect(() => {
    let active = true
    setLoading(true)
    setError(null)
    if (!modelId) return
    api.getProject(modelId).then((next) => {
      if (active) setProject(next)
    }).catch((reason) => {
      if (active) setError(messageFor(reason))
    }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [modelId])

  const duplicate = async () => {
    if (!project) return
    try {
      const copy = await api.duplicateProject(project.id, `${project.name} copy`)
      await refreshProjects()
      navigate(`/models/${copy.id}`)
    } catch (reason) { setError(`Duplicate failed: ${messageFor(reason)}`) }
  }

  const remove = async () => {
    if (!project) return
    setDeleteBusy(true)
    try {
      await api.deleteProject(project.id)
      await refreshProjects()
      navigate('/models')
    } catch (reason) {
      setError(`Delete failed: ${messageFor(reason)}`)
      setDeleteBusy(false)
      setDeleteOpen(false)
    }
  }

  const exportJson = async () => {
    if (!project) return
    try {
      const bundle = await api.exportProject(project.id)
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${safeFilename(project.name)}.neuro-weave.json`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (reason) { setError(`Export failed: ${messageFor(reason)}`) }
  }

  if (loading) return <section className="page-state panel"><span className="eyebrow">MODEL WORKSPACE</span><h1>Loading model…</h1></section>
  if (error || !project) return <section className="page-state panel"><span className="eyebrow">MODEL WORKSPACE</span><h1>Model unavailable</h1><p>{error ?? 'The requested model could not be found.'}</p><Link className="primary inline-action" to="/models">Back to models</Link></section>

  return <div className="model-workspace-page">
    <header className="workspace-header">
      <div className="workspace-header-top"><Link className="back-link" to="/models"><ArrowLeft size={15} />All models</Link><span className="workspace-kicker">MODEL WORKSPACE</span></div>
      <div className="workspace-title-row"><div><h1>{project.name}</h1><p>{project.description || 'No description added.'}</p></div><div className="workspace-header-actions"><Link className="secondary inline-action" to={`/builder/${project.id}`}><Settings2 size={15} />Edit architecture</Link><button className="secondary" onClick={duplicate}><Copy size={15} />Duplicate</button><button className="icon-button danger-button" title="Delete model" aria-label="Delete model" onClick={() => setDeleteOpen(true)}><Trash2 size={16} /></button></div></div>
      <div className="workspace-meta"><ValidationStatusBadge status={project.validation_status} /><CompilationStatusBadge compiled={Boolean(project.generated_source)} target={implementationLabel(project.execution_backend, project.keras_enabled)} /><span><GitBranch size={13} />Revision {project.current_revision}</span><span><Box size={13} />{project.graph.nodes.length} nodes · {project.graph.edges.length} connections</span><span><FileCode2 size={13} />{implementationLabel(project.execution_backend, project.keras_enabled)}</span></div>
      <nav className="workspace-tabs" aria-label="Model workspace sections"><NavLink to={`/models/${project.id}`} end>Overview</NavLink><NavLink to={`/models/${project.id}/compilation`}>Compilation</NavLink><NavLink to={`/models/${project.id}/inference`}>Inference</NavLink></nav>
    </header>
    <div className="workspace-body"><Outlet context={{ project, refresh } satisfies ModelWorkspaceContext} /></div>
    <div className="workspace-footer-actions"><button className="secondary" onClick={exportJson}><Download size={15} />Export JSON</button><Link className="secondary inline-action" to={`/models/${project.id}/compilation`}><UploadCloud size={15} />Open compilation</Link></div>
    {deleteOpen && <div className="dialog-backdrop"><section className="confirm-dialog panel" role="alertdialog" aria-modal="true" aria-labelledby="delete-model-title"><span className="eyebrow">DELETE MODEL</span><h2 id="delete-model-title">Delete “{project.name}”?</h2><p>This permanently removes the saved architecture, revisions, and generated snapshots. This action cannot be undone.</p><div className="dialog-actions"><button className="secondary" onClick={() => setDeleteOpen(false)} disabled={deleteBusy}>Cancel</button><button className="primary danger-action" onClick={remove} disabled={deleteBusy}>{deleteBusy ? 'Deleting…' : 'Delete permanently'}</button></div></section></div>}
  </div>
}

export function ModelOverviewPage() {
  const { project } = useOutletContext<ModelWorkspaceContext>()
  const { manifests } = useAppResources()
  const config = project.graph.model_configuration
  const compiled = Boolean(project.generated_source)
  const metricList = config.metrics.__default__?.join(', ') || 'None configured'
  return <div className="overview-page">
    <div className="overview-grid"><section className="overview-preview panel"><div className="section-heading"><div><span className="eyebrow">ARCHITECTURE PREVIEW</span><h2>Read-only graph</h2></div><span className="muted">Open Builder to edit</span></div><ReadOnlyGraphPreview graph={project.graph} manifests={manifests} /></section><aside className="overview-summary"><section className="summary-card workspace-summary-card"><div className="section-heading"><div><span className="eyebrow">MODEL SUMMARY</span><h2>Configuration</h2></div><Info size={16} /></div><dl className="detail-list"><div><dt>Execution backend</dt><dd>{project.execution_backend}</dd></div><div><dt>Keras</dt><dd>{project.keras_enabled ? 'Enabled' : 'Disabled'}</dd></div><div><dt>Optimizer</dt><dd>{config.optimizer}</dd></div><div><dt>Learning rate</dt><dd>{String(config.optimizer_parameters.learning_rate ?? '—')}</dd></div><div><dt>Default loss</dt><dd>{config.losses.__default__ ?? 'Not configured'}</dd></div><div><dt>Metrics</dt><dd>{metricList}</dd></div><div><dt>Mixed precision</dt><dd>{config.mixed_precision ? 'Enabled' : 'Disabled'}</dd></div></dl></section><section className="summary-card workspace-summary-card"><div className="section-heading"><div><span className="eyebrow">CURRENT REVISION</span><h2>Readiness</h2></div></div><div className="readiness-stack"><div><span>Validation</span><ValidationStatusBadge status={project.validation_status} /></div><div><span>Source compilation</span><CompilationStatusBadge compiled={compiled} target={implementationLabel(project.execution_backend, project.keras_enabled)} /></div><div><span>Generated source</span><strong>{project.generated_source ? 'Available' : 'Not generated'}</strong></div></div></section></aside></div>
    <section className="workspace-actions-section"><div className="section-heading"><div><span className="eyebrow">MODEL ACTIONS</span><h2>Choose a workflow</h2></div><span className="muted">Each workflow has its own workspace.</span></div><div className="workspace-action-grid"><Link className="workspace-action-card" to={`/models/${project.id}/compilation`}><FileCode2 size={20} /><span><strong>Compilation</strong><small>Generate source and export a runnable artifact.</small></span><span className="action-arrow">→</span></Link><Link className="workspace-action-card" to={`/models/${project.id}/inference`}><Play size={20} /><span><strong>Inference</strong><small>Run one safe dummy-tensor forward pass.</small></span><span className="action-arrow">→</span></Link><button className="workspace-action-card is-disabled" disabled><UploadCloud size={20} /><span><strong>Training</strong><small>Dataset workflows arrive in a later phase.</small></span><span className="planned-badge">PLANNED</span></button></div></section>
  </div>
}

function safeFilename(value: string): string { return value.replaceAll(/[^a-zA-Z0-9_-]/g, '_') || 'model' }
function messageFor(error: unknown): string { return error instanceof ApiError ? error.message : error instanceof Error ? error.message : String(error) }
