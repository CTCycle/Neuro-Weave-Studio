import { useMemo, useRef, useState } from 'react'
import { Box, FilePlus2, Filter, FolderOpen, Import, Search } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { CompilationStatusBadge, ValidationStatusBadge } from '../components/ModelStatusBadge'
import type { ExecutionBackend, ProjectDocument, ProjectExportBundle, ProjectSummary } from '../domain/types'
import { executionBackendShortLabels } from '../domain/types'
import { useAppResources } from '../state/useAppResources'

type CatalogFilter = 'all' | 'compiled' | 'uncompiled' | 'valid' | 'attention'

export function ModelCatalogPage() {
  const { projects, loading, error, refreshProjects, backendStatuses } = useAppResources()
  const navigate = useNavigate()
  const importInput = useRef<HTMLInputElement>(null)
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<CatalogFilter>('all')
  const [notice, setNotice] = useState<string | null>(null)
  const filtered = useMemo(() => projects.filter((project) => {
    const matchesQuery = `${project.name} ${project.description} ${project.execution_backend} ${project.keras_enabled ? 'keras' : 'native'}`.toLowerCase().includes(query.toLowerCase())
    const compiled = project.has_generated_source
    const matchesFilter = filter === 'all' || (filter === 'compiled' && compiled) || (filter === 'uncompiled' && !compiled) || (filter === 'valid' && project.validation_status === 'valid') || (filter === 'attention' && project.validation_status !== 'valid')
    return matchesQuery && matchesFilter
  }), [filter, projects, query])

  const importModel = async (file: File) => {
    try {
      const isKeras = file.name.toLowerCase().endsWith('.keras')
      let project: ProjectDocument
      let warnings: string[] = []
      if (isKeras) {
        const imported = await api.importKerasArchitecture(file, preferredBackend(backendStatuses), importName(file.name))
        project = imported.project
        warnings = imported.warnings
      } else {
        project = await api.importProject(JSON.parse(await file.text()) as ProjectExportBundle)
      }
      await refreshProjects()
      if (warnings.length) setNotice(warnings.join(' '))
      navigate(`/models/${project.id}`)
    } catch (reason) {
      setNotice(`Import failed: ${reason instanceof Error ? reason.message : String(reason)}`)
    } finally { if (importInput.current) importInput.current.value = '' }
  }

  if (loading) return <section className="page-state panel"><span className="eyebrow">MODEL CATALOG</span><h1>Loading saved models…</h1></section>

  return <div className="catalog-page"><header className="catalog-page-header"><div><span className="eyebrow">MODEL LIBRARY</span><h1>Saved models</h1><p>Browse architectures, review readiness, and open a dedicated model workflow.</p></div><div className="catalog-header-actions"><Link className="primary inline-action" to="/builder"><FilePlus2 size={16} />New model</Link><button className="secondary" onClick={() => importInput.current?.click()}><Import size={16} />Import model</button><input ref={importInput} type="file" accept="application/json,.json,.keras,application/vnd.keras.zip" hidden onChange={(event) => event.target.files?.[0] && importModel(event.target.files[0])} /></div></header><section className="catalog-controls panel"><label className="search-field"><Search size={16} /><input aria-label="Search models" placeholder="Search saved models" value={query} onChange={(event) => setQuery(event.target.value)} /></label><div className="catalog-filter-label"><Filter size={14} /><span>Filter</span></div><div className="catalog-filter-tabs" role="tablist" aria-label="Model filters">{(['all', 'compiled', 'uncompiled', 'valid', 'attention'] as CatalogFilter[]).map((item) => <button key={item} className={filter === item ? 'active' : ''} onClick={() => setFilter(item)}>{filterLabel(item)}</button>)}</div><span className="count-pill">{filtered.length} of {projects.length}</span></section>{error && <section className="catalog-inline-error panel"><strong>Some catalog data is unavailable.</strong><span>{error}</span></section>}{filtered.length === 0 ? <section className="catalog-empty panel"><FolderOpen size={28} /><h2>{projects.length ? 'No models match this view' : 'Your model library is empty'}</h2><p>{projects.length ? 'Try another search or filter.' : 'Create an architecture in Model Builder, then it will appear here.'}</p>{projects.length === 0 && <Link className="primary inline-action" to="/builder"><FilePlus2 size={16} />Create your first model</Link>}</section> : <div className="model-card-grid">{filtered.map((project) => <ModelCard key={project.id} project={project} />)}</div>}{notice && <button className="notice catalog-notice" aria-label="Dismiss import notice" aria-live="polite" onClick={() => setNotice(null)}>{notice}</button>}</div>
}

function ModelCard({ project }: { project: ProjectSummary }) {
  const compiled = project.has_generated_source
  return <Link className="model-card panel" to={`/models/${project.id}`}><div className="model-card-top"><span className="eyebrow">MODEL</span><span className="revision-label">r{project.current_revision}</span></div><div className="model-card-title"><h2>{project.name}</h2><span className="model-card-arrow">→</span></div><p>{project.description || 'No description added.'}</p><div className="model-card-meta"><ValidationStatusBadge status={project.validation_status} /><span className="model-profile-badge">{executionBackendShortLabels[project.execution_backend]}</span><span className="model-profile-badge">{project.keras_enabled ? 'Keras' : 'Native'}</span><CompilationStatusBadge compiled={compiled} /></div><div className="model-card-stats"><span><strong>{project.node_count}</strong> nodes</span><span><strong>{project.edge_count}</strong> connections</span></div><footer><span>Updated {formatUpdated(project.updated_at)}</span><Box size={13} /></footer></Link>
}

function filterLabel(filter: CatalogFilter): string { return filter === 'all' ? 'All models' : filter === 'attention' ? 'Needs attention' : filter[0].toUpperCase() + filter.slice(1) }
function formatUpdated(value: string): string { const date = new Date(value); return Number.isNaN(date.valueOf()) ? 'recently' : new Intl.DateTimeFormat(undefined, { year: 'numeric', month: 'short', day: 'numeric' }).format(date) }
function preferredBackend(statuses: { id: ExecutionBackend; available: boolean }[]): ExecutionBackend { return statuses.find((item) => item.id === 'pytorch' && item.available)?.id ?? statuses.find((item) => item.available)?.id ?? 'pytorch' }
function importName(value: string): string { return value.replace(/.(keras|json)$/i, '').replaceAll(/[_-]+/g, ' ').trim() || 'Imported Keras architecture' }
