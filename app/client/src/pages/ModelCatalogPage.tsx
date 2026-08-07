import { useMemo, useRef, useState } from 'react'
import { Box, FilePlus2, Filter, FolderOpen, Import, Search } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { CompilationStatusBadge, ValidationStatusBadge } from '../components/ModelStatusBadge'
import type { ProjectExportBundle, ProjectSummary } from '../domain/types'
import { codeTargetShortLabels } from '../domain/types'
import { useAppResources } from '../state/useAppResources'

type CatalogFilter = 'all' | 'compiled' | 'uncompiled' | 'valid' | 'attention'

export function ModelCatalogPage() {
  const { projects, loading, error, refreshProjects } = useAppResources()
  const navigate = useNavigate()
  const importInput = useRef<HTMLInputElement>(null)
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<CatalogFilter>('all')
  const [notice, setNotice] = useState<string | null>(null)
  const filtered = useMemo(() => projects.filter((project) => {
    const matchesQuery = `${project.name} ${project.description} ${project.code_target}`.toLowerCase().includes(query.toLowerCase())
    const compiled = project.compiled_targets.includes(project.code_target)
    const matchesFilter = filter === 'all' || (filter === 'compiled' && compiled) || (filter === 'uncompiled' && !compiled) || (filter === 'valid' && project.validation_status === 'valid') || (filter === 'attention' && project.validation_status !== 'valid')
    return matchesQuery && matchesFilter
  }), [filter, projects, query])

  const importModel = async (file: File) => {
    try {
      const bundle = JSON.parse(await file.text()) as ProjectExportBundle
      const project = await api.importProject(bundle)
      await refreshProjects()
      navigate(`/models/${project.id}`)
    } catch (reason) {
      setNotice(`Import failed: ${reason instanceof Error ? reason.message : String(reason)}`)
    } finally { if (importInput.current) importInput.current.value = '' }
  }

  if (loading) return <section className="page-state panel"><span className="eyebrow">MODEL CATALOG</span><h1>Loading saved models…</h1></section>

  return <div className="catalog-page"><header className="catalog-page-header"><div><span className="eyebrow">MODEL LIBRARY</span><h1>Saved models</h1><p>Browse architectures, review readiness, and open a dedicated model workflow.</p></div><div className="catalog-header-actions"><Link className="primary inline-action" to="/builder"><FilePlus2 size={16} />New model</Link><button className="secondary" onClick={() => importInput.current?.click()}><Import size={16} />Import JSON</button><input ref={importInput} type="file" accept="application/json,.json" hidden onChange={(event) => event.target.files?.[0] && importModel(event.target.files[0])} /></div></header><section className="catalog-controls panel"><label className="search-field"><Search size={16} /><input aria-label="Search models" placeholder="Search saved models" value={query} onChange={(event) => setQuery(event.target.value)} /></label><div className="catalog-filter-label"><Filter size={14} /><span>Filter</span></div><div className="catalog-filter-tabs" role="tablist" aria-label="Model filters">{(['all', 'compiled', 'uncompiled', 'valid', 'attention'] as CatalogFilter[]).map((item) => <button key={item} className={filter === item ? 'active' : ''} onClick={() => setFilter(item)}>{filterLabel(item)}</button>)}</div><span className="count-pill">{filtered.length} of {projects.length}</span></section>{error && <section className="catalog-inline-error panel"><strong>Some catalog data is unavailable.</strong><span>{error}</span></section>}{filtered.length === 0 ? <section className="catalog-empty panel"><FolderOpen size={28} /><h2>{projects.length ? 'No models match this view' : 'Your model library is empty'}</h2><p>{projects.length ? 'Try another search or filter.' : 'Create an architecture in Model Builder, then it will appear here.'}</p>{projects.length === 0 && <Link className="primary inline-action" to="/builder"><FilePlus2 size={16} />Create your first model</Link>}</section> : <div className="model-card-grid">{filtered.map((project) => <ModelCard key={project.id} project={project} />)}</div>}{notice && <button className="notice catalog-notice" onClick={() => setNotice(null)}>{notice}</button>}</div>
}

function ModelCard({ project }: { project: ProjectSummary }) {
  const compiled = project.compiled_targets.includes(project.code_target)
  return <Link className="model-card panel" to={`/models/${project.id}`}><div className="model-card-top"><span className="eyebrow">MODEL</span><span className="revision-label">r{project.current_revision}</span></div><div className="model-card-title"><h2>{project.name}</h2><span className="model-card-arrow">→</span></div><p>{project.description || 'No description added.'}</p><div className="model-card-meta"><ValidationStatusBadge status={project.validation_status} /><CompilationStatusBadge compiled={compiled} /></div><div className="model-card-stats"><span><strong>{project.node_count}</strong> nodes</span><span><strong>{project.edge_count}</strong> connections</span><span><strong>{codeTargetShortLabels[project.code_target]}</strong> target</span></div><footer><span>Updated {formatUpdated(project.updated_at)}</span><Box size={13} /></footer></Link>
}

function filterLabel(filter: CatalogFilter): string { return filter === 'all' ? 'All models' : filter === 'attention' ? 'Needs attention' : filter[0].toUpperCase() + filter.slice(1) }
function formatUpdated(value: string): string { const date = new Date(value); return Number.isNaN(date.valueOf()) ? 'recently' : new Intl.DateTimeFormat(undefined, { year: 'numeric', month: 'short', day: 'numeric' }).format(date) }
