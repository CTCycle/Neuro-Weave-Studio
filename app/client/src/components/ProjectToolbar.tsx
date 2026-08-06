import { Box, Copy, Download, FilePlus2, LayoutDashboard, MoreHorizontal, Redo2, Save, Undo2, Upload } from 'lucide-react'
import type { CodeTarget, ExecutionBackend, ProjectSummary } from '../domain/types'
import { useGraphStore } from '../state/graphStore'

interface Props {
  name: string
  onNameChange: (name: string) => void
  description: string
  onDescriptionChange: (description: string) => void
  codeTarget?: CodeTarget
  onCodeTargetChange?: (target: CodeTarget) => void
  backendStatuses?: Array<{ id: ExecutionBackend; available: boolean; reason_code?: string | null }>
  executionBackend?: ExecutionBackend | null
  onExecutionBackendChange?: (backend: ExecutionBackend | null) => void
  projects: ProjectSummary[]
  currentProjectId: string | null
  onNew: () => void
  onSave: () => void
  onOpen: (id: string) => void
  onDuplicate: () => void
  onValidate: () => void
  onCode: () => void
  onExport: () => void
  onBinaryExport: () => void
  onImport: () => void
  busy: boolean
}

const targetLabels: Record<CodeTarget, string> = { keras: 'Portable Keras', tensorflow: 'Native TensorFlow', pytorch: 'Native PyTorch', jax: 'Native JAX / Flax' }

export function ProjectToolbar(props: Props) {
  const undo = useGraphStore((state) => state.undo)
  const redo = useGraphStore((state) => state.redo)
  const selectedTarget = props.codeTarget ?? 'keras'
  const statuses = props.backendStatuses ?? []
  const selectedBackend = props.executionBackend ?? null
  const backendCount = statuses.filter((item) => item.available).length
  return (
    <header className="project-toolbar">
      <div className="brand-mark"><span>NW</span><div><strong>NeuroWeave</strong><small>MODEL STUDIO</small></div></div>
      <div className="project-identity"><input className="project-title" value={props.name} onChange={(event) => props.onNameChange(event.target.value)} aria-label="Project name" /><input className="project-description" value={props.description} onChange={(event) => props.onDescriptionChange(event.target.value)} aria-label="Project description" placeholder="Optional project description" /></div>
      <div className="target-controls">
        <span className="api-badge" title="Keras 3 is the fixed model API">Keras 3 API</span>
        <label className="toolbar-select"><span>Code target</span><select value={selectedTarget} onChange={(event) => props.onCodeTargetChange?.(event.target.value as CodeTarget)} aria-label="Code target"><option value="keras">Portable Keras</option>{Object.entries(targetLabels).filter(([value]) => value !== 'keras').map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label className="toolbar-select"><span>Execution backend</span><select value={selectedBackend ?? ''} onChange={(event) => { const value = event.target.value || null; props.onExecutionBackendChange?.(value as ExecutionBackend | null) }} aria-label="Execution backend" disabled={props.busy || backendCount === 0}><option value="">{backendCount === 0 ? 'None operational' : backendCount === 1 ? 'Select backend' : 'Choose backend…'}</option>{statuses.map((backend) => <option key={backend.id} value={backend.id} disabled={!backend.available}>{`${backend.id}${backend.available ? '' : ` · ${backend.reason_code ?? 'unavailable'}`}`}</option>)}</select></label>
      </div>
      <div className="toolbar-group project-actions">
        <button title="New project" onClick={props.onNew}><FilePlus2 size={17} /></button>
        <select value={props.currentProjectId ?? ''} onChange={(event) => event.target.value && props.onOpen(event.target.value)} aria-label="Open project"><option value="">Open project…</option>{props.projects.map((project) => <option key={project.id} value={project.id}>{project.name} · r{project.current_revision}</option>)}</select>
        <button title="Save" disabled={props.busy} onClick={props.onSave}><Save size={17} /></button>
        <button title="Duplicate project" disabled={!props.currentProjectId} onClick={props.onDuplicate}><Copy size={17} /></button>
      </div>
      <div className="toolbar-group"><button title="Undo" onClick={undo}><Undo2 size={17} /></button><button title="Redo" onClick={redo}><Redo2 size={17} /></button></div>
      <div className="toolbar-actions"><button className="secondary" onClick={props.onValidate}><LayoutDashboard size={16} />Validate</button><button className="primary" onClick={props.onCode} disabled={props.busy}>Preview code</button><button className="secondary binary-action" disabled={!props.currentProjectId || !selectedBackend || props.busy} onClick={props.onBinaryExport}><Box size={16} />Export .keras</button></div>
      <details className="toolbar-overflow"><summary aria-label="More project actions"><MoreHorizontal size={18} /></summary><div className="overflow-menu"><button onClick={props.onImport}><Upload size={16} />Import JSON</button><button disabled={!props.currentProjectId} onClick={props.onExport}><Download size={16} />Export JSON</button></div></details>
    </header>
  )
}
