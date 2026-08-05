import { Box, Copy, Download, FilePlus2, LayoutDashboard, Redo2, Save, Undo2, Upload } from 'lucide-react'
import type { KerasBackend, ProjectSummary, TargetFramework } from '../domain/types'
import { useGraphStore } from '../state/graphStore'

interface Props {
  name: string
  onNameChange: (name: string) => void
  description: string
  onDescriptionChange: (description: string) => void
  framework: TargetFramework
  onFrameworkChange: (framework: TargetFramework) => void
  availableKerasBackends: KerasBackend[]
  kerasBackend: KerasBackend | null
  onKerasBackendChange: (backend: KerasBackend | null) => void
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

export function ProjectToolbar(props: Props) {
  const undo = useGraphStore((state) => state.undo)
  const redo = useGraphStore((state) => state.redo)
  return (
    <header className="project-toolbar">
      <div className="brand-mark"><span>NW</span><div><strong>NeuroWeave</strong><small>MODEL STUDIO</small></div></div>
      <div className="project-identity"><input className="project-title" value={props.name} onChange={(event) => props.onNameChange(event.target.value)} aria-label="Project name" /><input className="project-description" value={props.description} onChange={(event) => props.onDescriptionChange(event.target.value)} aria-label="Project description" placeholder="Optional project description" /></div>
      <select className="framework-select" value={props.framework} onChange={(event) => props.onFrameworkChange(event.target.value as TargetFramework)} aria-label="Target framework"><option value="keras">Keras</option><option value="tensorflow_keras">TensorFlow through Keras</option><option value="pytorch">PyTorch</option></select>
      <select className="framework-select" value={props.kerasBackend ?? ''} onChange={(event) => props.onKerasBackendChange((event.target.value || null) as KerasBackend | null)} aria-label="Keras backend" disabled={props.availableKerasBackends.length === 0 || props.busy} title={props.availableKerasBackends.length === 0 ? 'Install a Keras backend from the launcher' : 'Backend used for Keras binary export'}><option value="">{props.availableKerasBackends.length === 0 ? 'Install Keras backend' : 'Select Keras backend…'}</option>{props.availableKerasBackends.map((backend) => <option key={backend} value={backend}>{backend === 'tensorflow' ? 'Keras · TensorFlow' : 'Keras · Torch'}</option>)}</select>
      <div className="toolbar-group">
        <button title="New project" onClick={props.onNew}><FilePlus2 size={17} /></button>
        <select value={props.currentProjectId ?? ''} onChange={(event) => event.target.value && props.onOpen(event.target.value)} aria-label="Open project"><option value="">Open project…</option>{props.projects.map((project) => <option key={project.id} value={project.id}>{project.name} · r{project.current_revision}</option>)}</select>
        <button title="Save" disabled={props.busy} onClick={props.onSave}><Save size={17} /></button>
        <button title="Duplicate project" disabled={!props.currentProjectId} onClick={props.onDuplicate}><Copy size={17} /></button>
      </div>
      <div className="toolbar-group"><button title="Undo" onClick={undo}><Undo2 size={17} /></button><button title="Redo" onClick={redo}><Redo2 size={17} /></button></div>
      <div className="toolbar-actions"><button className="secondary" onClick={props.onImport}><Upload size={16} />Import JSON</button><button className="secondary" disabled={!props.currentProjectId} onClick={props.onExport}><Download size={16} />Export JSON</button><button className="secondary" disabled={!props.currentProjectId || props.framework === 'pytorch' || !props.kerasBackend} onClick={props.onBinaryExport}><Box size={16} />Export .keras</button><button className="secondary" onClick={props.onValidate}><LayoutDashboard size={16} />Validate</button><button className="primary" onClick={props.onCode}>Preview code</button></div>
    </header>
  )
}
