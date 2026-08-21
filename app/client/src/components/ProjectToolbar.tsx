import { Hammer, LayoutDashboard, Play, Save, Undo2, Redo2, WandSparkles } from 'lucide-react'
import type { ExecutionBackend } from '../domain/types'
import { executionBackendShortLabels } from '../domain/types'
import { useGraphStore } from '../state/graphStore'

interface Props {
  name: string
  onNameChange: (name: string) => void
  description: string
  onDescriptionChange: (description: string) => void
  executionBackend: ExecutionBackend
  onExecutionBackendChange: (backend: ExecutionBackend) => void
  kerasEnabled: boolean
  onKerasEnabledChange: (enabled: boolean) => void
  onSave: () => void
  onCompile: () => void
  onValidate: () => void
  busy: boolean
  modelId: string | null
  onOpenWorkspace: () => void
}

export function ProjectToolbar(props: Props) {
  const undo = useGraphStore((state) => state.undo)
  const redo = useGraphStore((state) => state.redo)
  return (
    <section className="builder-toolbar panel">
      <div className="builder-toolbar-heading"><div><span className="eyebrow">MODEL BUILDER</span><strong>Architecture authoring</strong></div><span className="builder-badge"><Hammer size={14} />Drafting mode</span></div>
      <div className="builder-identity">
        <input className="project-title" value={props.name} onChange={(event) => props.onNameChange(event.target.value)} aria-label="Model name" />
        <input className="project-description" value={props.description} onChange={(event) => props.onDescriptionChange(event.target.value)} aria-label="Model description" placeholder="Optional model description" />
      </div>
      <label className="toolbar-select builder-target"><span>Backend</span><select value={props.executionBackend} onChange={(event) => props.onExecutionBackendChange(event.target.value as ExecutionBackend)} aria-label="Execution backend">{Object.entries(executionBackendShortLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      <label className="toolbar-toggle"><span>Keras</span><input type="checkbox" checked={props.kerasEnabled} onChange={(event) => props.onKerasEnabledChange(event.target.checked)} aria-label="Enable Keras" /><strong>{props.kerasEnabled ? 'Enabled' : 'Native'}</strong></label>
      <div className="toolbar-group builder-history"><button title="Undo" onClick={undo}><Undo2 size={17} /></button><button title="Redo" onClick={redo}><Redo2 size={17} /></button></div>
      <div className="builder-actions"><button className="secondary" onClick={props.onValidate} disabled={props.busy}><LayoutDashboard size={16} />Validate</button><button className="secondary" onClick={props.onCompile} disabled={props.busy}><WandSparkles size={16} />Save &amp; compile</button><button className="primary" onClick={props.onSave} disabled={props.busy}><Save size={16} />Save model</button>{props.modelId && <button className="secondary" onClick={props.onOpenWorkspace} disabled={props.busy}><Play size={16} />Open workspace</button>}</div>
    </section>
  )
}
