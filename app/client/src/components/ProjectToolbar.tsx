import { useEffect, useRef, useState } from 'react'
import { FileText, Hammer, LayoutDashboard, Play, Save, Undo2, Redo2, X } from 'lucide-react'
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
  onValidate: () => void
  busy: boolean
  dirty: boolean
  modelId: string | null
  onOpenWorkspace: () => void
}

export function ProjectToolbar(props: Props) {
  const undo = useGraphStore((state) => state.undo)
  const redo = useGraphStore((state) => state.redo)
  const statusLabel = props.busy ? 'Saving…' : props.dirty ? 'Drafting' : 'Saved'
  const [descriptionOpen, setDescriptionOpen] = useState(false)
  const [descriptionDraft, setDescriptionDraft] = useState(props.description)
  const descriptionControl = useRef<HTMLDivElement>(null)
  const descriptionInput = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (!descriptionOpen) setDescriptionDraft(props.description)
  }, [descriptionOpen, props.description])

  useEffect(() => {
    if (!descriptionOpen) return
    const onPointerDown = (event: PointerEvent) => {
      if (!descriptionControl.current?.contains(event.target as Node)) setDescriptionOpen(false)
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        setDescriptionDraft(props.description)
        setDescriptionOpen(false)
      }
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    requestAnimationFrame(() => descriptionInput.current?.focus())
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [descriptionOpen, props.description])

  const saveDescription = () => {
    props.onDescriptionChange(descriptionDraft.trim())
    setDescriptionOpen(false)
  }

  const clearDescription = () => {
    props.onDescriptionChange('')
    setDescriptionDraft('')
    setDescriptionOpen(false)
  }

  const cancelDescription = () => {
    setDescriptionDraft(props.description)
    setDescriptionOpen(false)
  }

  return (
    <section className="builder-toolbar panel">
      <div className={`builder-toolbar-heading ${props.dirty ? 'is-dirty' : ''}`}><span className="eyebrow">MODEL BUILDER</span><span className="builder-badge" aria-live="polite" title={props.busy ? 'Saving changes' : props.dirty ? 'Drafting with unsaved changes' : 'Saved'}><Hammer size={13} /><span>{statusLabel}</span></span></div>
      <div className="builder-identity">
        <input className="project-title" value={props.name} onChange={(event) => props.onNameChange(event.target.value)} aria-label="Model name" />
        <div className="description-control" ref={descriptionControl}>
          <button type="button" className={`toolbar-description ${props.description ? 'has-description' : ''}`} aria-expanded={descriptionOpen} aria-haspopup="dialog" onClick={() => setDescriptionOpen((open) => !open)} title={props.description ? 'Edit model description' : 'Add model description'}>
            <FileText size={14} /><span>Description</span>{props.description && <i aria-hidden="true" />}
          </button>
          {descriptionOpen && <div className="description-popover" role="dialog" aria-label="Model description editor">
            <div className="description-popover-heading"><div><span className="eyebrow">MODEL METADATA</span><strong>Description</strong></div><button type="button" className="popover-close" onClick={cancelDescription} aria-label="Close description editor"><X size={14} /></button></div>
            <textarea ref={descriptionInput} value={descriptionDraft} onChange={(event) => setDescriptionDraft(event.target.value)} maxLength={2000} placeholder="Describe the model purpose, inputs, or expected outputs." aria-label="Model description" />
            <div className="description-popover-footer"><small>{descriptionDraft.length}/2000</small><div><button type="button" className="secondary" onClick={cancelDescription}>Cancel</button>{(props.description || descriptionDraft) && <button type="button" className="secondary danger-text" onClick={clearDescription}>Clear</button>}<button type="button" className="primary" onClick={saveDescription}>Save</button></div></div>
          </div>}
        </div>
      </div>
      <label className="toolbar-select builder-target"><span>Backend</span><select value={props.executionBackend} onChange={(event) => props.onExecutionBackendChange(event.target.value as ExecutionBackend)} aria-label="Execution backend">{Object.entries(executionBackendShortLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      <label className="toolbar-toggle" title="Optional Keras API layer over the selected native backend."><span>Enable Keras</span><input type="checkbox" checked={props.kerasEnabled} onChange={(event) => props.onKerasEnabledChange(event.target.checked)} aria-label="Enable Keras" aria-describedby="keras-control-help" /><strong>{props.kerasEnabled ? 'On' : 'Off'}</strong><span id="keras-control-help" className="sr-only">Keras is an optional API layer over the selected native execution backend.</span></label>
      <div className="toolbar-group builder-history"><button type="button" aria-label="Undo" title="Undo" onClick={undo}><Undo2 size={17} /></button><button type="button" aria-label="Redo" title="Redo" onClick={redo}><Redo2 size={17} /></button></div>
      <div className="builder-actions"><button type="button" className="secondary" aria-label="Validate" onClick={props.onValidate} disabled={props.busy}><LayoutDashboard size={16} /><span className="action-label">Validate</span></button><button type="button" className="primary" aria-label="Save model" onClick={props.onSave} disabled={props.busy}><Save size={16} /><span className="action-label">Save model</span></button>{props.modelId && <button type="button" className="secondary" aria-label="Open workspace" onClick={props.onOpenWorkspace} disabled={props.busy}><Play size={16} /><span className="action-label">Open workspace</span></button>}</div>
    </section>
  )
}
