import { useEffect, useState } from 'react'
import { Box, ClipboardCheck, LoaderCircle, RefreshCw } from 'lucide-react'
import { useOutletContext } from 'react-router-dom'
import { api } from '../api/client'
import { CodePreview } from '../components/CodePreview'
import { CompilationStatusBadge } from '../components/ModelStatusBadge'
import { ModelExportModal } from '../components/ModelExportModal'
import { implementationLabel } from '../domain/types'
import type { ModelWorkspaceContext } from './ModelWorkspacePage'

export function ModelCompilationPage() {
  const { project, refresh } = useOutletContext<ModelWorkspaceContext>()
  const [code, setCode] = useState(project.generated_source ?? null)
  const [sourceBusy, setSourceBusy] = useState(false)
  const [exportOpen, setExportOpen] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const compiled = Boolean(project.generated_source)

  useEffect(() => {
    setCode(project.generated_source ?? null)
  }, [project])

  const compile = async () => {
    setSourceBusy(true)
    setNotice(null)
    try {
      const result = await api.sourceForProject(project.id)
      setCode(result.code)
      await refresh()
      setNotice(`Compiled ${implementationLabel(project.execution_backend, project.keras_enabled)} source for revision ${project.current_revision}.`)
    } catch (error) {
      setNotice(`Compilation failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally { setSourceBusy(false) }
  }

  return <div className="workflow-page compilation-page">
    <div className="workflow-intro"><div><span className="eyebrow">COMPILATION</span><h2>Generate a model artifact</h2><p>Compilation creates source for the current saved revision. It does not train weights.</p></div><CompilationStatusBadge compiled={compiled} target={implementationLabel(project.execution_backend, project.keras_enabled)} /></div>
    <div className="workflow-grid">
      <section className="workflow-card panel"><div className="section-heading"><div><span className="eyebrow">EXECUTION PROFILE</span><h3>{implementationLabel(project.execution_backend, project.keras_enabled)}</h3></div><ClipboardCheck size={18} /></div><p className="muted">Revision {project.current_revision} is the saved graph used for this source snapshot.</p><div className="workflow-stat-row"><div><span>Backend</span><strong>{project.execution_backend}</strong></div><div><span>Keras</span><strong>{project.keras_enabled ? 'Enabled' : 'Disabled'}</strong></div><div><span>Source</span><strong>{compiled ? 'Available' : 'Not generated'}</strong></div></div><button className="primary" onClick={compile} disabled={sourceBusy}>{sourceBusy ? <LoaderCircle className="spin" size={16} /> : compiled ? <RefreshCw size={16} /> : <ClipboardCheck size={16} />}{sourceBusy ? 'Compiling…' : compiled ? 'Recompile source' : 'Compile source'}</button></section>
      <section className="workflow-card panel"><div className="section-heading"><div><span className="eyebrow">UNIFIED EXPORT</span><h3>Choose an artifact</h3></div><Box size={18} /></div><p className="muted">Inspect the live compatibility matrix, configure format-specific options, and download only after structural reload and inference verification.</p><div className="workflow-stat-row"><div><span>Profile</span><strong>{project.keras_enabled ? 'Keras' : 'Native'}</strong></div><div><span>Revision</span><strong>{project.current_revision}</strong></div><div><span>Training state</span><strong>Unavailable</strong></div></div><button className="secondary" type="button" onClick={() => setExportOpen(true)}><Box size={16} />Open export center</button></section>
    </div>
    {code ? <CodePreview code={code} embedded downloadFilename={`${safeFilename(project.name)}.${project.keras_enabled ? 'keras' : project.execution_backend}.py`} /> : <section className="generated-code-placeholder panel"><FileCodeGlyph /><strong>No source snapshot yet</strong><span>Compile this saved revision to inspect generated code here.</span></section>}
    {notice && <button className="notice workflow-notice" onClick={() => setNotice(null)}>{notice}</button>}
    {exportOpen && <ModelExportModal project={project} onClose={() => setExportOpen(false)} />}
  </div>
}

function FileCodeGlyph() { return <span className="placeholder-glyph"><ClipboardCheck size={22} /></span> }
function safeFilename(value: string): string { return value.replaceAll(/[^a-zA-Z0-9_-]/g, '_') || 'model' }
