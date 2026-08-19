import { useEffect, useState } from 'react'
import { Box, ClipboardCheck, Download, LoaderCircle, RefreshCw } from 'lucide-react'
import { api } from '../api/client'
import { CodePreview } from '../components/CodePreview'
import { CompilationStatusBadge } from '../components/ModelStatusBadge'
import { codeTargetLabels, codeTargetShortLabels } from '../domain/types'
import { useAppResources } from '../state/useAppResources'
import { Link, useOutletContext } from 'react-router-dom'
import type { ModelWorkspaceContext } from './ModelWorkspacePage'

export function ModelCompilationPage() {
  const { project, refresh } = useOutletContext<ModelWorkspaceContext>()
  const { backendStatuses, selectedExecutionBackend } = useAppResources()
  const [code, setCode] = useState(project.generated_code_snapshots[project.code_target] ?? null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const compiled = Boolean(project.generated_code_snapshots[project.code_target])
  const selectedBackendStatus = backendStatuses.find((item) => item.id === selectedExecutionBackend)

  useEffect(() => {
    setCode(project.generated_code_snapshots[project.code_target] ?? null)
  }, [project])
  const compile = async () => {
    setBusy(true)
    setNotice(null)
    try {
      const result = await api.sourceForProject(project.id, project.code_target)
      setCode(result.code)
      await refresh()
      setNotice(`Compiled ${codeTargetShortLabels[project.code_target]} source for revision ${project.current_revision}.`)
    } catch (error) {
      setNotice(`Compilation failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally { setBusy(false) }
  }

  const exportBinary = async () => {
    if (!selectedExecutionBackend) return
    setBusy(true)
    try {
      const blob = await api.exportKerasModel(project.id, selectedExecutionBackend)
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${safeFilename(project.name)}.keras`
      anchor.click()
      URL.revokeObjectURL(url)
      setNotice(`Exported a Keras model with ${selectedExecutionBackend}.`)
    } catch (error) {
      setNotice(`Binary export failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally { setBusy(false) }
  }

  return <div className="workflow-page compilation-page"><div className="workflow-intro"><div><span className="eyebrow">COMPILATION</span><h2>Generate a model artifact</h2><p>Compilation creates source for the current saved revision. It does not train weights.</p></div><CompilationStatusBadge compiled={compiled} target={codeTargetShortLabels[project.code_target]} /></div><div className="workflow-grid"><section className="workflow-card panel"><div className="section-heading"><div><span className="eyebrow">SOURCE TARGET</span><h3>{codeTargetLabels[project.code_target]}</h3></div><ClipboardCheck size={18} /></div><p className="muted">Revision {project.current_revision} is the saved graph used for this source snapshot.</p><div className="workflow-stat-row"><div><span>Target</span><strong>{codeTargetShortLabels[project.code_target]}</strong></div><div><span>Revision</span><strong>r{project.current_revision}</strong></div><div><span>Source</span><strong>{compiled ? 'Available' : 'Not generated'}</strong></div></div><button className="primary" onClick={compile} disabled={busy}>{busy ? <LoaderCircle className="spin" size={16} /> : compiled ? <RefreshCw size={16} /> : <ClipboardCheck size={16} />}{busy ? 'Compiling…' : compiled ? 'Recompile source' : 'Compile source'}</button></section><section className="workflow-card panel"><div className="section-heading"><div><span className="eyebrow">BINARY EXPORT</span><h3>Runnable Keras artifact</h3></div><Box size={18} /></div><p className="muted">The binary export uses the selected operational backend and remains separate from source compilation.</p><div className="backend-selection-summary"><span>Session backend</span>{selectedBackendStatus ? <strong>{selectedBackendStatus.id} · {(selectedBackendStatus.accelerator ?? 'cpu').toUpperCase()}</strong> : <Link to="/environment">Choose an operational backend in Environment</Link>}</div><button className="secondary" onClick={exportBinary} disabled={!selectedExecutionBackend || busy}><Download size={16} />Export .keras</button></section></div>{code ? <CodePreview code={code} embedded /> : <section className="generated-code-placeholder panel"><FileCodeGlyph /><strong>No source snapshot yet</strong><span>Compile this saved revision to inspect generated code here.</span></section>}{notice && <button className="notice workflow-notice" onClick={() => setNotice(null)}>{notice}</button>}</div>
}

function FileCodeGlyph() { return <span className="placeholder-glyph"><ClipboardCheck size={22} /></span> }
function safeFilename(value: string): string { return value.replaceAll(/[^a-zA-Z0-9_-]/g, '_') || 'model' }
