import { useEffect, useMemo, useState } from 'react'
import { ClipboardCheck, FileCode2, LoaderCircle, RefreshCw, Settings2 } from 'lucide-react'
import { useOutletContext } from 'react-router-dom'
import { api } from '../api/client'
import { CodePreview } from '../components/CodePreview'
import { CompilationConfigurationModal } from '../components/CompilationConfigurationModal'
import { CompilationStatusBadge } from '../components/ModelStatusBadge'
import { ValidationPanel } from '../components/ValidationPanel'
import { implementationLabel } from '../domain/types'
import type { CompilationCatalog, ModelConfiguration, NativeCompilationCapability, ValidationResult } from '../domain/types'
import { useAppResources } from '../state/useAppResources'
import type { ModelWorkspaceContext } from './ModelWorkspacePage'

export function ModelCompilationPage() {
  const { project, refresh } = useOutletContext<ModelWorkspaceContext>()
  const { compilationCapabilities, readiness } = useAppResources()
  const [code, setCode] = useState(project.generated_source ?? null)
  const [sourceBusy, setSourceBusy] = useState(false)
  const [configurationOpen, setConfigurationOpen] = useState(false)
  const [configurationBusy, setConfigurationBusy] = useState(false)
  const [validationBusy, setValidationBusy] = useState(false)
  const [validation, setValidation] = useState<ValidationResult | null>(null)
  const [validationExpanded, setValidationExpanded] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const compiled = Boolean(project.generated_source)
  const kerasCatalog = compilationCapabilities?.keras
  const nativeCapability = compilationCapabilities?.native[project.execution_backend]
  const outputIds = useMemo(() => project.graph.nodes.filter((node) => node.definition_id.startsWith('output.')).map((node) => node.id), [project.graph.nodes])

  useEffect(() => {
    setCode(project.generated_source ?? null)
  }, [project])

  const compile = async () => {
    setSourceBusy(true)
    setNotice(null)
    try {
      const result = await api.sourceForProject(project.id)
      setCode(result.code)
      const refreshed = await refresh()
      setNotice(`Compiled ${implementationLabel(refreshed.execution_backend, refreshed.keras_enabled)} source for revision ${refreshed.current_revision}.`)
    } catch (error) {
      setNotice(`Compilation failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally { setSourceBusy(false) }
  }

  const validate = async (graph = project.graph) => {
    setValidationBusy(true)
    setNotice(null)
    try {
      const result = await api.validate(graph, project.execution_backend, project.keras_enabled)
      setValidation(result)
      setValidationExpanded(true)
      setNotice(result.valid ? 'Compilation configuration is valid for this graph.' : 'Compilation configuration has blocking findings.')
      return result
    } catch (error) {
      setNotice(`Validation failed: ${error instanceof Error ? error.message : String(error)}`)
      return null
    } finally { setValidationBusy(false) }
  }

  const saveConfiguration = async (configuration: ModelConfiguration) => {
    if (configurationBusy) return
    setConfigurationBusy(true)
    setNotice(null)
    const graph = { ...project.graph, model_configuration: configuration }
    try {
      const result = await api.validate(graph, project.execution_backend, project.keras_enabled)
      setValidation(result)
      setValidationExpanded(true)
      const saved = await api.updateProject(project.id, {
        name: project.name,
        description: project.description,
        execution_backend: project.execution_backend,
        keras_enabled: project.keras_enabled,
        graph,
        expected_revision: project.current_revision,
      })
      await refresh()
      setConfigurationOpen(false)
      setCode(null)
      setNotice(result.valid ? `Saved compilation settings in revision ${saved.current_revision}.` : `Saved revision ${saved.current_revision} with validation findings.`)
    } catch (error) {
      setNotice(`Saving compilation settings failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally { setConfigurationBusy(false) }
  }

  return <div className="workflow-page compilation-page">
    <div className="workflow-intro"><div><span className="eyebrow">COMPILATION</span><h2>Configure and generate source</h2><p>Compilation owns the saved execution profile and generated source for this revision. It does not train weights or download artifacts.</p></div><CompilationStatusBadge compiled={compiled} target={implementationLabel(project.execution_backend, project.keras_enabled)} /></div>
    <div className="workflow-grid">
      <section className="workflow-card panel"><div className="section-heading"><div><span className="eyebrow">EXECUTION PROFILE</span><h3>{implementationLabel(project.execution_backend, project.keras_enabled)}</h3></div><Settings2 size={18} /></div><p className="muted">Revision {project.current_revision} is the saved graph used for this source snapshot.</p><div className="workflow-stat-row"><div><span>Backend</span><strong>{project.execution_backend}</strong></div><div><span>Configuration</span><strong>{configurationSummary(project, kerasCatalog, nativeCapability)}</strong></div><div><span>Source</span><strong>{compiled ? 'Available' : 'Not generated'}</strong></div></div><div className="workflow-card-actions"><button className="secondary" type="button" onClick={() => setConfigurationOpen(true)} disabled={!readiness.runtime || configurationBusy}><Settings2 size={16} />{configurationBusy ? 'Saving…' : 'Configure compilation'}</button><button className="secondary" type="button" onClick={() => void validate()} disabled={validationBusy}>{validationBusy ? <LoaderCircle className="spin" size={16} /> : <ClipboardCheck size={16} />}{validationBusy ? 'Validating…' : 'Validate profile'}</button></div></section>
      <section className="workflow-card panel"><div className="section-heading"><div><span className="eyebrow">GENERATED SOURCE</span><h3>{compiled ? 'Current source snapshot' : 'No source snapshot'}</h3></div><FileCode2 size={18} /></div><p className="muted">Generate inspectable source from the current saved revision. Copy is available here; artifact downloads belong to Export.</p><div className="workflow-stat-row"><div><span>Revision</span><strong>{project.current_revision}</strong></div><div><span>Profile</span><strong>{project.keras_enabled ? 'Keras API' : 'Native API'}</strong></div><div><span>Training state</span><strong>Unavailable</strong></div></div><button className="primary" onClick={compile} disabled={sourceBusy}>{sourceBusy ? <LoaderCircle className="spin" size={16} /> : compiled ? <RefreshCw size={16} /> : <ClipboardCheck size={16} />}{sourceBusy ? 'Compiling…' : compiled ? 'Recompile source' : 'Compile source'}</button></section>
    </div>
    {code ? <CodePreview code={code} embedded allowDownload={false} downloadFilename={`${safeFilename(project.name)}.${project.keras_enabled ? 'keras' : project.execution_backend}.py`} /> : <section className="generated-code-placeholder panel"><FileCodeGlyph /><strong>No source snapshot yet</strong><span>Compile this saved revision to inspect generated code here.</span></section>}
    {validation && <ValidationPanel result={validation} expanded={validationExpanded} onExpandedChange={setValidationExpanded} />}
    {notice && <button className="notice workflow-notice" onClick={() => setNotice(null)}>{notice}</button>}
    {configurationOpen && <CompilationConfigurationModal configuration={project.graph.model_configuration} kerasEnabled={project.keras_enabled} kerasCatalog={kerasCatalog} nativeCapability={nativeCapability} outputIds={outputIds} onClose={() => { if (!configurationBusy) setConfigurationOpen(false) }} onApply={(configuration) => void saveConfiguration(configuration)} />}
  </div>
}

function FileCodeGlyph() { return <span className="placeholder-glyph"><ClipboardCheck size={22} /></span> }
function safeFilename(value: string): string { return value.replaceAll(/[^a-zA-Z0-9_-]/g, '_') || 'model' }

function configurationSummary(project: ModelWorkspaceContext['project'], kerasCatalog: CompilationCatalog | null | undefined, nativeCapability: NativeCompilationCapability | null | undefined): string {
  if (project.keras_enabled) {
    const optimizer = kerasCatalog?.optimizers.find((item) => item.id === project.graph.model_configuration.keras.optimizer)?.label ?? project.graph.model_configuration.keras.optimizer
    const lossId = project.graph.model_configuration.keras.losses.__default__
    const loss = lossId ? kerasCatalog?.losses.find((item) => item.id === lossId)?.label ?? lossId : 'No loss'
    return `${optimizer} · ${loss}`
  }
  if (project.execution_backend === 'tensorflow') return `XLA ${project.graph.model_configuration.native.tensorflow.jit_compile ? 'on' : 'off'}`
  if (project.execution_backend === 'pytorch') return `torch.compile ${project.graph.model_configuration.native.pytorch.compile_enabled ? 'on' : 'off'}`
  return `${nativeCapability?.compiler ?? 'jax.jit'} ${project.graph.model_configuration.native.jax.jit_enabled ? 'on' : 'off'}`
}
