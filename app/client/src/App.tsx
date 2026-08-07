import { useEffect, useRef, useState } from 'react'
import { ApiError, api } from './api/client'
import { CodePreview } from './components/CodePreview'
import { GraphCanvas } from './components/GraphCanvas'
import { ModelConfigurationPanel } from './components/ModelConfigurationPanel'
import { ModelSmokeTestDrawer, type SmokeTestViewState } from './components/ModelSmokeTestDrawer'
import { NodeCatalog } from './components/NodeCatalog'
import { ProjectToolbar } from './components/ProjectToolbar'
import { PropertiesInspector } from './components/PropertiesInspector'
import { ValidationPanel } from './components/ValidationPanel'
import type { CodeTarget, ExecutionBackend, ExecutionBackendStatus, ProjectExportBundle, ProjectSummary, SmokeTestErrorDetail, ValidationResult } from './domain/types'
import { useGraphStore } from './state/graphStore'

export default function App() {
  const [projectId, setProjectId] = useState<string | null>(null)
  const [revision, setRevision] = useState(1)
  const [name, setName] = useState('Untitled model')
  const [description, setDescription] = useState('')
  const [codeTarget, setCodeTarget] = useState<CodeTarget>('keras')
  const [backendStatuses, setBackendStatuses] = useState<ExecutionBackendStatus[]>([])
  const [executionBackend, setExecutionBackend] = useState<ExecutionBackend | null>(null)
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [validation, setValidation] = useState<ValidationResult | null>(null)
  const [code, setCode] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [railPanel, setRailPanel] = useState<'catalog' | 'inspector'>('catalog')
  const [drawer, setDrawer] = useState<'catalog' | 'inspector' | null>(null)
  const [smokeTestOpen, setSmokeTestOpen] = useState(false)
  const [smokeTestState, setSmokeTestState] = useState<SmokeTestViewState>({ status: 'ready' })
  const importInput = useRef<HTMLInputElement>(null)
  const manifests = useGraphStore((state) => state.manifests)
  const setManifests = useGraphStore((state) => state.setManifests)
  const replaceGraph = useGraphStore((state) => state.replaceGraph)
  const reset = useGraphStore((state) => state.reset)
  const toGraph = useGraphStore((state) => state.toGraph)
  const applyDiagnostics = useGraphStore((state) => state.applyDiagnostics)
  const addNode = useGraphStore((state) => state.addNode)
  const graphNodes = useGraphStore((state) => state.nodes)
  const graphEdges = useGraphStore((state) => state.edges)
  const modelConfiguration = useGraphStore((state) => state.modelConfiguration)

  const refreshProjects = async () => setProjects(await api.listProjects())
  useEffect(() => {
    Promise.all([api.catalog(), api.listProjects(), api.metadata()])
      .then(([catalog, projectList, metadata]) => {
        setManifests(catalog)
        setProjects(projectList)
        setBackendStatuses(metadata.execution_backends)
        const available = metadata.execution_backends.filter((item) => item.available)
        setExecutionBackend(available.length === 1 ? available[0].id : null)
      })
      .catch((error) => setNotice(`Backend unavailable: ${String(error)}`))
  }, [setManifests])

  useEffect(() => {
    setSmokeTestState({ status: 'ready' })
  }, [executionBackend, graphNodes, graphEdges, modelConfiguration])

  const runValidation = async () => {
    setBusy(true)
    try {
      const result = await api.validate(toGraph(), codeTarget)
      setValidation(result)
      applyDiagnostics(result.diagnostics, result.shape_summaries)
      setNotice(result.valid ? `Graph is valid for ${codeTarget} code.` : 'Graph has blocking findings.')
      return result
    } catch (error) {
      setNotice(`Validation failed: ${String(error)}`)
      return null
    } finally { setBusy(false) }
  }

  const openSmokeTest = () => {
    setSmokeTestState({ status: 'ready' })
    setSmokeTestOpen(true)
  }

  const runSmokeTest = async () => {
    setSmokeTestOpen(true)
    if (!executionBackend) {
      setSmokeTestState({ status: 'error', error: { code: 'backend_unavailable', message: 'Select an operational execution backend before running the smoke test.' } })
      return
    }
    setSmokeTestState({ status: 'loading' })
    setBusy(true)
    try {
      const result = await api.smokeTest(toGraph(), executionBackend)
      setSmokeTestState({ status: 'succeeded', result })
    } catch (error) {
      setSmokeTestState({ status: 'error', error: smokeTestError(error) })
    } finally { setBusy(false) }
  }

  const save = async () => {
    setBusy(true)
    try {
      const graph = toGraph()
      const project = projectId
        ? await api.updateProject(projectId, { name, description, code_target: codeTarget, graph, expected_revision: revision })
        : await api.createProject({ name, description, code_target: codeTarget, graph })
      setProjectId(project.id)
      setRevision(project.current_revision)
      setNotice(`Saved revision ${project.current_revision}.`)
      await refreshProjects()
    } catch (error) { setNotice(`Save failed: ${String(error)}`) } finally { setBusy(false) }
  }

  const open = async (id: string) => {
    try {
      const project = await api.getProject(id)
      replaceGraph(project)
      setProjectId(project.id); setRevision(project.current_revision); setName(project.name); setDescription(project.description); setCodeTarget(project.code_target); setValidation(null); setCode(null); setSmokeTestOpen(false)
    } catch (error) { setNotice(`Open failed: ${String(error)}`) }
  }

  const newProject = () => {
    reset(); setProjectId(null); setRevision(1); setName('Untitled model'); setDescription(''); setCodeTarget('keras'); setValidation(null); setCode(null); setSmokeTestOpen(false)
  }

  const previewCode = async () => {
    setBusy(true)
    try {
      const result = await api.compile(toGraph(), codeTarget)
      setValidation({ valid: result.valid, diagnostics: result.diagnostics, shape_summaries: [], total_parameters: null })
      if (result.code) setCode(result.code)
      else setNotice('Code generation is blocked by compatibility findings.')
    } catch (error) { setNotice(`Code generation failed: ${String(error)}`) } finally { setBusy(false) }
  }

  const duplicate = async () => {
    if (!projectId) return
    try {
      const project = await api.duplicateProject(projectId, `${name} copy`)
      await refreshProjects(); await open(project.id)
    } catch (error) { setNotice(`Duplicate failed: ${String(error)}`) }
  }

  const exportJson = async () => {
    if (!projectId) return
    try {
      const bundle = await api.exportProject(projectId)
      downloadBlob(new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' }), `${safeFilename(name)}.neuro-weave.json`)
    } catch (error) { setNotice(`JSON export failed: ${String(error)}`) }
  }

  const exportKerasModel = async () => {
    if (!projectId || !executionBackend) return
    setBusy(true)
    try {
      downloadBlob(await api.exportKerasModel(projectId, executionBackend), `${safeFilename(name)}.keras`)
      setNotice(`Keras 3 model exported with ${executionBackend}.`)
    } catch (error) { setNotice(`Keras export failed: ${String(error)}`) } finally { setBusy(false) }
  }

  const importJson = async (file: File) => {
    try {
      const bundle = JSON.parse(await file.text()) as ProjectExportBundle
      const project = await api.importProject(bundle)
      await refreshProjects(); await open(project.id)
    } catch (error) { setNotice(`Import failed: ${String(error)}`) }
  }

  return (
    <div className="app-shell">
      <ProjectToolbar name={name} onNameChange={setName} description={description} onDescriptionChange={setDescription} codeTarget={codeTarget} onCodeTargetChange={setCodeTarget} backendStatuses={backendStatuses} executionBackend={executionBackend} onExecutionBackendChange={setExecutionBackend} projects={projects} currentProjectId={projectId} onNew={newProject} onSave={save} onOpen={open} onDuplicate={duplicate} onValidate={runValidation} onSmokeTestOpen={openSmokeTest} onCode={previewCode} onExport={exportJson} onBinaryExport={exportKerasModel} onImport={() => importInput.current?.click()} busy={busy} />
      <input ref={importInput} type="file" accept="application/json,.json" hidden onChange={(event) => event.target.files?.[0] && importJson(event.target.files[0])} />
      <div className="responsive-rail-tabs" role="tablist" aria-label="Workspace panels">
        <button className={railPanel === 'catalog' ? 'active' : ''} onClick={() => setRailPanel('catalog')}>Catalog</button>
        <button className={railPanel === 'inspector' ? 'active' : ''} onClick={() => setRailPanel('inspector')}>Inspector</button>
      </div>
      <main className="workspace-grid" data-rail-panel={railPanel}>
        <div className={`drawer-slot ${drawer === 'catalog' ? 'is-open' : ''}`}><NodeCatalog manifests={manifests} codeTarget={codeTarget} isOpen={drawer === 'catalog'} onClose={() => setDrawer(null)} onAdd={(manifest) => addNode(manifest, { x: 120 + useGraphStore.getState().nodes.length * 280, y: 120 })} /></div>
        <section className="center-workspace"><ModelConfigurationPanel /><GraphCanvas /><ValidationPanel result={validation} /></section>
        <div className={`drawer-slot ${drawer === 'inspector' ? 'is-open' : ''}`}><PropertiesInspector isOpen={drawer === 'inspector'} onClose={() => setDrawer(null)} /></div>
      </main>
      <div className="small-viewport-actions"><button onClick={() => setDrawer(drawer === 'catalog' ? null : 'catalog')}>Catalog</button><button onClick={() => setDrawer(drawer === 'inspector' ? null : 'inspector')}>Inspector</button></div>
      <p className="viewport-notice">1024×768 recommended for full graph authoring. Smaller screens use constrained drawers.</p>
      {notice && <button className="notice" onClick={() => setNotice(null)}>{notice}</button>}
      {code && <CodePreview code={code} onClose={() => setCode(null)} />}
      {smokeTestOpen && <ModelSmokeTestDrawer backend={executionBackend} backendStatus={backendStatuses.find((item) => item.id === executionBackend)} state={smokeTestState} onRun={runSmokeTest} onClose={() => setSmokeTestOpen(false)} />}
    </div>
  )
}

function safeFilename(value: string): string { return value.replaceAll(/[^a-zA-Z0-9_-]/g, '_') || 'model' }

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

function smokeTestError(error: unknown): SmokeTestErrorDetail {
  if (error instanceof ApiError && typeof error.detail === 'object' && error.detail !== null) {
    const detail = error.detail as Partial<SmokeTestErrorDetail>
    if (typeof detail.code === 'string' && typeof detail.message === 'string') return detail as SmokeTestErrorDetail
  }
  return { code: 'request_failed', message: error instanceof Error ? error.message : String(error) }
}
