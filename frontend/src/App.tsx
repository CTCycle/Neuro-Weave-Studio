import { useEffect, useRef, useState } from 'react'
import { api } from './api/client'
import { CodePreview } from './components/CodePreview'
import { GraphCanvas } from './components/GraphCanvas'
import { ModelConfigurationPanel } from './components/ModelConfigurationPanel'
import { NodeCatalog } from './components/NodeCatalog'
import { ProjectToolbar } from './components/ProjectToolbar'
import { PropertiesInspector } from './components/PropertiesInspector'
import { ValidationPanel } from './components/ValidationPanel'
import type { ProjectExportBundle, ProjectSummary, TargetFramework, ValidationResult } from './domain/types'
import { useGraphStore } from './state/graphStore'

export default function App() {
  const [projectId, setProjectId] = useState<string | null>(null)
  const [revision, setRevision] = useState(1)
  const [name, setName] = useState('Untitled model')
  const [description, setDescription] = useState('')
  const [framework, setFramework] = useState<TargetFramework>('keras')
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [validation, setValidation] = useState<ValidationResult | null>(null)
  const [code, setCode] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const importInput = useRef<HTMLInputElement>(null)
  const manifests = useGraphStore((state) => state.manifests)
  const setManifests = useGraphStore((state) => state.setManifests)
  const replaceGraph = useGraphStore((state) => state.replaceGraph)
  const reset = useGraphStore((state) => state.reset)
  const toGraph = useGraphStore((state) => state.toGraph)
  const applyDiagnostics = useGraphStore((state) => state.applyDiagnostics)

  const refreshProjects = async () => setProjects(await api.listProjects())
  useEffect(() => {
    Promise.all([api.catalog(), api.listProjects()])
      .then(([catalog, projectList]) => { setManifests(catalog); setProjects(projectList) })
      .catch((error) => setNotice(`Backend unavailable: ${String(error)}`))
  }, [setManifests])

  const runValidation = async () => {
    setBusy(true)
    try {
      const result = await api.validate(toGraph(), framework)
      setValidation(result)
      applyDiagnostics(result.diagnostics, result.shape_summaries)
      setNotice(result.valid ? 'Graph is valid for the selected framework.' : 'Graph has blocking findings.')
      return result
    } catch (error) {
      setNotice(`Validation failed: ${String(error)}`)
      return null
    } finally { setBusy(false) }
  }

  const save = async () => {
    setBusy(true)
    try {
      const graph = toGraph()
      const project = projectId
        ? await api.updateProject(projectId, { name, description, target_framework: framework, graph, expected_revision: revision })
        : await api.createProject({ name, description, target_framework: framework, graph })
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
      setProjectId(project.id); setRevision(project.current_revision); setName(project.name); setDescription(project.description); setFramework(project.target_framework); setValidation(null); setCode(null)
    } catch (error) { setNotice(`Open failed: ${String(error)}`) }
  }

  const newProject = () => {
    reset(); setProjectId(null); setRevision(1); setName('Untitled model'); setDescription(''); setFramework('keras'); setValidation(null); setCode(null)
  }

  const previewCode = async () => {
    setBusy(true)
    try {
      const result = await api.compile(toGraph(), framework)
      setValidation({ valid: result.valid, diagnostics: result.diagnostics, shape_summaries: [], total_parameters: null })
      if (result.code) setCode(result.code)
      else setNotice('Code generation is blocked by validation findings.')
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
    if (!projectId) return
    setBusy(true)
    try {
      downloadBlob(await api.exportKerasModel(projectId), `${safeFilename(name)}.keras`)
      setNotice('Uninitialized Keras model exported.')
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
      <ProjectToolbar name={name} onNameChange={setName} description={description} onDescriptionChange={setDescription} framework={framework} onFrameworkChange={setFramework} projects={projects} currentProjectId={projectId} onNew={newProject} onSave={save} onOpen={open} onDuplicate={duplicate} onValidate={runValidation} onCode={previewCode} onExport={exportJson} onBinaryExport={exportKerasModel} onImport={() => importInput.current?.click()} busy={busy} />
      <input ref={importInput} type="file" accept="application/json,.json" hidden onChange={(event) => event.target.files?.[0] && importJson(event.target.files[0])} />
      <main className="workspace-grid">
        <NodeCatalog manifests={manifests} framework={framework} />
        <section className="center-workspace"><ModelConfigurationPanel /><GraphCanvas /><ValidationPanel result={validation} /></section>
        <PropertiesInspector />
      </main>
      {notice && <button className="notice" onClick={() => setNotice(null)}>{notice}</button>}
      {code && <CodePreview code={code} onClose={() => setCode(null)} />}
    </div>
  )
}

function safeFilename(value: string): string {
  return value.replaceAll(/[^a-zA-Z0-9_-]/g, '_') || 'model'
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}
