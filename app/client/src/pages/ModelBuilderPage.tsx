import { useEffect, useMemo, useRef, useState } from 'react'
import { useBeforeUnload, useBlocker, useNavigate, useParams } from 'react-router-dom'
import { ApiError, api } from '../api/client'
import { GraphCanvas } from '../components/GraphCanvas'
import { ModelConfigurationPanel } from '../components/ModelConfigurationPanel'
import { NodeCatalog } from '../components/NodeCatalog'
import { ProjectToolbar } from '../components/ProjectToolbar'
import { PropertiesInspector } from '../components/PropertiesInspector'
import { ValidationPanel } from '../components/ValidationPanel'
import type { ExecutionBackend, ValidationResult } from '../domain/types'
import { useAppResources } from '../state/useAppResources'
import { useGraphStore } from '../state/graphStore'

function snapshot(name: string, description: string, executionBackend: ExecutionBackend, kerasEnabled: boolean, graph: unknown): string {
  return JSON.stringify({ name, description, executionBackend, kerasEnabled, graph })
}

function preferredBackend(statuses: { id: ExecutionBackend; available: boolean }[]): ExecutionBackend {
  if (statuses.some((item) => item.id === 'pytorch' && item.available)) return 'pytorch'
  return statuses.find((item) => item.available)?.id ?? 'pytorch'
}

export function ModelBuilderPage() {
  const { modelId } = useParams<{ modelId: string }>()
  const navigate = useNavigate()
  const { manifests, loading, refreshProjects, backendStatuses } = useAppResources()
  const setManifests = useGraphStore((state) => state.setManifests)
  const replaceGraph = useGraphStore((state) => state.replaceGraph)
  const reset = useGraphStore((state) => state.reset)
  const toGraph = useGraphStore((state) => state.toGraph)
  const applyShapeSummaries = useGraphStore((state) => state.applyShapeSummaries)
  const addNode = useGraphStore((state) => state.addNode)
  const nodes = useGraphStore((state) => state.nodes)
  const edges = useGraphStore((state) => state.edges)
  const modelConfiguration = useGraphStore((state) => state.modelConfiguration)
  const [name, setName] = useState('Untitled model')
  const [description, setDescription] = useState('')
  const [executionBackend, setExecutionBackend] = useState<ExecutionBackend>('pytorch')
  const [kerasEnabled, setKerasEnabled] = useState(true)
  const [revision, setRevision] = useState(1)
  const [validation, setValidation] = useState<ValidationResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [mobilePanel, setMobilePanel] = useState<'catalog' | 'inspector' | null>(null)
  const allowNextNavigation = useRef(false)
  const [initialized, setInitialized] = useState(false)
  const initializedKey = useRef<string | null>(null)
  const baseline = useRef('')
  const draftKey = useMemo(() => {
    void nodes
    void edges
    void modelConfiguration
    return snapshot(name, description, executionBackend, kerasEnabled, useGraphStore.getState().toGraph())
  }, [description, edges, executionBackend, kerasEnabled, modelConfiguration, name, nodes])
  const dirty = initialized && baseline.current !== draftKey
  const blocker = useBlocker(() => {
    if (allowNextNavigation.current) {
      allowNextNavigation.current = false
      return false
    }
    return dirty
  })
  useBeforeUnload((event) => {
    if (!dirty) return
    event.preventDefault()
    event.returnValue = ''
  })

  useEffect(() => {
    if (manifests.length) setManifests(manifests)
  }, [manifests, setManifests])

  useEffect(() => {
    if (loading) return
    const key = modelId ?? 'new'
    if (initializedKey.current === key) return
    initializedKey.current = key
    setInitialized(false)
    setValidation(null)
    setNotice(null)
    if (!modelId) {
      reset()
      setName('Untitled model')
      setDescription('')
      const backend = preferredBackend(backendStatuses)
      setExecutionBackend(backend)
      setKerasEnabled(true)
      setRevision(1)
      const nextGraph = useGraphStore.getState().toGraph()
      baseline.current = snapshot('Untitled model', '', backend, true, nextGraph)
      setInitialized(true)
      return
    }
    api.getProject(modelId).then((project) => {
      replaceGraph(project)
      setName(project.name)
      setDescription(project.description)
      setExecutionBackend(project.execution_backend)
      setKerasEnabled(project.keras_enabled)
      setRevision(project.current_revision)
      baseline.current = snapshot(project.name, project.description, project.execution_backend, project.keras_enabled, useGraphStore.getState().toGraph())
      setInitialized(true)
    }).catch((error) => {
      setNotice(`Unable to open model: ${messageFor(error)}`)
      setInitialized(true)
    })
  }, [backendStatuses, loading, modelId, replaceGraph, reset])

  const runValidation = async () => {
    setBusy(true)
    try {
      const result = await api.validate(toGraph(), executionBackend, kerasEnabled)
      setValidation(result)
      applyShapeSummaries(result.shape_summaries)
      setNotice(result.valid ? 'Model execution profile is valid.' : 'Model has blocking findings.')
    } catch (error) {
      setNotice(`Validation failed: ${messageFor(error)}`)
    } finally { setBusy(false) }
  }

  const persist = async (compile: boolean) => {
    setBusy(true)
    try {
      const payload = { name, description, execution_backend: executionBackend, keras_enabled: kerasEnabled, graph: toGraph() }
      const project = modelId
        ? await api.updateProject(modelId, { ...payload, expected_revision: revision })
        : await api.createProject(payload)
      baseline.current = snapshot(project.name, project.description, project.execution_backend, project.keras_enabled, useGraphStore.getState().toGraph())
      setName(project.name)
      setDescription(project.description)
      setExecutionBackend(project.execution_backend)
      setKerasEnabled(project.keras_enabled)
      setRevision(project.current_revision)
      await refreshProjects()
      if (compile) {
        await api.sourceForProject(project.id)
        await refreshProjects()
        setNotice(`Saved revision ${project.current_revision} and compiled source for the current execution profile.`)
      } else {
        setNotice(`Saved revision ${project.current_revision}.`)
      }
      if (!modelId) {
        allowNextNavigation.current = true
        navigate(`/builder/${project.id}`, { replace: true })
      }
    } catch (error) {
      setNotice(`${compile ? 'Save and compile' : 'Save'} failed: ${messageFor(error)}`)
    } finally { setBusy(false) }
  }

  if (loading && !initialized) return <section className="page-state panel"><span className="eyebrow">MODEL BUILDER</span><h1>Loading authoring workspace…</h1></section>

  return (
    <div className="builder-page">
      <ProjectToolbar name={name} onNameChange={setName} description={description} onDescriptionChange={setDescription} executionBackend={executionBackend} onExecutionBackendChange={setExecutionBackend} kerasEnabled={kerasEnabled} onKerasEnabledChange={setKerasEnabled} onSave={() => persist(false)} onCompile={() => persist(true)} onValidate={runValidation} busy={busy} modelId={modelId ?? null} onOpenWorkspace={() => modelId && navigate(`/models/${modelId}`)} />
      <div className="builder-mobile-actions"><button className={mobilePanel === 'catalog' ? 'active' : ''} onClick={() => setMobilePanel(mobilePanel === 'catalog' ? null : 'catalog')}>Catalog</button><button className={mobilePanel === 'inspector' ? 'active' : ''} onClick={() => setMobilePanel(mobilePanel === 'inspector' ? null : 'inspector')}>Inspector</button></div>
      <div className="builder-workspace-grid">
        <div className={`drawer-slot builder-catalog ${mobilePanel === 'catalog' ? 'is-open' : ''}`}><NodeCatalog manifests={manifests} executionBackend={executionBackend} kerasEnabled={kerasEnabled} isOpen={mobilePanel === 'catalog'} onClose={() => setMobilePanel(null)} onAdd={(manifest) => addNode(manifest, { x: 120 + useGraphStore.getState().nodes.length * 280, y: 120 })} /></div>
        <section className="builder-center-workspace"><ModelConfigurationPanel /><GraphCanvas /><ValidationPanel result={validation} /></section>
        <div className={`drawer-slot builder-inspector ${mobilePanel === 'inspector' ? 'is-open' : ''}`}><PropertiesInspector onClose={() => setMobilePanel(null)} isOpen={mobilePanel === 'inspector'} executionBackend={executionBackend} kerasEnabled={kerasEnabled} /></div>
      </div>
      {blocker.state === 'blocked' && <div className="dialog-backdrop"><section className="confirm-dialog panel" role="dialog" aria-modal="true" aria-labelledby="unsaved-title"><span className="eyebrow">UNSAVED CHANGES</span><h2 id="unsaved-title">Leave the Model Builder?</h2><p>Your architecture edits have not been saved. Leave and discard them?</p><div className="dialog-actions"><button className="secondary" onClick={() => blocker.reset?.()}>Stay</button><button className="primary" onClick={() => blocker.proceed?.()}>Discard changes</button></div></section></div>}
      {notice && <button className="notice builder-notice" onClick={() => setNotice(null)}>{notice}</button>}
    </div>
  )
}

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return error instanceof Error ? error.message : String(error)
}
