import { useEffect, useRef, useState } from 'react'
import { useBeforeUnload, useBlocker, useLocation, useNavigate, useParams } from 'react-router-dom'
import { ApiError, api } from '../api/client'
import { GraphCanvas } from '../components/GraphCanvas'
import { ModelConfigurationPanel } from '../components/ModelConfigurationPanel'
import { ModelTabBar } from '../components/ModelTabBar'
import { NodeCatalog } from '../components/NodeCatalog'
import { ProjectToolbar } from '../components/ProjectToolbar'
import { PropertiesInspector } from '../components/PropertiesInspector'
import { ResizableValidationRegion } from '../components/ResizableValidationRegion'
import { ValidationPanel } from '../components/ValidationPanel'
import type { ExecutionBackend, ProjectDocument } from '../domain/types'
import { useAppResources } from '../state/useAppResources'
import { useBuilderTabsStore, type BuilderTab, type NewBuilderTab } from '../state/builderTabsStore'
import { useGraphStore, type GraphEditorSession } from '../state/graphStore'

function draftSnapshot(name: string, description: string, executionBackend: ExecutionBackend, kerasEnabled: boolean, graph: unknown): string {
  return JSON.stringify({ name, description, executionBackend, kerasEnabled, graph })
}

function preferredBackend(statuses: { id: ExecutionBackend; available: boolean }[]): ExecutionBackend {
  if (statuses.some((item) => item.id === 'pytorch' && item.available)) return 'pytorch'
  return statuses.find((item) => item.available)?.id ?? 'pytorch'
}

function projectTabInput(project: ProjectDocument, graph: GraphEditorSession): NewBuilderTab {
  return {
    projectId: project.id,
    name: project.name,
    description: project.description,
    executionBackend: project.execution_backend,
    kerasEnabled: project.keras_enabled,
    revision: project.current_revision,
    graph,
    validation: null,
    savedBaseline: draftSnapshot(project.name, project.description, project.execution_backend, project.keras_enabled, project.graph),
  }
}

function untitledTabInput(executionBackend: ExecutionBackend, graph: GraphEditorSession, modelGraph: unknown): NewBuilderTab {
  return {
    projectId: null,
    name: 'Untitled model',
    description: '',
    executionBackend,
    kerasEnabled: true,
    revision: 1,
    graph,
    validation: null,
    savedBaseline: draftSnapshot('Untitled model', '', executionBackend, true, modelGraph),
  }
}

type DraftMetadataPatch = Partial<Pick<BuilderTab, 'name' | 'description' | 'executionBackend' | 'kerasEnabled'>>

export function ModelBuilderPage() {
  const { modelId } = useParams<{ modelId: string }>()
  const location = useLocation()
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
  const exportSettings = useGraphStore((state) => state.exportSettings)
  const futureTrainingConfiguration = useGraphStore((state) => state.futureTrainingConfiguration)
  const selectedNodeId = useGraphStore((state) => state.selectedNodeId)
  const selectedEdgeId = useGraphStore((state) => state.selectedEdgeId)
  const copiedNodeIds = useGraphStore((state) => state.copiedNodeIds)
  const past = useGraphStore((state) => state.past)
  const future = useGraphStore((state) => state.future)
  const restoreEditorSession = useGraphStore((state) => state.restoreEditorSession)
  const tabs = useBuilderTabsStore((state) => state.tabs)
  const activeTabId = useBuilderTabsStore((state) => state.activeTabId)
  const openTab = useBuilderTabsStore((state) => state.openTab)
  const focusTab = useBuilderTabsStore((state) => state.focusTab)
  const updateTab = useBuilderTabsStore((state) => state.updateTab)
  const closeTab = useBuilderTabsStore((state) => state.closeTab)
  const activeTab = tabs.find((tab) => tab.id === activeTabId) ?? null
  const dirtyTabCount = tabs.filter((tab) => tab.dirty).length
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [mobilePanel, setMobilePanel] = useState<'catalog' | 'inspector' | null>(null)
  const [pendingCloseTabId, setPendingCloseTabId] = useState<string | null>(null)
  const [initialized, setInitialized] = useState(false)
  const allowNextNavigation = useRef(false)
  const initializedKey = useRef<string | null>(null)

  const blocker = useBlocker(() => {
    if (allowNextNavigation.current) {
      allowNextNavigation.current = false
      return false
    }
    return dirtyTabCount > 0
  })
  useBeforeUnload((event) => {
    if (!dirtyTabCount) return
    event.preventDefault()
    event.returnValue = ''
  })

  useEffect(() => {
    if (manifests.length) setManifests(manifests)
  }, [manifests, setManifests])

  useEffect(() => {
    if (!initialized || !activeTabId) return
    const current = useBuilderTabsStore.getState().tabs.find((tab) => tab.id === activeTabId)
    if (!current) return
    const graphState = useGraphStore.getState()
    updateTab(activeTabId, {
      graph: graphState.captureEditorSession(),
      dirty: current.savedBaseline !== draftSnapshot(current.name, current.description, current.executionBackend, current.kerasEnabled, graphState.toGraph()),
    })
  }, [activeTabId, copiedNodeIds, edges, exportSettings, future, futureTrainingConfiguration, initialized, modelConfiguration, nodes, past, selectedEdgeId, selectedNodeId, updateTab])

  useEffect(() => {
    if (loading) return
    const routeKey = modelId ? `project:${modelId}` : 'new'
    if (initializedKey.current === routeKey) return
    initializedKey.current = routeKey
    allowNextNavigation.current = false
    setInitialized(false)
    setNotice(null)
    const tabsState = useBuilderTabsStore.getState()
    const existing = modelId
      ? tabsState.tabs.find((tab) => tab.projectId === modelId)
      : tabsState.activeTabId ? tabsState.tabs.find((tab) => tab.id === tabsState.activeTabId && tab.projectId === null) : null
    if (existing) {
      focusTab(existing.id)
      restoreEditorSession(existing.graph)
      setInitialized(true)
      return
    }
    if (!modelId) {
      reset()
      const graph = useGraphStore.getState().captureEditorSession()
      openTab(untitledTabInput(preferredBackend(backendStatuses), graph, useGraphStore.getState().toGraph()))
      setInitialized(true)
      return
    }
    let cancelled = false
    api.getProject(modelId).then((project) => {
      if (cancelled) return
      replaceGraph(project)
      const graph = useGraphStore.getState().captureEditorSession()
      openTab(projectTabInput(project, graph))
      setInitialized(true)
    }).catch((error) => {
      if (cancelled) return
      setNotice(`Unable to open model: ${messageFor(error)}`)
      setInitialized(true)
    })
    return () => { cancelled = true }
  }, [backendStatuses, focusTab, loading, modelId, openTab, replaceGraph, reset, restoreEditorSession])

  const updateDraftMetadata = (patch: DraftMetadataPatch) => {
    if (!activeTabId) return
    const current = useBuilderTabsStore.getState().tabs.find((tab) => tab.id === activeTabId)
    if (!current) return
    const next = { ...current, ...patch }
    const graphState = useGraphStore.getState()
    updateTab(activeTabId, {
      ...patch,
      dirty: current.savedBaseline !== draftSnapshot(next.name, next.description, next.executionBackend, next.kerasEnabled, graphState.toGraph()),
    })
  }

  const syncActiveTab = () => {
    if (!activeTabId) return
    const current = useBuilderTabsStore.getState().tabs.find((tab) => tab.id === activeTabId)
    if (!current) return
    const graphState = useGraphStore.getState()
    updateTab(activeTabId, {
      graph: graphState.captureEditorSession(),
      dirty: current.savedBaseline !== draftSnapshot(current.name, current.description, current.executionBackend, current.kerasEnabled, graphState.toGraph()),
    })
  }

  const openNewTab = () => {
    syncActiveTab()
    reset()
    const graph = useGraphStore.getState().captureEditorSession()
    openTab(untitledTabInput(preferredBackend(backendStatuses), graph, useGraphStore.getState().toGraph()))
    restoreEditorSession(graph)
    initializedKey.current = 'new'
    setInitialized(true)
    if (location.pathname !== '/builder') {
      allowNextNavigation.current = true
      navigate('/builder', { replace: true })
    }
  }

  const switchTab = (tabId: string) => {
    if (tabId === activeTabId) return
    syncActiveTab()
    const target = useBuilderTabsStore.getState().tabs.find((tab) => tab.id === tabId)
    if (!target) return
    focusTab(tabId)
    restoreEditorSession(target.graph)
    initializedKey.current = target.projectId ? `project:${target.projectId}` : 'new'
    setInitialized(true)
    const nextPath = target.projectId ? `/builder/${target.projectId}` : '/builder'
    if (location.pathname !== nextPath) {
      allowNextNavigation.current = true
      navigate(nextPath, { replace: true })
    }
  }

  const closeTabAndNavigate = (tabId: string) => {
    const target = useBuilderTabsStore.getState().tabs.find((tab) => tab.id === tabId)
    if (!target) return
    const wasActive = target.id === activeTabId
    if (wasActive) syncActiveTab()
    closeTab(tabId)
    if (!wasActive) return
    const state = useBuilderTabsStore.getState()
    const next = state.activeTabId ? state.tabs.find((tab) => tab.id === state.activeTabId) : null
    if (!next) {
      openNewTab()
      return
    }
    restoreEditorSession(next.graph)
    initializedKey.current = next.projectId ? `project:${next.projectId}` : 'new'
    setInitialized(true)
    const nextPath = next.projectId ? `/builder/${next.projectId}` : '/builder'
    if (location.pathname !== nextPath) {
      allowNextNavigation.current = true
      navigate(nextPath, { replace: true })
    }
  }

  const requestCloseTab = (tabId: string) => {
    const target = useBuilderTabsStore.getState().tabs.find((tab) => tab.id === tabId)
    if (!target) return
    if (target.dirty) setPendingCloseTabId(tabId)
    else closeTabAndNavigate(tabId)
  }

  const runValidation = async () => {
    if (!activeTab || !activeTabId) return
    setBusy(true)
    try {
      const result = await api.validate(toGraph(), activeTab.executionBackend, activeTab.kerasEnabled)
      updateTab(activeTabId, { validation: result })
      applyShapeSummaries(result.shape_summaries)
      setNotice(result.valid ? 'Model execution profile is valid.' : 'Model has blocking findings.')
    } catch (error) {
      setNotice(`Validation failed: ${messageFor(error)}`)
    } finally { setBusy(false) }
  }

  const persist = async (compile: boolean) => {
    if (!activeTab || !activeTabId) return
    const tabId = activeTabId
    const current = useBuilderTabsStore.getState().tabs.find((tab) => tab.id === tabId)
    if (!current) return
    setBusy(true)
    try {
      const graph = useGraphStore.getState().toGraph()
      const payload = { name: current.name, description: current.description, execution_backend: current.executionBackend, keras_enabled: current.kerasEnabled, graph }
      const project = current.projectId
        ? await api.updateProject(current.projectId, { ...payload, expected_revision: current.revision })
        : await api.createProject(payload)
      const savedGraph = useGraphStore.getState().captureEditorSession()
      updateTab(tabId, {
        projectId: project.id,
        name: project.name,
        description: project.description,
        executionBackend: project.execution_backend,
        kerasEnabled: project.keras_enabled,
        revision: project.current_revision,
        graph: savedGraph,
        savedBaseline: draftSnapshot(project.name, project.description, project.execution_backend, project.keras_enabled, graph),
        dirty: false,
      })
      await refreshProjects()
      if (compile) {
        await api.sourceForProject(project.id)
        await refreshProjects()
        setNotice(`Saved revision ${project.current_revision} and compiled source for the current execution profile.`)
      } else {
        setNotice(`Saved revision ${project.current_revision}.`)
      }
      if (!current.projectId) {
        allowNextNavigation.current = true
        navigate(`/builder/${project.id}`, { replace: true })
      }
    } catch (error) {
      setNotice(`${compile ? 'Save and compile' : 'Save'} failed: ${messageFor(error)}`)
    } finally { setBusy(false) }
  }

  if ((loading && !initialized) || !activeTab) return <section className="page-state panel"><span className="eyebrow">MODEL BUILDER</span><h1>{notice ?? 'Loading authoring workspace…'}</h1></section>

  return (
    <div className="builder-page">
      <ProjectToolbar name={activeTab.name} onNameChange={(name) => updateDraftMetadata({ name })} description={activeTab.description} onDescriptionChange={(description) => updateDraftMetadata({ description })} executionBackend={activeTab.executionBackend} onExecutionBackendChange={(executionBackend) => updateDraftMetadata({ executionBackend })} kerasEnabled={activeTab.kerasEnabled} onKerasEnabledChange={(kerasEnabled) => updateDraftMetadata({ kerasEnabled })} onSave={() => persist(false)} onCompile={() => persist(true)} onValidate={runValidation} busy={busy} modelId={activeTab.projectId} onOpenWorkspace={() => activeTab.projectId && navigate(`/models/${activeTab.projectId}`)} />
      <ModelTabBar tabs={tabs} activeTabId={activeTabId} onSelect={switchTab} onClose={requestCloseTab} onNew={openNewTab} />
      <div className="builder-mobile-actions"><button className={mobilePanel === 'catalog' ? 'active' : ''} onClick={() => setMobilePanel(mobilePanel === 'catalog' ? null : 'catalog')}>Catalog</button><button className={mobilePanel === 'inspector' ? 'active' : ''} onClick={() => setMobilePanel(mobilePanel === 'inspector' ? null : 'inspector')}>Inspector</button></div>
      <div className="builder-workspace-grid">
        <div className={`drawer-slot builder-catalog ${mobilePanel === 'catalog' ? 'is-open' : ''}`}><NodeCatalog manifests={manifests} executionBackend={activeTab.executionBackend} kerasEnabled={activeTab.kerasEnabled} isOpen={mobilePanel === 'catalog'} onClose={() => setMobilePanel(null)} onAdd={(manifest) => addNode(manifest, { x: 120 + useGraphStore.getState().nodes.length * 280, y: 120 })} /></div>
        <section className="builder-center-workspace"><ModelConfigurationPanel kerasEnabled={activeTab.kerasEnabled} /><GraphCanvas /><ResizableValidationRegion><ValidationPanel result={activeTab.validation} /></ResizableValidationRegion></section>
        <div className={`drawer-slot builder-inspector ${mobilePanel === 'inspector' ? 'is-open' : ''}`}><PropertiesInspector onClose={() => setMobilePanel(null)} isOpen={mobilePanel === 'inspector'} executionBackend={activeTab.executionBackend} kerasEnabled={activeTab.kerasEnabled} /></div>
      </div>
      {blocker.state === 'blocked' && <div className="dialog-backdrop"><section className="confirm-dialog panel" role="dialog" aria-modal="true" aria-labelledby="unsaved-title"><span className="eyebrow">UNSAVED CHANGES</span><h2 id="unsaved-title">Leave the Model Builder?</h2><p>{dirtyTabCount === 1 ? 'One model tab has unsaved changes.' : `${dirtyTabCount} model tabs have unsaved changes.`} Leave and discard them?</p><div className="dialog-actions"><button className="secondary" onClick={() => blocker.reset?.()}>Stay</button><button className="primary" onClick={() => blocker.proceed?.()}>Discard changes</button></div></section></div>}
      {pendingCloseTabId && <div className="dialog-backdrop"><section className="confirm-dialog panel" role="dialog" aria-modal="true" aria-labelledby="close-tab-title"><span className="eyebrow">UNSAVED TAB</span><h2 id="close-tab-title">Close this model tab?</h2><p>Its architecture edits have not been saved. Close and discard them?</p><div className="dialog-actions"><button className="secondary" onClick={() => setPendingCloseTabId(null)}>Keep tab</button><button className="primary" onClick={() => { const tabId = pendingCloseTabId; setPendingCloseTabId(null); closeTabAndNavigate(tabId) }}>Close tab</button></div></section></div>}
      {notice && <button className="notice builder-notice" onClick={() => setNotice(null)}>{notice}</button>}
    </div>
  )
}

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return error instanceof Error ? error.message : String(error)
}
