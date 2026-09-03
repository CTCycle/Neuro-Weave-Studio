import { afterEach, describe, expect, it } from 'vitest'
import { emptyModelConfiguration } from '../../../domain/types'
import type { GraphEditorSession } from '../../../domain/graph-mapper'
import { BuilderTabsStore } from './builder-tabs.store'

const session = (label: string): GraphEditorSession => ({ nodes: [], edges: [], outputBindings: [], modelConfiguration: { ...emptyModelConfiguration(), keras: { ...emptyModelConfiguration().keras, optimizer: label } }, exportSettings: { include_comments: true, include_shape_assertions: true, preferred_binary_format: 'keras' }, futureTrainingConfiguration: {}, selectedNodeId: null, selectedEdgeId: null, copiedNodeIds: [], past: [], future: [] })
function tab(projectId: string | null, label: string) { return { projectId, name: projectId ? `Project ${projectId}` : 'Untitled model', description: '', executionBackend: 'pytorch' as const, kerasEnabled: true, revision: 1, graph: session(label), validation: null, validationExpanded: false, validationHeight: 160, savedBaseline: label } }

describe('builder tabs store', () => {
  let store: BuilderTabsStore
  afterEach(() => store?.reset())

  it('focuses an already-open project instead of duplicating its workspace', () => {
    store = new BuilderTabsStore()
    const firstId = store.openTab(tab('project-1', 'first'))!
    store.openTab(tab(null, 'draft'))
    const duplicateId = store.openTab(tab('project-1', 'duplicate'))
    expect(duplicateId).toBe(firstId)
    expect(store.tabs()).toHaveLength(2)
    expect(store.activeTabId()).toBe(firstId)
  })

  it('isolates graph and validation state between tabs and selects an adjacent tab on close', () => {
    store = new BuilderTabsStore()
    const firstId = store.openTab(tab('project-1', 'first'))!
    const secondId = store.openTab(tab('project-2', 'second'))!
    store.updateTab(firstId, { dirty: true, graph: { ...session('first-edited'), selectedNodeId: 'node-first' }, validation: { valid: false, diagnostics: [{ code: 'FIRST', severity: 'error', message: 'FIRST finding', details: {} }], shape_summaries: [], total_parameters: 0 }, validationExpanded: true, validationHeight: 240 })
    store.updateTab(secondId, { graph: { ...session('second-edited'), selectedNodeId: 'node-second' }, validation: { valid: false, diagnostics: [{ code: 'SECOND', severity: 'error', message: 'SECOND finding', details: {} }], shape_summaries: [], total_parameters: 0 } })
    const first = store.tabs().find((item) => item.id === firstId)
    const second = store.tabs().find((item) => item.id === secondId)
    expect(first?.graph.modelConfiguration.keras.optimizer).toBe('first-edited')
    expect(first?.validation?.diagnostics[0].code).toBe('FIRST')
    expect(first?.validationExpanded).toBe(true)
    expect(first?.validationHeight).toBe(240)
    expect(second?.graph.modelConfiguration.keras.optimizer).toBe('second-edited')
    expect(second?.validation?.diagnostics[0].code).toBe('SECOND')
    expect(second?.validationExpanded).toBe(false)
    store.focusTab(firstId)
    store.closeTab(firstId)
    expect(store.activeTabId()).toBe(secondId)
    expect(store.tabs()).toHaveLength(1)
  })

  it('caps the workspace at eight open model tabs', () => {
    store = new BuilderTabsStore()
    for (let index = 0; index < 8; index += 1) store.openTab(tab(`project-${index}`, `tab-${index}`))

    expect(store.tabs()).toHaveLength(8)
    expect(store.canOpenTab()).toBe(false)
    expect(store.openTab(tab('project-8', 'ninth'))).toBeNull()
    expect(store.tabs()).toHaveLength(8)
  })
})
