import { afterEach, describe, expect, it } from 'vitest'
import type { GraphEditorSession } from './graphStore'
import { useBuilderTabsStore } from './builderTabsStore'
import { emptyModelConfiguration } from '../domain/types'

const session = (label: string): GraphEditorSession => ({
  nodes: [],
  edges: [],
  modelConfiguration: { ...emptyModelConfiguration(), optimizer: label },
  exportSettings: { include_comments: true, include_shape_assertions: true, preferred_binary_format: 'keras' },
  futureTrainingConfiguration: {},
  selectedNodeId: null,
  selectedEdgeId: null,
  copiedNodeIds: [],
  past: [],
  future: [],
})
function tab(projectId: string | null, label: string) {
  return {
    projectId,
    name: projectId ? `Project ${projectId}` : 'Untitled model',
    description: '',
    executionBackend: 'pytorch' as const,
    kerasEnabled: true,
    revision: 1,
    graph: session(label),
    validation: null,
    savedBaseline: label,
  }
}

describe('builder tabs store', () => {
  afterEach(() => useBuilderTabsStore.getState().reset())

  it('focuses an already-open project instead of duplicating it', () => {
    const firstId = useBuilderTabsStore.getState().openTab(tab('project-1', 'first'))
    const untitledId = useBuilderTabsStore.getState().openTab(tab(null, 'draft'))
    const duplicateId = useBuilderTabsStore.getState().openTab(tab('project-1', 'duplicate'))

    expect(duplicateId).toBe(firstId)
    expect(useBuilderTabsStore.getState().tabs).toHaveLength(2)
    expect(useBuilderTabsStore.getState().activeTabId).toBe(firstId)
    expect(useBuilderTabsStore.getState().tabs.find((item) => item.id === untitledId)?.projectId).toBeNull()
  })

  it('keeps tab sessions independent and selects an adjacent tab after close', () => {
    const firstId = useBuilderTabsStore.getState().openTab(tab('project-1', 'first'))
    const secondId = useBuilderTabsStore.getState().openTab(tab('project-2', 'second'))
    useBuilderTabsStore.getState().updateTab(firstId, { dirty: true, graph: session('edited-first') })
    useBuilderTabsStore.getState().focusTab(firstId)
    useBuilderTabsStore.getState().closeTab(firstId)

    expect(useBuilderTabsStore.getState().activeTabId).toBe(secondId)
    expect(useBuilderTabsStore.getState().tabs[0].graph.modelConfiguration.optimizer).toBe('second')
  })
})
