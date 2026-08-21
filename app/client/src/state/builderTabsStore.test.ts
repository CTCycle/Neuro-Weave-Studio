import { afterEach, describe, expect, it } from 'vitest'
import type { GraphEditorSession } from './graphStore'
import { useBuilderTabsStore } from './builderTabsStore'
import { emptyModelConfiguration, type ValidationResult } from '../domain/types'

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
    validationExpanded: false,
    validationHeight: 160,
    savedBaseline: label,
  }
}

const validation = (code: string): ValidationResult => ({
  valid: false,
  diagnostics: [{ code, severity: 'error', message: `${code} finding`, details: {} }],
  shape_summaries: [],
  total_parameters: 0,
})

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

  it('keeps validation UI state independent between workspaces', () => {
    const firstId = useBuilderTabsStore.getState().openTab(tab('project-1', 'first'))
    const secondId = useBuilderTabsStore.getState().openTab(tab('project-2', 'second'))
    useBuilderTabsStore.getState().updateTab(firstId, {
      graph: { ...session('first-edited'), selectedNodeId: 'node-first' },
      validation: validation('FIRST'),
      validationExpanded: true,
      validationHeight: 240,
    })
    useBuilderTabsStore.getState().updateTab(secondId, {
      graph: { ...session('second-edited'), selectedNodeId: 'node-second' },
      validation: validation('SECOND'),
    })
    useBuilderTabsStore.getState().focusTab(firstId)

    const first = useBuilderTabsStore.getState().tabs.find((item) => item.id === firstId)
    const second = useBuilderTabsStore.getState().tabs.find((item) => item.id === secondId)
    expect(first?.graph.selectedNodeId).toBe('node-first')
    expect(first?.graph.modelConfiguration.optimizer).toBe('first-edited')
    expect(first?.validation?.diagnostics[0].code).toBe('FIRST')
    expect(first?.validationExpanded).toBe(true)
    expect(first?.validationHeight).toBe(240)
    expect(second?.graph.selectedNodeId).toBe('node-second')
    expect(second?.graph.modelConfiguration.optimizer).toBe('second-edited')
    expect(second?.validation?.diagnostics[0].code).toBe('SECOND')
    expect(second?.validationExpanded).toBe(false)
  })
})
