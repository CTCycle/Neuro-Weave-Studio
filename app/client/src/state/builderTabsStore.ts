import { create } from 'zustand'
import type { ExecutionBackend, ValidationResult } from '../domain/types'
import type { GraphEditorSession } from './graphStore'

export interface BuilderTab {
  id: string
  projectId: string | null
  name: string
  description: string
  executionBackend: ExecutionBackend
  kerasEnabled: boolean
  revision: number
  graph: GraphEditorSession
  validation: ValidationResult | null
  savedBaseline: string
  dirty: boolean
}

export type NewBuilderTab = Omit<BuilderTab, 'id' | 'dirty'> & {
  id?: string
  dirty?: boolean
}

interface BuilderTabsState {
  tabs: BuilderTab[]
  activeTabId: string | null
  openTab: (tab: NewBuilderTab) => string
  focusTab: (tabId: string) => void
  updateTab: (tabId: string, patch: Partial<Omit<BuilderTab, 'id'>>) => void
  closeTab: (tabId: string) => void
  reset: () => void
}

const newTabId = () => `builder-tab-${crypto.randomUUID()}`

function cloneTab(tab: BuilderTab): BuilderTab {
  return {
    ...tab,
    graph: structuredClone(tab.graph),
    validation: tab.validation ? structuredClone(tab.validation) : null,
  }
}

export const useBuilderTabsStore = create<BuilderTabsState>((set, get) => ({
  tabs: [],
  activeTabId: null,
  openTab: (input) => {
    const existing = input.projectId
      ? get().tabs.find((tab) => tab.projectId === input.projectId)
      : null
    if (existing) {
      set({ activeTabId: existing.id })
      return existing.id
    }
    const tab: BuilderTab = {
      ...input,
      id: input.id ?? newTabId(),
      dirty: input.dirty ?? false,
      graph: structuredClone(input.graph),
      validation: input.validation ? structuredClone(input.validation) : null,
    }
    set((state) => ({ tabs: [...state.tabs, tab], activeTabId: tab.id }))
    return tab.id
  },
  focusTab: (tabId) => set((state) => state.tabs.some((tab) => tab.id === tabId) ? { activeTabId: tabId } : state),
  updateTab: (tabId, patch) => set((state) => ({
    tabs: state.tabs.map((tab) => tab.id === tabId ? cloneTab({ ...tab, ...patch }) : tab),
  })),
  closeTab: (tabId) => set((state) => {
    const index = state.tabs.findIndex((tab) => tab.id === tabId)
    if (index < 0) return state
    const tabs = state.tabs.filter((tab) => tab.id !== tabId)
    if (state.activeTabId !== tabId) return { tabs }
    const nextTab = tabs[Math.max(0, index - 1)] ?? tabs[index] ?? null
    return { tabs, activeTabId: nextTab?.id ?? null }
  }),
  reset: () => set({ tabs: [], activeTabId: null }),
}))
