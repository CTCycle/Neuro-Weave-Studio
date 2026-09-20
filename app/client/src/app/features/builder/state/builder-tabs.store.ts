import { Injectable, computed, signal } from '@angular/core'
import type { ExecutionBackend, ValidationResult } from '../../../domain/types'
import type { GraphEditorSession } from '../../../domain/graph-mapper'

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
  validationExpanded: boolean
  validationHeight: number
  savedBaseline: string
  dirty: boolean
}

export type NewBuilderTab = Omit<BuilderTab, 'id' | 'dirty'> & {
  id?: string
  dirty?: boolean
}

export const MAX_OPEN_BUILDER_TABS = 8

export function shouldReuseProjectTab(tab: Pick<BuilderTab, 'dirty' | 'revision'>, currentRevision: number): boolean {
  return tab.dirty || tab.revision === currentRevision
}

@Injectable({ providedIn: 'root' })
export class BuilderTabsStore {
  readonly tabs = signal<BuilderTab[]>([])
  readonly activeTabId = signal<string | null>(null)
  readonly canOpenTab = computed(() => this.tabs().length < MAX_OPEN_BUILDER_TABS)

  openTab(input: NewBuilderTab): string | null {
    const existing = input.projectId ? this.tabs().find((tab) => tab.projectId === input.projectId) : undefined
    if (existing) {
      this.activeTabId.set(existing.id)
      return existing.id
    }
    if (!this.canOpenTab()) return null
    const tab: BuilderTab = {
      ...input,
      id: input.id ?? `builder-tab-${crypto.randomUUID()}`,
      dirty: input.dirty ?? false,
      graph: structuredClone(input.graph),
      validation: input.validation ? structuredClone(input.validation) : null,
    }
    this.tabs.update((tabs) => [...tabs, tab])
    this.activeTabId.set(tab.id)
    return tab.id
  }

  focusTab(tabId: string): void {
    if (this.tabs().some((tab) => tab.id === tabId)) this.activeTabId.set(tabId)
  }

  updateTab(tabId: string, patch: Partial<Omit<BuilderTab, 'id'>>): void {
    this.tabs.update((tabs) => tabs.map((tab) => tab.id === tabId ? cloneTab({ ...tab, ...patch }) : tab))
  }

  closeTab(tabId: string): void {
    const tabs = this.tabs()
    const index = tabs.findIndex((tab) => tab.id === tabId)
    if (index < 0) return
    const nextTabs = tabs.filter((tab) => tab.id !== tabId)
    this.tabs.set(nextTabs)
    if (this.activeTabId() === tabId) this.activeTabId.set(nextTabs[Math.max(0, index - 1)]?.id ?? nextTabs[index]?.id ?? null)
  }

  reset(): void {
    this.tabs.set([])
    this.activeTabId.set(null)
  }
}

function cloneTab(tab: BuilderTab): BuilderTab {
  return {
    ...tab,
    graph: structuredClone(tab.graph),
    validation: tab.validation ? structuredClone(tab.validation) : null,
  }
}
