import { Plus, X } from 'lucide-react'
import type { BuilderTab } from '../state/builderTabsStore'

interface Props {
  tabs: BuilderTab[]
  activeTabId: string | null
  onSelect: (tabId: string) => void
  onClose: (tabId: string) => void
  onNew: () => void
}
export function ModelTabBar({ tabs, activeTabId, onSelect, onClose, onNew }: Props) {
  return (
    <nav className="model-tab-bar" aria-label="Open models">
      <div className="model-tab-list" role="tablist" aria-label="Model editor tabs">
        {tabs.map((tab) => {
          const label = tab.name.trim() || 'Untitled model'
          return (
            <div key={tab.id} className={`model-tab ${tab.id === activeTabId ? 'active' : ''}`}>
              <button type="button" role="tab" aria-selected={tab.id === activeTabId} aria-controls="builder-editor-workspace" className="model-tab-select" onClick={() => onSelect(tab.id)} title={`Open ${label}`}>
                <span className={`model-tab-status ${tab.dirty ? 'dirty' : ''}`} aria-hidden="true">{tab.dirty ? '●' : '○'}</span>
                <span className="model-tab-name">{label}</span>
                <small>{tab.projectId ? `r${tab.revision}` : 'draft'}</small>
              </button>
              <button type="button" className="model-tab-close" aria-label={`Close ${label}`} onClick={() => onClose(tab.id)}><X size={14} /></button>
            </div>
          )
        })}
      </div>
      <button type="button" className="model-tab-new" aria-label="New model" onClick={onNew}><Plus size={15} /><span className="action-label">New model</span></button>
    </nav>
  )
}
