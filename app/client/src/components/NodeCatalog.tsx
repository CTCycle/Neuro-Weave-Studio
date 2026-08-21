import { useEffect, useMemo, useRef, useState } from 'react'
import { Search } from 'lucide-react'
import type { ExecutionBackend, NodeManifest } from '../domain/types'
import { implementationKey } from '../domain/types'

interface Props {
  manifests: NodeManifest[]
  executionBackend?: ExecutionBackend
  kerasEnabled?: boolean
  onClose?: () => void
  onAdd?: (manifest: NodeManifest) => void
  isOpen?: boolean
}

export function NodeCatalog({ manifests, executionBackend = 'pytorch', kerasEnabled = true, onClose, onAdd, isOpen = true }: Props) {
  const [query, setQuery] = useState('')
  const [openCategories, setOpenCategories] = useState<Record<string, boolean>>({})
  const listRef = useRef<HTMLDivElement>(null)
  const categories = useMemo(() => [...new Set(manifests.map((manifest) => manifest.category))].sort(), [manifests])
  const filtered = useMemo(() => manifests.filter((manifest) => {
    const text = `${manifest.display_name} ${manifest.id} ${manifest.description}`.toLowerCase()
    return text.includes(query.toLowerCase())
  }), [manifests, query])
  const grouped = useMemo(() => categories.map((category) => ({
    category,
    manifests: filtered.filter((manifest) => manifest.category === category),
  })).filter((group) => group.manifests.length > 0), [categories, filtered])

  useEffect(() => {
    if (isOpen && listRef.current) listRef.current.scrollTop = 0
  }, [isOpen])

  useEffect(() => {
    setOpenCategories((current) => {
      const next = { ...current }
      categories.forEach((category) => {
        if (!(category in next)) next[category] = false
      })
      return next
    })
  }, [categories])

  useEffect(() => {
    if (!query.trim()) return
    setOpenCategories((current) => ({
      ...current,
      ...Object.fromEntries(grouped.map((group) => [group.category, true])),
    }))
  }, [grouped, query])

  return (
    <aside className="catalog-panel panel">
      <div className="panel-heading">
        <div><span className="eyebrow">BUILDING BLOCKS</span><h2>Node catalog</h2></div>
        {onClose && <button className="panel-close" onClick={onClose} aria-label="Close catalog">×</button>}
        <span className="catalog-count">{filtered.length} node{filtered.length === 1 ? '' : 's'}</span>
      </div>
      <label className="search-field"><Search size={16} /><input aria-label="Search nodes" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search layers and operations" /></label>
      <div ref={listRef} className="catalog-list">
        {grouped.length === 0 && <div className="catalog-empty-state"><strong>No matching nodes</strong><span>Try a different layer, operation, or category.</span></div>}
        {grouped.map((group) => <section className="catalog-category" key={group.category}>
          <button type="button" className="catalog-category-toggle" aria-expanded={Boolean(openCategories[group.category])} aria-controls={`catalog-${group.category}`} onClick={() => setOpenCategories((current) => ({ ...current, [group.category]: !current[group.category] }))}>
            <span><strong>{formatCategory(group.category)}</strong><small>{group.manifests.length} node{group.manifests.length === 1 ? '' : 's'}</small></span>
            <span aria-hidden="true">{openCategories[group.category] ? '−' : '+'}</span>
          </button>
          {openCategories[group.category] && <div id={`catalog-${group.category}`} className="catalog-category-items">{group.manifests.map((manifest) => {
            const supported = manifest.implementations[implementationKey(executionBackend, kerasEnabled)]?.supported ?? false
            return (
              <article
                key={`${manifest.id}@${manifest.version}`}
                className={`catalog-item ${supported ? '' : 'export-limited'}`}
                draggable
                role="button"
                tabIndex={0}
                title="Drag to the canvas or activate to add"
                onClick={() => onAdd?.(manifest)}
                onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onAdd?.(manifest) } }}
                onDragStart={(event) => {
                  event.dataTransfer.setData('application/neuro-weave-node', manifest.id)
                  event.dataTransfer.effectAllowed = 'copy'
                }}
              >
                <div><strong>{manifest.display_name}</strong><small>{manifest.description}</small></div>
                {!supported && <span className="compatibility-badge">Export limited</span>}
              </article>
            )
          })}</div>}
        </section>)}
      </div>
    </aside>
  )
}

function formatCategory(category: string): string {
  return category.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase())
}
