import { useEffect, useMemo, useRef, useState } from 'react'
import { Search } from 'lucide-react'
import type { CodeTarget, NodeManifest } from '../domain/types'

interface Props {
  manifests: NodeManifest[]
  codeTarget?: CodeTarget
  onClose?: () => void
  onAdd?: (manifest: NodeManifest) => void
  isOpen?: boolean
}

export function NodeCatalog({ manifests, codeTarget, onClose, onAdd, isOpen = true }: Props) {
  const selectedTarget = codeTarget ?? 'keras'
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('all')
  const listRef = useRef<HTMLDivElement>(null)
  const categories = useMemo(() => ['all', ...new Set(manifests.map((manifest) => manifest.category))], [manifests])
  const filtered = useMemo(() => manifests.filter((manifest) => {
    const matchesCategory = category === 'all' || manifest.category === category
    const text = `${manifest.display_name} ${manifest.id} ${manifest.description}`.toLowerCase()
    return matchesCategory && text.includes(query.toLowerCase())
  }), [manifests, query, category])

  useEffect(() => {
    if (isOpen && listRef.current) listRef.current.scrollTop = 0
  }, [isOpen])

  return (
    <aside className="catalog-panel panel">
      <div className="panel-heading">
        <div><span className="eyebrow">BUILDING BLOCKS</span><h2>Node catalog</h2></div>
        {onClose && <button className="panel-close" onClick={onClose} aria-label="Close catalog">×</button>}
        <span className="count-pill">{filtered.length}</span>
      </div>
      <label className="search-field"><Search size={16} /><input aria-label="Search nodes" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search layers and operations" /></label>
      <div className="category-tabs" role="tablist">
        {categories.map((item) => <button key={item} className={category === item ? 'active' : ''} onClick={() => setCategory(item)}>{item.replace('_', ' ')}</button>)}
      </div>
      <div ref={listRef} className="catalog-list">
        {filtered.map((manifest) => {
          const supported = manifest.frameworks[selectedTarget]?.supported ?? false
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
        })}
      </div>
    </aside>
  )
}
