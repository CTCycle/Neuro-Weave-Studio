import { useMemo, useState } from 'react'
import { Search } from 'lucide-react'
import type { NodeManifest, TargetFramework } from '../domain/types'

interface Props {
  manifests: NodeManifest[]
  framework: TargetFramework
}

export function NodeCatalog({ manifests, framework }: Props) {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('all')
  const categories = useMemo(() => ['all', ...new Set(manifests.map((manifest) => manifest.category))], [manifests])
  const filtered = useMemo(() => manifests.filter((manifest) => {
    const matchesCategory = category === 'all' || manifest.category === category
    const text = `${manifest.display_name} ${manifest.id} ${manifest.description}`.toLowerCase()
    return matchesCategory && text.includes(query.toLowerCase())
  }), [manifests, query, category])

  return (
    <aside className="catalog-panel panel">
      <div className="panel-heading">
        <div><span className="eyebrow">BUILDING BLOCKS</span><h2>Node catalog</h2></div>
        <span className="count-pill">{filtered.length}</span>
      </div>
      <label className="search-field"><Search size={16} /><input aria-label="Search nodes" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search layers and operations" /></label>
      <div className="category-tabs" role="tablist">
        {categories.map((item) => <button key={item} className={category === item ? 'active' : ''} onClick={() => setCategory(item)}>{item.replace('_', ' ')}</button>)}
      </div>
      <div className="catalog-list">
        {filtered.map((manifest) => {
          const supported = manifest.frameworks[framework]?.supported ?? false
          return (
            <article
              key={`${manifest.id}@${manifest.version}`}
              className={`catalog-item ${supported ? '' : 'export-limited'}`}
              draggable
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
