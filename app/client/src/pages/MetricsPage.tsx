import { Activity, Info } from 'lucide-react'
import type { CompilationOption } from '../domain/types'
import { useAppResources } from '../state/useAppResources'

export function MetricsPage() {
  const { compilationCapabilities, loading } = useAppResources()
  const catalog = compilationCapabilities?.keras

  if (loading) {
    return <section className="page-state panel"><span className="eyebrow">METRICS CATALOG</span><h1>Loading catalog…</h1></section>
  }

  if (!catalog) {
    return <section className="page-state panel"><span className="eyebrow">METRICS CATALOG</span><h1>Catalog unavailable</h1><p>The runtime metadata did not include the Keras metrics and loss catalog.</p></section>
  }

  return (
    <div className="metrics-page">
      <header className="catalog-page-header">
        <div>
          <span className="eyebrow">COMPILATION CATALOG</span>
          <h1>Metrics</h1>
          <p>Review the read-only metric and loss definitions available to model compilation.</p>
        </div>
        <div className="environment-summary"><Activity size={16} /><strong>{catalog.metrics.length + catalog.losses.length} definitions</strong></div>
      </header>
      <div className="metrics-note panel"><Info size={16} /><span>This catalog is intentionally read-only for now. It is the shared metadata boundary for the future editable metrics and loss catalog.</span></div>
      <div className="metrics-grid">
        <CatalogSection title="Metrics" eyebrow="EVALUATION" items={catalog.metrics} />
        <CatalogSection title="Loss functions" eyebrow="OPTIMIZATION" items={catalog.losses} />
      </div>
    </div>
  )
}

function CatalogSection({ title, eyebrow, items }: { title: string; eyebrow: string; items: CompilationOption[] }) {
  return (
    <section className="metrics-section panel" aria-labelledby={`metrics-${title.toLowerCase().replaceAll(' ', '-')}`}>
      <div className="section-heading"><div><span className="eyebrow">{eyebrow}</span><h2 id={`metrics-${title.toLowerCase().replaceAll(' ', '-')}`}>{title}</h2></div><span className="catalog-count">{items.length}</span></div>
      <div className="metrics-list">
        {items.map((item) => <article className="metrics-card" key={item.id}><div className="metrics-card-heading"><div><strong>{item.label}</strong><code>{item.id}</code></div><span className="read-only-badge">READ ONLY</span></div><p>{item.description}</p>{item.formula && <div className="metrics-formula"><span>Formula</span><code>{item.formula}</code></div>}</article>)}
      </div>
    </section>
  )
}
