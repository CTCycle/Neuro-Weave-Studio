import { Clipboard, X } from 'lucide-react'

export function CodePreview({ code, onClose, embedded = false }: { code: string; onClose?: () => void; embedded?: boolean }) {
  if (embedded) return <section className="generated-code-panel panel"><div className="drawer-heading"><div><span className="eyebrow">FRAMEWORK ADAPTER</span><h2>Generated source</h2></div><button className="icon-button" title="Copy code" onClick={() => navigator.clipboard.writeText(code)}><Clipboard size={18} /></button></div><pre><code>{code}</code></pre></section>
  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={onClose}>
      <aside className="code-drawer" role="dialog" aria-modal="true" aria-label="Generated code" onMouseDown={(event) => event.stopPropagation()}>
        <div className="drawer-heading"><div><span className="eyebrow">FRAMEWORK ADAPTER</span><h2>Generated source</h2></div><div><button className="icon-button" title="Copy code" onClick={() => navigator.clipboard.writeText(code)}><Clipboard size={18} /></button><button className="icon-button" title="Close" onClick={onClose}><X size={18} /></button></div></div>
        <pre><code>{code}</code></pre>
      </aside>
    </div>
  )
}
