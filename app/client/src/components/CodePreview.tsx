import { useEffect, useState } from 'react'
import { Clipboard, Check, X } from 'lucide-react'

export function CodePreview({ code, onClose, embedded = false }: { code: string; onClose?: () => void; embedded?: boolean }) {
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle')

  useEffect(() => {
    if (copyState !== 'copied') return
    const timeout = window.setTimeout(() => setCopyState('idle'), 2200)
    return () => window.clearTimeout(timeout)
  }, [copyState])

  const copyCode = async () => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(code)
      } else {
        const textarea = document.createElement('textarea')
        textarea.value = code
        textarea.setAttribute('readonly', '')
        textarea.style.position = 'fixed'
        textarea.style.opacity = '0'
        document.body.appendChild(textarea)
        textarea.select()
        const copied = document.execCommand('copy')
        textarea.remove()
        if (!copied) throw new Error('Clipboard access is unavailable.')
      }
      setCopyState('copied')
    } catch {
      setCopyState('failed')
    }
  }

  const copyControl = <div className="code-copy-control"><button className="icon-button" title={copyState === 'copied' ? 'Copied' : 'Copy code'} aria-label={copyState === 'copied' ? 'Copied' : 'Copy code'} onClick={copyCode}>{copyState === 'copied' ? <Check size={18} /> : <Clipboard size={18} />}</button>{copyState === 'copied' && <span className="copy-status" role="status">Copied</span>}{copyState === 'failed' && <span className="copy-status copy-status-error" role="status">Copy unavailable</span>}</div>

  if (embedded) return <section className="generated-code-panel panel"><div className="drawer-heading"><div><span className="eyebrow">FRAMEWORK ADAPTER</span><h2>Generated source</h2></div>{copyControl}</div><pre><code>{code}</code></pre></section>
  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={onClose}>
      <aside className="code-drawer" role="dialog" aria-modal="true" aria-label="Generated code" onMouseDown={(event) => event.stopPropagation()}>
        <div className="drawer-heading"><div><span className="eyebrow">FRAMEWORK ADAPTER</span><h2>Generated source</h2></div><div>{copyControl}<button className="icon-button" title="Close" onClick={onClose}><X size={18} /></button></div></div>
        <pre><code>{code}</code></pre>
      </aside>
    </div>
  )
}
