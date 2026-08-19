import { CheckCircle2, ChevronDown, XCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { ValidationDiagnostic, ValidationResult } from '../domain/types'
import { useGraphStore } from '../state/graphStore'

export function ValidationPanel({ result }: { result: ValidationResult | null }) {
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    if (result) setExpanded(!result.valid)
  }, [result])

  return (
    <section className={`validation-panel panel ${result ? 'has-result' : 'idle'}`}>
      <button type="button" className="validation-summary" aria-expanded={expanded} onClick={() => result && setExpanded((current) => !current)}>
        <span><span className="eyebrow">GRAPH HEALTH</span><strong>Validation</strong>{result && <small>{result.diagnostics.length ? `${result.diagnostics.length} finding${result.diagnostics.length === 1 ? '' : 's'}` : 'No findings'}</small>}</span>
        <span className="validation-summary-status">{result ? <span className={`status-pill ${result.valid ? 'valid' : 'invalid'}`}>{result.valid ? <CheckCircle2 size={15} /> : <XCircle size={15} />}{result.valid ? 'Ready' : 'Blocked'}</span> : <small>Not run</small>}<ChevronDown className={expanded ? 'expanded' : ''} size={16} /></span>
      </button>
      {!result && <p className="muted">Run validation to inspect graph structure, tensor shapes, and code-target compatibility.</p>}
      {result && expanded && <div className="validation-body">
        <div className="model-stats"><div><span>Nodes</span><strong>{result.shape_summaries.length}</strong></div><div><span>Parameters</span><strong>{result.total_parameters?.toLocaleString() ?? 'Unknown'}</strong></div><div><span>Diagnostics</span><strong>{result.diagnostics.length}</strong></div></div>
        <div className="diagnostic-list">{result.diagnostics.length === 0 ? <div className="diagnostic info"><strong>info</strong><span>No validation findings.</span></div> : result.diagnostics.map((item, index) => <button type="button" key={`${item.code}-${index}`} className={`diagnostic ${item.severity} ${item.node_id || item.edge_id ? 'is-actionable' : ''}`} onClick={() => selectDiagnosticTarget(item)}><strong>{item.severity}</strong><span>{item.message}<small>{diagnosticContext(item)}</small></span></button>)}</div>
      </div>}
    </section>
  )
}

function selectDiagnosticTarget(diagnostic: ValidationDiagnostic) {
  if (diagnostic.node_id) useGraphStore.getState().selectNode(diagnostic.node_id)
  else if (diagnostic.edge_id) useGraphStore.getState().selectEdge(diagnostic.edge_id)
}

function diagnosticContext(diagnostic: ValidationDiagnostic): string {
  if (diagnostic.node_id) return `Node ${diagnostic.node_id}`
  if (diagnostic.edge_id) return `Connection ${diagnostic.edge_id}`
  if (diagnostic.parameter_id) return `Parameter ${diagnostic.parameter_id}`
  return 'Graph-level finding'
}
