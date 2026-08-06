import { CheckCircle2, XCircle } from 'lucide-react'
import type { ValidationResult } from '../domain/types'
import { useGraphStore } from '../state/graphStore'

export function ValidationPanel({ result }: { result: ValidationResult | null }) {
  return (
    <section className={`validation-panel panel ${result ? 'has-result' : 'idle'}`}>
      <div className="panel-heading"><div><span className="eyebrow">GRAPH HEALTH</span><h2>Validation</h2></div>{result && <span className={`status-pill ${result.valid ? 'valid' : 'invalid'}`}>{result.valid ? <CheckCircle2 size={15} /> : <XCircle size={15} />}{result.valid ? 'Ready' : 'Blocked'}</span>}</div>
      {!result && <p className="muted">Run validation to inspect graph structure, tensor shapes, and code-target compatibility.</p>}
      {result && <>
        <div className="model-stats"><div><span>Nodes</span><strong>{result.shape_summaries.length}</strong></div><div><span>Parameters</span><strong>{result.total_parameters?.toLocaleString() ?? 'Unknown'}</strong></div><div><span>Diagnostics</span><strong>{result.diagnostics.length}</strong></div></div>
        <div className="diagnostic-list">{result.diagnostics.length === 0 ? <div className="diagnostic info"><strong>info</strong><span>No validation findings.</span></div> : result.diagnostics.map((item, index) => <button key={`${item.code}-${index}`} className={`diagnostic ${item.severity}`} onClick={() => item.node_id && selectDiagnosticNode(item.node_id)}><strong>{item.severity}</strong><span>{item.message}</span></button>)}</div>
      </>}
    </section>
  )
}

function selectDiagnosticNode(nodeId: string) {
  useGraphStore.getState().selectNode(nodeId)
}
