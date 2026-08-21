import { CheckCircle2, ChevronDown, XCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { ValidationDiagnostic, ValidationResult } from '../domain/types'
import { useGraphStore } from '../state/graphStore'

interface Props {
  result: ValidationResult | null
  supplementalDiagnostics?: ValidationDiagnostic[]
}

export function ValidationPanel({ result, supplementalDiagnostics = [] }: Props) {
  const [expanded, setExpanded] = useState(false)
  const diagnostics = mergeDiagnostics(result?.diagnostics ?? [], supplementalDiagnostics)
  const blocked = diagnostics.some((item) => item.severity === 'error')
  const valid = Boolean(result?.valid) && !blocked

  useEffect(() => {
    if (result || supplementalDiagnostics.length) setExpanded(blocked)
  }, [blocked, result, supplementalDiagnostics.length])

  return (
    <section className={`validation-panel panel ${result || supplementalDiagnostics.length ? 'has-result' : 'idle'}`}>
      <button type="button" className="validation-summary" aria-expanded={expanded} onClick={() => (result || supplementalDiagnostics.length > 0) && setExpanded((current) => !current)}>
        <span><span className="eyebrow">GRAPH HEALTH</span><strong>Validation</strong>{(result || supplementalDiagnostics.length > 0) && <small>{diagnostics.length ? `${diagnostics.length} finding${diagnostics.length === 1 ? '' : 's'}` : 'No findings'}</small>}</span>
        <span className="validation-summary-status">{result || supplementalDiagnostics.length ? <span className={`status-pill ${valid ? 'valid' : 'invalid'}`}>{valid ? <CheckCircle2 size={15} /> : <XCircle size={15} />}{valid ? 'Ready' : 'Blocked'}</span> : <small>Not run</small>}<ChevronDown className={expanded ? 'expanded' : ''} size={16} /></span>
      </button>
      {!result && !supplementalDiagnostics.length && <p className="muted">Run validation to inspect graph, parameters, shapes, source generation, runtime, and backend compatibility.</p>}
      {(result || supplementalDiagnostics.length > 0) && expanded && <div className="validation-body">
        <div className="model-stats"><div><span>Nodes</span><strong>{result?.shape_summaries.length ?? 0}</strong></div><div><span>Parameters</span><strong>{result?.total_parameters?.toLocaleString() ?? 'Unknown'}</strong></div><div><span>Diagnostics</span><strong>{diagnostics.length}</strong></div></div>
        <div className="diagnostic-list">{diagnostics.length === 0 ? <div className="diagnostic info"><strong>info</strong><span>No validation findings.</span></div> : diagnostics.map((item, index) => <button type="button" key={`${item.code}-${index}`} className={`diagnostic ${item.severity} ${item.node_id || item.edge_id ? 'is-actionable' : ''}`} onClick={() => selectDiagnosticTarget(item)}><strong>{item.severity}</strong><span>{item.message}<small>{diagnosticContext(item)}</small></span></button>)}</div>
      </div>}
    </section>
  )
}

function mergeDiagnostics(primary: ValidationDiagnostic[], supplemental: ValidationDiagnostic[]): ValidationDiagnostic[] {
  const seen = new Set<string>()
  return [...primary, ...supplemental].filter((diagnostic) => {
    const key = [diagnostic.code, diagnostic.message, diagnostic.node_id ?? '', diagnostic.edge_id ?? '', diagnostic.parameter_id ?? ''].join('|')
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
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
