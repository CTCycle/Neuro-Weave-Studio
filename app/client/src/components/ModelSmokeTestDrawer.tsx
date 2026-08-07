import { CheckCircle2, CircleAlert, FlaskConical, LoaderCircle, Play, X } from 'lucide-react'
import type {
  ExecutionBackend,
  ExecutionBackendStatus,
  SmokeTestErrorDetail,
  SmokeTestResponse,
  TensorExecutionInfo,
} from '../domain/types'

export type SmokeTestViewState =
  | { status: 'ready' }
  | { status: 'loading' }
  | { status: 'succeeded'; result: SmokeTestResponse }
  | { status: 'error'; error: SmokeTestErrorDetail }

interface Props {
  backend: ExecutionBackend | null
  backendStatus?: ExecutionBackendStatus
  state: SmokeTestViewState
  onRun: () => void
  onClose?: () => void
  embedded?: boolean
}

export function ModelSmokeTestDrawer({ backend, backendStatus, state, onRun, onClose, embedded = false }: Props) {
  const content = <>
    <div className="smoke-test-intro"><FlaskConical size={18} /><p>Generated dummy inputs, one forward pass in inference mode, and no training or weight updates.</p></div>
    <div className="smoke-test-context"><span>Backend</span><strong>{backend ?? 'Choose an operational backend'}</strong><span>Device</span><strong>{backendStatus?.device_name ?? (backendStatus?.available ? (backendStatus.accelerator ?? 'CPU') : 'Unavailable')}</strong></div>
    {state.status === 'ready' && <ReadyState backend={backend} onRun={onRun} />}
    {state.status === 'loading' && <div className="smoke-test-state loading" role="status"><LoaderCircle className="spin" size={22} /><strong>Running one dummy-tensor pass…</strong><span>Building the configured model and checking tensor shapes.</span></div>}
    {state.status === 'succeeded' && <SuccessState result={state.result} />}
    {state.status === 'error' && <ErrorState error={state.error} onRun={onRun} />}
  </>
  if (embedded) return <section className="smoke-test-panel panel"><div className="drawer-heading"><div><span className="eyebrow">INFERENCE-ONLY DRY RUN</span><h2>Model smoke test</h2></div><FlaskConical size={18} /></div><div className="smoke-test-content">{content}</div></section>
  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={onClose}>
      <aside className="code-drawer smoke-test-drawer" role="dialog" aria-modal="true" aria-label="Model smoke test" onMouseDown={(event) => event.stopPropagation()}>
        <div className="drawer-heading">
          <div><span className="eyebrow">INFERENCE-ONLY DRY RUN</span><h2>Model smoke test</h2></div>
          <button className="icon-button" title="Close" onClick={onClose}><X size={18} /></button>
        </div>
        <div className="smoke-test-content">{content}</div>
      </aside>
    </div>
  )
}

function ReadyState({ backend, onRun }: { backend: ExecutionBackend | null; onRun: () => void }) {
  return <div className="smoke-test-state ready"><div><strong>Ready for a safe dry run</strong><span>The current graph will be validated before it is executed.</span></div><button className="primary" onClick={onRun} disabled={!backend}><Play size={16} />Run dummy inference</button>{!backend && <small>Select an operational backend in the toolbar first.</small>}</div>
}

function SuccessState({ result }: { result: SmokeTestResponse }) {
  return <div className="smoke-test-result"><div className="smoke-test-status success"><CheckCircle2 size={18} /><strong>Execution succeeded</strong><span>{result.duration_ms.toFixed(1)} ms</span></div><div className="smoke-test-stats"><Stat label="Status" value="Succeeded" /><Stat label="Device" value={`${result.accelerator.toUpperCase()} · ${result.device_name}`} /><Stat label="Parameters" value={result.parameter_count.toLocaleString()} /></div><TensorSection label="Input tensors" tensors={result.inputs} /><TensorSection label="Output tensors" tensors={result.outputs} /><p className="smoke-test-footnote">This result confirms model construction and inference only. Weights were not trained or updated.</p></div>
}

function ErrorState({ error, onRun }: { error: SmokeTestErrorDetail; onRun: () => void }) {
  return <div className="smoke-test-result"><div className="smoke-test-status error"><CircleAlert size={18} /><strong>Execution blocked</strong><span>{error.code}</span></div><p className="smoke-test-error-message">{error.message}</p>{error.backend && <p className="muted">Backend: {error.backend}{error.reason_code ? ` · ${error.reason_code}` : ''}</p>}{error.diagnostics && <div className="smoke-test-diagnostics">{error.diagnostics.map((diagnostic, index) => <div className="diagnostic error" key={`${diagnostic.code}-${index}`}><strong>{diagnostic.code}</strong><span>{diagnostic.message}</span></div>)}</div>}<button className="secondary" onClick={onRun}>Try again</button></div>
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>
}

function TensorSection({ label, tensors }: { label: string; tensors: TensorExecutionInfo[] }) {
  return <section className="tensor-section"><h3>{label}</h3>{tensors.length === 0 ? <p className="muted">No tensors reported.</p> : tensors.map((tensor) => <div className="tensor-row" key={tensor.node_id}><strong>{tensor.node_id}</strong><code>{`[${tensor.shape.join(', ')}]`}</code><span>{tensor.dtype}</span></div>)}</section>
}
