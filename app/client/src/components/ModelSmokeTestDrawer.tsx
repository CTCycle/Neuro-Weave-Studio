import { CheckCircle2, CircleAlert, FlaskConical, LoaderCircle, Play, X } from 'lucide-react'
import type {
  ExecutionBackend,
  ExecutionBackendStatus,
  JobResponse,
  SmokeTestErrorDetail,
  SmokeTestResponse,
  TensorExecutionInfo,
} from '../domain/types'

export type SmokeTestViewState =
  | { status: 'ready' }
  | { status: 'loading'; job?: JobResponse }
  | { status: 'succeeded'; result: SmokeTestResponse; job?: JobResponse }
  | { status: 'error'; error: SmokeTestErrorDetail; job?: JobResponse }

interface Props {
  backend: ExecutionBackend | null
  backendStatus?: ExecutionBackendStatus
  state: SmokeTestViewState
  onRun: () => void
  onStop?: () => void
  onClose?: () => void
  embedded?: boolean
}

export function ModelSmokeTestDrawer({ backend, backendStatus, state, onRun, onStop, onClose, embedded = false }: Props) {
  const content = <>
    <div className="smoke-test-intro"><FlaskConical size={18} /><p>Generated dummy inputs, one forward pass in inference mode, and no training or weight updates.</p></div>
    <div className="smoke-test-context"><span>Backend</span><strong>{backend ?? 'Choose an operational backend'}</strong><span>Device</span><strong>{backendStatus?.device_name ?? (backendStatus?.available ? (backendStatus.accelerator ?? 'CPU') : 'Unavailable')}</strong></div>
    {state.status === 'ready' && <ReadyState backend={backend} onRun={onRun} />}
    {state.status === 'loading' && <LoadingState job={state.job} onStop={onStop} />}
    {state.status === 'succeeded' && <SuccessState result={state.result} />}
    {state.status === 'error' && <ErrorState error={state.error} job={state.job} onRun={onRun} />}
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
  return <div className="smoke-test-state ready"><div><strong>Ready for a safe dry run</strong><span>The current graph will be validated before it is executed.</span></div><button className="primary" onClick={onRun} disabled={!backend}><Play size={16} />Run dummy inference</button>{!backend && <small>Select an operational backend on the Environment page first.</small>}</div>
}

function LoadingState({ job, onStop }: { job?: JobResponse; onStop?: () => void }) {
  const memory = job?.diagnostics.process_rss_bytes
    ? `${Math.round(job.diagnostics.process_rss_bytes / 1024 / 1024)} MB RSS`
    : 'Memory pending'
  return <div className="smoke-test-state loading" role="status"><LoaderCircle className="spin" size={22} /><strong>{job?.progress.message ?? 'Starting isolated worker…'}</strong><span>{job?.progress.phase ?? 'starting'} · {job?.worker.device_id ?? 'device pending'} · PID {job?.pid ?? 'pending'} · {memory}</span>{onStop && <button className="secondary" onClick={onStop}><X size={15} />Stop worker</button>}</div>
}

function SuccessState({ result }: { result: SmokeTestResponse }) {
  return <div className="smoke-test-result"><div className="smoke-test-status success"><CheckCircle2 size={18} /><strong>Execution succeeded</strong><span>{result.duration_ms.toFixed(1)} ms</span></div><div className="smoke-test-stats"><Stat label="Status" value="Succeeded" /><Stat label="Device" value={`${result.accelerator.toUpperCase()} · ${result.device_name}`} /><Stat label="Parameters" value={result.parameter_count.toLocaleString()} /></div><TensorSection label="Input tensors" tensors={result.inputs} /><TensorSection label="Output tensors" tensors={result.outputs} />{result.compilation && <CompilationSection summary={result.compilation} />}<p className="smoke-test-footnote">This result confirms model construction and inference only. Weights were not trained or updated.</p></div>
}

function ErrorState({ error, job, onRun }: { error: SmokeTestErrorDetail; job?: JobResponse; onRun: () => void }) {
  return <div className="smoke-test-result"><div className="smoke-test-status error"><CircleAlert size={18} /><strong>{job?.state === 'terminated' ? 'Execution stopped' : 'Execution blocked'}</strong><span>{error.code}</span></div><p className="smoke-test-error-message">{error.message}</p>{job && <p className="muted">Worker {job.worker.backend} · {job.worker.device_id} · PID {job.pid ?? 'reaped'} · exit {job.exit_code ?? 'n/a'}{job.forced_termination ? ' · force terminated' : ''}</p>}{error.backend && <p className="muted">Backend: {error.backend}{error.reason_code ? ` · ${error.reason_code}` : ''}</p>}{error.diagnostics && <div className="smoke-test-diagnostics">{error.diagnostics.map((diagnostic, index) => <div className="diagnostic error" key={`${diagnostic.code}-${index}`}><strong>{diagnostic.code}</strong><span>{diagnostic.message}</span></div>)}</div>}<button className="secondary" onClick={onRun}>Try again</button></div>
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>
}

function TensorSection({ label, tensors }: { label: string; tensors: TensorExecutionInfo[] }) {
  return <section className="tensor-section"><h3>{label}</h3>{tensors.length === 0 ? <p className="muted">No tensors reported.</p> : tensors.map((tensor) => <div className="tensor-row" key={tensor.node_id}><strong>{tensor.node_id}</strong><code>{`[${tensor.shape.join(', ')}]`}</code><span>{tensor.dtype}</span></div>)}</section>
}

function CompilationSection({ summary }: { summary: NonNullable<SmokeTestResponse['compilation']> }) {
  const lossCount = Object.keys(summary.losses).length
  const metricCount = Object.values(summary.metrics).reduce<number>((count, value) => count + (typeof value === 'object' && value !== null ? Object.keys(value).length : 0), 0)
  return <section className="tensor-section compilation-result-section"><h3>Compilation helpers</h3><div className="smoke-test-context"><span>Verification</span><strong>{summary.verified ? 'Passed' : 'Failed'}</strong><span>Optimizer</span><strong>{summary.optimizer.id ?? 'Not configured'}{summary.optimizer.instantiated ? ' · instantiated' : ''}</strong><span>Helpers</span><strong>{lossCount} loss{lossCount === 1 ? '' : 'es'} · {metricCount} metric{metricCount === 1 ? '' : 's'}</strong></div></section>
}
