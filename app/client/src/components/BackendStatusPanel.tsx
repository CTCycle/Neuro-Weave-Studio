import { AlertTriangle, CheckCircle2, Cpu, PackageX, Terminal } from 'lucide-react'
import type { ExecutionBackendStatus } from '../domain/types'
import { useAppResources } from '../state/useAppResources'

const labels: Record<ExecutionBackendStatus['id'], string> = {
  tensorflow: 'TensorFlow',
  pytorch: 'PyTorch',
  jax: 'JAX',
}

export function BackendStatusPanel() {
  const { backendStatuses, loading } = useAppResources()

  if (loading || backendStatuses.length === 0) return null

  const operationalCount = backendStatuses.filter((item) => item.available).length
  return (
    <section className="backend-status-panel" aria-label="Compute environment status">
      <div className="backend-status-heading">
        <div>
          <span className="eyebrow">COMPUTE ENVIRONMENT</span>
          <h2>Execution backends</h2>
          <p>
            {operationalCount ? `${operationalCount} operational backend${operationalCount === 1 ? '' : 's'}` : 'No operational backend'}
            {' · '}Keras 3 is the core model API.
          </p>
        </div>
        <span className="backend-status-command"><Terminal size={13} />start_on_windows.ps1 -Action ManageBackends</span>
      </div>
      <div className="backend-status-grid">
        {backendStatuses.map((status) => <BackendStatusCard key={status.id} status={status} />)}
      </div>
    </section>
  )
}

function BackendStatusCard({ status }: { status: ExecutionBackendStatus }) {
  const state = status.available ? 'usable' : status.installed ? 'unavailable' : 'missing'
  const StateIcon = status.available ? CheckCircle2 : status.installed ? AlertTriangle : PackageX
  const detail = status.available
    ? `${status.accelerator ?? 'cpu'}${status.device_name ? ` · ${status.device_name}` : ''}`
    : status.installed
      ? formatReason(status.reason_code)
      : 'Not installed'

  return (
    <article className={`backend-status-card ${state}`}>
      <div className="backend-status-card-top">
        <span className="backend-status-name"><Cpu size={14} />{labels[status.id]}</span>
        <span className="backend-status-state"><StateIcon size={13} />{status.available ? 'Usable' : status.installed ? 'Installed' : 'Missing'}</span>
      </div>
      <strong>{status.version ? `v${status.version}` : 'Optional framework'}</strong>
      <span>{detail}</span>
    </article>
  )
}

function formatReason(reason: string | null | undefined): string {
  if (reason === 'dependency_missing') return 'Installed package could not be imported'
  if (reason === 'worker_failed') return 'Worker probe failed'
  if (reason === 'backend_import_failed') return 'Backend import failed'
  return reason ? reason.replaceAll('_', ' ') : 'Capability probe unavailable'
}
