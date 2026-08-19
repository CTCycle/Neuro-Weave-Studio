import { Cpu, Info } from 'lucide-react'
import { BackendStatusPanel } from '../components/BackendStatusPanel'
import { useAppResources } from '../state/useAppResources'

export function EnvironmentPage() {
  const { backendStatuses, loading } = useAppResources()
  const operationalCount = backendStatuses.filter((item) => item.available).length

  return (
    <div className="environment-page">
      <div className="environment-header">
        <div>
          <span className="eyebrow">RUNTIME ENVIRONMENT</span>
          <h1>Execution backends</h1>
          <p>Select the session-scoped API, backend, device, and memory policy used by future jobs.</p>
        </div>
        <div className="environment-summary"><Cpu size={16} /><strong>{loading ? 'Reading runtime…' : `${operationalCount} operational`}</strong></div>
      </div>
      <div className="environment-note panel"><Info size={16} /><span>Availability comes from fresh probe workers. Keras jobs are operational; native-framework inference remains explicitly unavailable. Selection stays in this session and is not saved into model data.</span></div>
      <BackendStatusPanel />
    </div>
  )
}
