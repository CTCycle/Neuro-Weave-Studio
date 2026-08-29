import { Cpu, Info } from 'lucide-react'
import { BackendStatusPanel } from '../components/BackendStatusPanel'
import { useAppResources } from '../state/useAppResources'

export function EnvironmentPage() {
  const { backendStatuses, readiness } = useAppResources()
  const operationalCount = backendStatuses.filter((item) => item.available).length

  return (
    <div className="environment-page">
      <div className="environment-header">
        <div>
          <span className="eyebrow">RUNTIME ENVIRONMENT</span>
          <h1>Execution backends</h1>
          <p>Configure the API, device, and memory policy preference stored for each execution backend.</p>
        </div>
        <div className="environment-summary"><Cpu size={16} /><strong>{!readiness.runtime ? 'Reading runtime…' : `${operationalCount} operational`}</strong></div>
      </div>
      <div className="environment-note panel"><Info size={16} /><span>Availability comes from isolated worker discovery. Matching environment snapshots are reused until packages, hardware, or backend configuration changes. Each model uses the preference for its persisted backend.</span></div>
      <BackendStatusPanel />
    </div>
  )
}
