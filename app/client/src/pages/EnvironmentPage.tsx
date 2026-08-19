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
          <p>Select the request-scoped runtime used for binary export and inference checks.</p>
        </div>
        <div className="environment-summary"><Cpu size={16} /><strong>{loading ? 'Reading runtime…' : `${operationalCount} operational`}</strong></div>
      </div>
      <div className="environment-note panel"><Info size={16} /><span>Backend availability is detected from the live runtime. Your selection stays in this session and is not saved into model data.</span></div>
      <BackendStatusPanel />
    </div>
  )
}
