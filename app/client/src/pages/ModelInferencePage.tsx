import { useEffect, useState } from 'react'
import { FlaskConical } from 'lucide-react'
import { Link, useOutletContext } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import { ModelSmokeTestDrawer, type SmokeTestViewState } from '../components/ModelSmokeTestDrawer'
import type { SmokeTestErrorDetail } from '../domain/types'
import { useAppResources } from '../state/useAppResources'
import type { ModelWorkspaceContext } from './ModelWorkspacePage'

export function ModelInferencePage() {
  const { project } = useOutletContext<ModelWorkspaceContext>()
  const { backendStatuses, selectedExecutionBackend } = useAppResources()
  const backend = selectedExecutionBackend
  const [state, setState] = useState<SmokeTestViewState>({ status: 'ready' })

  useEffect(() => {
    setState({ status: 'ready' })
  }, [project.id, project.current_revision, selectedExecutionBackend])

  const run = async () => {
    if (!backend) {
      setState({ status: 'error', error: { code: 'backend_unavailable', message: 'Select an operational execution backend before running the smoke test.' } })
      return
    }
    setState({ status: 'loading' })
    try {
      setState({ status: 'succeeded', result: await api.smokeTest(project.graph, backend) })
    } catch (error) {
      if (error instanceof ApiError && typeof error.detail === 'object' && error.detail !== null) setState({ status: 'error', error: error.detail as SmokeTestErrorDetail })
      else setState({ status: 'error', error: { code: 'request_failed', message: error instanceof Error ? error.message : String(error) } })
    }
  }

  return <div className="workflow-page inference-page"><div className="workflow-intro"><div><span className="eyebrow">INFERENCE</span><h2>Run a safe model check</h2><p>This workspace creates compatible dummy inputs and performs one forward pass. It never trains or updates weights.</p></div><div className="backend-selection-summary workflow-backend-summary"><span>Session backend</span>{backend ? <strong>{backendStatuses.find((item) => item.id === backend)?.device_name ?? backend}</strong> : <Link to="/environment">Choose in Environment</Link>}</div></div><section className="inference-controls panel"><div className="inference-control-note"><FlaskConical size={18} /><span>Inference-only dry run. The selected backend is request-scoped and is not stored in model data.</span></div><Link className="secondary environment-link" to="/environment">{backend ? 'Change backend' : 'Choose backend'}</Link></section><ModelSmokeTestDrawer backend={backend} backendStatus={backendStatuses.find((item) => item.id === backend)} state={state} onRun={run} embedded /></div>
}
