import { useEffect, useState } from 'react'
import { Cpu, FlaskConical } from 'lucide-react'
import { useOutletContext } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import { ModelSmokeTestDrawer, type SmokeTestViewState } from '../components/ModelSmokeTestDrawer'
import type { ExecutionBackend, SmokeTestErrorDetail } from '../domain/types'
import { useAppResources } from '../state/useAppResources'
import type { ModelWorkspaceContext } from './ModelWorkspacePage'

export function ModelInferencePage() {
  const { project } = useOutletContext<ModelWorkspaceContext>()
  const { backendStatuses } = useAppResources()
  const available = backendStatuses.filter((item) => item.available)
  const [backend, setBackend] = useState<ExecutionBackend | null>(null)
  const [state, setState] = useState<SmokeTestViewState>({ status: 'ready' })

  useEffect(() => {
    const available = backendStatuses.filter((item) => item.available)
    setBackend(available.length === 1 ? available[0].id : null)
    setState({ status: 'ready' })
  }, [project.id, project.current_revision, backendStatuses])

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

  return <div className="workflow-page inference-page"><div className="workflow-intro"><div><span className="eyebrow">INFERENCE</span><h2>Run a safe model check</h2><p>This workspace creates compatible dummy inputs and performs one forward pass. It never trains or updates weights.</p></div><div className="backend-chip"><Cpu size={16} />{available.length ? `${available.length} operational backend${available.length === 1 ? '' : 's'}` : 'No operational backend'}</div></div><section className="inference-controls panel"><label className="field"><span>Execution backend</span><select aria-label="Inference execution backend" value={backend ?? ''} onChange={(event) => { setBackend((event.target.value || null) as ExecutionBackend | null); setState({ status: 'ready' }) }} disabled={available.length === 0}><option value="">{available.length === 0 ? 'None operational' : available.length === 1 ? 'Select backend' : 'Choose backend…'}</option>{backendStatuses.map((item) => <option key={item.id} value={item.id} disabled={!item.available}>{`${item.id}${item.available ? ` · ${(item.accelerator ?? 'cpu').toUpperCase()}` : ` · ${item.reason_code ?? 'unavailable'}`}`}</option>)}</select></label><div className="inference-control-note"><FlaskConical size={18} /><span>Inference-only dry run. The selected backend is request-scoped and is not stored in model data.</span></div></section><ModelSmokeTestDrawer backend={backend} backendStatus={backendStatuses.find((item) => item.id === backend)} state={state} onRun={run} embedded /></div>
}
