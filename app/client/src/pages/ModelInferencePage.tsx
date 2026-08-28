import { useEffect, useMemo, useRef, useState } from 'react'
import { FlaskConical } from 'lucide-react'
import { Link, useOutletContext } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import { ModelSmokeTestDrawer, type SmokeTestViewState } from '../components/ModelSmokeTestDrawer'
import type { JobResponse, SmokeTestErrorDetail, SmokeTestResponse, WorkerConfiguration } from '../domain/types'
import { useAppResources } from '../state/useAppResources'
import type { ModelWorkspaceContext } from './ModelWorkspacePage'

export function ModelInferencePage() {
  const { project } = useOutletContext<ModelWorkspaceContext>()
  const { backendStatuses, runtimePreferences } = useAppResources()
  const backend = project.execution_backend
  const runtimePreference = runtimePreferences[backend]
  const runtimeSelection = useMemo(
    () => profileRuntimeSelection(project.keras_enabled, backend, backendStatuses, runtimePreference),
    [backend, backendStatuses, project.keras_enabled, runtimePreference],
  )
  const [state, setState] = useState<SmokeTestViewState>({ status: 'ready' })
  const mounted = useRef(true)

  useEffect(() => {
    setState({ status: 'ready' })
  }, [project.id, project.current_revision, runtimeSelection])

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false }
  }, [])

  const run = async () => {
    if (!runtimeSelection) {
      setState({ status: 'error', error: { code: 'backend_unavailable', message: 'Select an operational execution backend before running the smoke test.' } })
      return
    }
    setState({ status: 'loading' })
    try {
      let job = await api.submitSmokeInference(project.id, project.graph, runtimeSelection)
      if (mounted.current) setState({ status: 'loading', job })
      while (mounted.current && !['succeeded', 'terminated', 'failed'].includes(job.state)) {
        await new Promise((resolve) => window.setTimeout(resolve, 250))
        job = await api.getJob(job.id)
        if (mounted.current) setState({ status: 'loading', job })
      }
      if (!mounted.current) return
      if (job.state === 'succeeded') setState({ status: 'succeeded', result: smokeResult(job), job })
      else setState({ status: 'error', error: job.error ?? { code: 'worker_failed', message: 'The inference worker did not complete.' }, job })
    } catch (error) {
      if (error instanceof ApiError && typeof error.detail === 'object' && error.detail !== null) setState({ status: 'error', error: error.detail as SmokeTestErrorDetail })
      else setState({ status: 'error', error: { code: 'request_failed', message: error instanceof Error ? error.message : String(error) } })
    }
  }

  const stop = async () => {
    if (state.status !== 'loading' || !state.job) return
    try {
      const job = await api.stopJob(state.job.id)
      if (mounted.current) setState({ status: 'loading', job })
    } catch (error) {
      setState({ status: 'error', error: { code: 'stop_failed', message: error instanceof Error ? error.message : String(error) } })
    }
  }

  return <div className="workflow-page inference-page"><div className="workflow-intro"><div><span className="eyebrow">INFERENCE</span><h2>Run a safe model check</h2><p>This workspace creates compatible dummy inputs and performs one forward pass. It never trains or updates weights.</p></div><div className="backend-selection-summary workflow-backend-summary"><span>Model backend preference</span>{runtimeSelection ? <strong>{runtimeSelection.api} · {backend} · {runtimeSelection.device_id}</strong> : <Link to="/environment">Configure {backend} in Environment</Link>}</div></div><section className="inference-controls panel"><div className="inference-control-note"><FlaskConical size={18} /><span>Inference uses the model's persisted backend and its matching runtime preference. It never changes model data.</span></div><Link className="secondary environment-link" to="/environment">Configure runtime</Link></section><ModelSmokeTestDrawer backend={backend} backendStatus={backendStatuses.find((item) => item.id === backend)} state={state} onRun={run} onStop={stop} embedded /></div>
}

function profileRuntimeSelection(
  kerasEnabled: boolean,
  backend: JobResponse['worker']['backend'],
  backendStatuses: ReturnType<typeof useAppResources>['backendStatuses'],
  preference: WorkerConfiguration | undefined,
): WorkerConfiguration | null {
  const status = backendStatuses.find((item) => item.id === backend)
  if (!status || !preference) return null
  const api = kerasEnabled ? 'keras' : 'native'
  const apiStatus = status.runtime_apis?.find((item) => item.id === api)
  if (apiStatus && !apiStatus.available) return null
  return { ...preference, api }
}

function smokeResult(job: JobResponse): SmokeTestResponse {
  const result = job.result ?? {}
  return {
    status: 'succeeded',
    execution_backend: job.worker.backend,
    accelerator: String(result.accelerator ?? 'cpu'),
    device_name: String(result.device_name ?? job.worker.device_id),
    inputs: (result.inputs ?? []) as SmokeTestResponse['inputs'],
    outputs: (result.outputs ?? []) as SmokeTestResponse['outputs'],
    parameter_count: Number(result.parameter_count ?? 0),
    duration_ms: Number(result.duration_ms ?? 0),
    compilation: (result.compilation ?? null) as SmokeTestResponse['compilation'],
  }
}
