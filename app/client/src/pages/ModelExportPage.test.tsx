import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type {
  AppResources,
} from '../state/appResourcesContext'
import { useAppResources } from '../state/useAppResources'
import type {
  ExportCapability,
  ExportCapabilitiesResponse,
  ExportOptionDefinition,
  ExecutionBackendStatus,
  JobResponse,
  ProjectDocument,
  WorkerConfiguration,
} from '../domain/types'
import { ModelExportContent } from './ModelExportPage'

vi.mock('../state/useAppResources', () => ({ useAppResources: vi.fn() }))

const mockedResources = vi.mocked(useAppResources)

const project: ProjectDocument = {
  id: '00000000-0000-0000-0000-000000000001',
  name: 'Parity Demo',
  description: 'A deterministic export test project.',
  created_at: '2026-08-27T00:00:00Z',
  updated_at: '2026-08-27T00:00:00Z',
  schema_version: 3,
  execution_backend: 'pytorch',
  keras_enabled: false,
  current_revision: 7,
  graph: {
    schema_version: 3,
    nodes: [],
    edges: [],
    model_configuration: {
      keras: {
        optimizer: 'adam',
        optimizer_parameters: {},
        losses: {},
        metrics: {},
        default_initializer: 'glorot_uniform',
        default_regularizer: 'none',
        default_constraint: 'none',
        mixed_precision: false,
        random_seed: 11,
      },
      native: {
        tensorflow: { jit_compile: false },
        pytorch: { optimizer: 'adam', optimizer_parameters: { learning_rate: 0.001 }, losses: {}, metrics: {}, compile_enabled: false, mode: 'default', fullgraph: false, dynamic: false },
        jax: { jit_enabled: false },
      },
    },
    export_settings: { include_comments: true, include_shape_assertions: true, preferred_binary_format: 'keras' },
    future_training_configuration: {},
  },
  validation_status: 'valid',
  generated_source: null,
}

const nativeWorker: WorkerConfiguration = {
  api: 'native',
  backend: 'pytorch',
  device_id: 'cpu',
  memory_policy: { kind: 'default' },
  precision: 'float32',
}

const backendStatus: ExecutionBackendStatus = {
  id: 'pytorch',
  installed: true,
  version: '2.8.0',
  available: true,
  devices: [{ device_id: 'cpu', lease_key: 'cpu', kind: 'cpu', index: 0, name: 'CPU', accelerator: 'cpu', available: true, supported_memory_policies: ['default'] }],
  runtime_apis: [{ id: 'keras', available: false, reason_code: 'keras_disabled' }, { id: 'native', available: true }],
}

const option = (overrides: Partial<ExportOptionDefinition>): ExportOptionDefinition => ({
  id: 'filename_stem',
  label: 'Filename stem',
  type: 'string',
  default: null,
  choices: [],
  description: '',
  ...overrides,
})

const capability = (overrides: Partial<ExportCapability>): ExportCapability => ({
  format: 'project_bundle',
  label: 'Project bundle',
  category: 'project',
  available: true,
  reason_code: null,
  reason: null,
  execution_backend: 'pytorch',
  runtime_api: null,
  conversion: 'canonical',
  layout: 'file',
  extension: '.neuro-weave.json',
  payload: 'project',
  inference_only: true,
  training_state: 'unavailable',
  compatibility: 'Canonical project representation',
  limitations: ['Initialized inference artifact; training state is unavailable.'],
  dependencies: [],
  options: [option({})],
  ...overrides,
})

const formats: ExportCapability[] = [
  capability({}),
  capability({ format: 'source', label: 'Generated source', category: 'source', extension: '.py', payload: 'source', compatibility: 'Generated native source' }),
  capability({ format: 'keras', label: 'Keras model', category: 'native', conversion: 'keras', extension: '.keras', payload: 'full_model', available: false, reason_code: 'keras_disabled', reason: 'Keras is disabled for this native profile.' }),
  capability({ format: 'pytorch_exported_program', label: 'PyTorch ExportedProgram', category: 'native', conversion: 'native', extension: '.pt2', payload: 'full_model', runtime_api: 'native', compatibility: 'Native PyTorch inference program' }),
  capability({
    format: 'onnx',
    label: 'ONNX interchange',
    category: 'interchange',
    conversion: 'native',
    extension: '.onnx',
    payload: 'full_model',
    runtime_api: 'native',
    compatibility: 'Named inference interchange',
    options: [
      option({}),
      option({ id: 'opset_version', label: 'ONNX opset', type: 'integer', default: 18, choices: [17, 18, 19, 20] }),
      option({ id: 'dynamic_batch', label: 'Dynamic batch', type: 'boolean', default: true }),
      option({ id: 'external_data', label: 'External data', type: 'choice', default: 'auto', choices: ['auto', 'always', 'never'] }),
    ],
  }),
  capability({ format: 'safetensors', label: 'SafeTensors package', category: 'checkpoint', conversion: 'native', layout: 'archive', extension: '.safetensors.zip', payload: 'weights_only', runtime_api: 'native', compatibility: 'Weights-only package' }),
]

const response: ExportCapabilitiesResponse = {
  project_id: project.id,
  project_revision: project.current_revision,
  execution_backend: project.execution_backend,
  keras_enabled: project.keras_enabled,
  valid: true,
  diagnostics: [],
  formats,
}

function resources(): AppResources {
  return {
    manifests: [],
    projects: [],
    backendStatuses: [backendStatus],
    runtimeApis: [{ id: 'native', available: true }, { id: 'keras', available: false }],
    compilationCapabilities: null,
    runtimePreferences: { pytorch: nativeWorker },
    runtimePreferenceBackend: 'pytorch',
    readiness: { catalog: true, projects: true, runtime: true },
    error: null,
    setRuntimePreferenceBackend: vi.fn(),
    setRuntimePreference: vi.fn(),
    refreshProjects: vi.fn(),
    clearError: vi.fn(),
  }
}

function renderPage() {
  mockedResources.mockReturnValue(resources())
  vi.spyOn(api, 'exportCapabilities').mockResolvedValue(response)
  render(<ModelExportContent project={project} />)
}

function job(overrides: Partial<JobResponse>): JobResponse {
  return {
    id: 'job-1',
    operation: 'model_export',
    state: 'running',
    worker: nativeWorker,
    device_lease: { device_id: 'cpu', state: 'running', job_id: 'job-1', pid: 100 },
    pid: 100,
    created_at: '2026-08-27T00:00:00Z',
    started_at: '2026-08-27T00:00:01Z',
    updated_at: '2026-08-27T00:00:01Z',
    finished_at: null,
    exit_code: null,
    cancellation_requested: false,
    forced_termination: false,
    restart_reason: 'new_job',
    progress: { phase: 'building', message: 'Building export', percent: 30 },
    diagnostics: {},
    result: null,
    artifact: null,
    error: null,
    ...overrides,
  }
}

describe('ModelExportPage', () => {
  beforeEach(() => {
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:export') })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('groups capabilities, shows explicit disabled reasons, and only reveals ONNX options for ONNX', async () => {
    renderPage()
    expect(await screen.findByRole('heading', { name: 'Project' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Source' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Native' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Interchange' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Weights / Checkpoint' })).toBeInTheDocument()

    const unavailable = screen.getByRole('button', { name: /Keras model/ })
    expect(unavailable).toBeDisabled()
    expect(unavailable).toHaveTextContent('Keras is disabled for this native profile.')
    expect(screen.queryByLabelText('ONNX opset')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /ONNX interchange/ }))
    expect(screen.getByLabelText('ONNX opset')).toHaveValue('18')
    expect(screen.getByLabelText('Dynamic batch dimensions')).toBeChecked()
    expect(screen.getByLabelText('External data')).toHaveValue('auto')
    expect(within(screen.getByRole('complementary')).queryByText('Weights only')).not.toBeInTheDocument()
  })

  it('polls a model export job and downloads the verified artifact on success', async () => {
    renderPage()
    await screen.findByRole('heading', { name: 'Project' })
    fireEvent.click(screen.getByRole('button', { name: /PyTorch ExportedProgram/ }))
    vi.spyOn(api, 'submitModelExport').mockResolvedValue(job({}))
    vi.spyOn(api, 'getJob').mockResolvedValue(job({ state: 'succeeded', progress: { phase: 'completed', message: 'Export completed', percent: 100 }, artifact: { format: 'pytorch_exported_program', filename: 'parity.pt2', layout: 'file', payload: 'full_model', conversion: 'native', execution_backend: 'pytorch', runtime_api: 'native', project_id: project.id, project_revision: 7, graph_sha256: 'a'.repeat(64), files: [{ path: 'parity.pt2', size_bytes: 5, sha256: 'b'.repeat(64), media_type: 'application/octet-stream' }], inputs: [], outputs: [], tool_versions: {}, verification: { structural: true, reload: true, inference: true, semantic_equivalence: 'verified', notes: [] } } }))
    vi.spyOn(api, 'jobArtifact').mockResolvedValue(new Blob(['artifact']))

    fireEvent.click(screen.getByRole('button', { name: 'Export artifact' }))
    await waitFor(() => expect(api.jobArtifact).toHaveBeenCalledWith('job-1'), { timeout: 2500 })
    expect(await screen.findByText('Downloaded verified parity.pt2.')).toBeInTheDocument()
    expect(URL.createObjectURL).toHaveBeenCalled()
  })

  it('keeps a failed job retryable and reports cancellation after stop', async () => {
    renderPage()
    await screen.findByRole('heading', { name: 'Project' })
    fireEvent.click(screen.getByRole('button', { name: /PyTorch ExportedProgram/ }))
    const submit = vi.spyOn(api, 'submitModelExport').mockResolvedValue(job({}))
    const getJob = vi.spyOn(api, 'getJob').mockResolvedValue(job({ state: 'failed', progress: { phase: 'validating', message: 'Export failed', percent: 80 }, error: { code: 'verification_failed', message: 'Inference mismatch', details: {} } }))

    fireEvent.click(screen.getByRole('button', { name: 'Export artifact' }))
    expect(await screen.findByText(/Inference mismatch/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry export' })).toBeEnabled()

    getJob.mockResolvedValueOnce(job({ state: 'running' })).mockResolvedValueOnce(job({ state: 'terminated', progress: { phase: 'completed', message: 'Cancelled', percent: null } }))
    vi.spyOn(api, 'stopJob').mockResolvedValue(job({ state: 'stop_requested', cancellation_requested: true, progress: { phase: 'stopping', message: 'Stopping export', percent: null } }))
    fireEvent.click(screen.getByRole('button', { name: 'Retry export' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Cancel export' })).toBeInTheDocument(), { timeout: 1000 })
    fireEvent.click(screen.getByRole('button', { name: 'Cancel export' }))
    expect(await screen.findByText(/Cancellation requested/)).toBeInTheDocument()
    expect(submit).toHaveBeenCalledTimes(2)
  })
})
