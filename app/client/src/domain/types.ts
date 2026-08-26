import type { Edge, Node, XYPosition } from '@xyflow/react'

export type ExecutionBackend = 'tensorflow' | 'pytorch' | 'jax'
export type RuntimeApi = 'keras' | 'native'
export type MemoryPolicyKind = 'default' | 'tensorflow_growth' | 'tensorflow_limit_mb' | 'jax_no_preallocation' | 'jax_preallocate_fraction' | 'jax_platform_allocator' | 'pytorch_fraction'
export type DiagnosticSeverity = 'error' | 'warning' | 'compatibility' | 'info'
export type PortKind = 'tensor' | 'tensor_list' | 'scalar' | 'mask'

export interface PortDefinition {
  id: string
  display_name: string
  kind: PortKind
  required: boolean
  accepts_multiple: boolean
  rank?: number | null
  element_kind?: PortKind | null
  description: string
}

export interface ParameterDefinition {
  id: string
  display_name: string
  type: 'string' | 'integer' | 'number' | 'boolean' | 'enum' | 'shape' | 'axis' | 'expression' | 'list' | 'tuple'
  default: unknown
  required: boolean
  constraints: Record<string, unknown>
  options: unknown[]
  description: string
  group?: string
  nullable?: boolean
  ui?: Record<string, unknown>
}

export interface ParameterCapability {
  supported: boolean
  allowed_values?: unknown[] | null
  constraints: Record<string, unknown>
  mapping?: string | null
  notes?: string | null
}

export interface ImplementationMapping {
  supported: boolean
  symbol?: string
  notes?: string
  parameters?: Record<string, ParameterCapability>
  execution_backends?: Record<string, boolean>
}

export interface NodeManifest {
  id: string
  version: number
  display_name: string
  category: string
  description: string
  inputs: PortDefinition[]
  outputs: PortDefinition[]
  parameters: ParameterDefinition[]
  shape_rule: { kind: string; config: Record<string, unknown> }
  implementations: Record<string, ImplementationMapping>
  ui: { icon: string; accent: string; width: number; height: number }
  deprecated: boolean
  documentation_ref: string
}

export interface Position {
  x: number
  y: number
}

export interface GraphNodeInstance {
  id: string
  definition_id: string
  definition_version: number
  label?: string | null
  position: Position
  parameters: Record<string, unknown>
}

export interface GraphEdgeInstance {
  id: string
  source_node_id: string
  source_port_id: string
  target_node_id: string
  target_port_id: string
}

export interface ModelConfiguration {
  optimizer: string
  optimizer_parameters: Record<string, unknown>
  losses: Record<string, string>
  metrics: Record<string, string[]>
  default_initializer: string
  default_regularizer: string
  default_constraint: string
  mixed_precision: boolean
  random_seed?: number | null
}

export type CompilationParameterType = 'number' | 'integer' | 'boolean'

export interface CompilationParameter {
  id: string
  label: string
  type: CompilationParameterType
  default: number | boolean
  description: string
  constraints: Record<string, number | boolean>
}

export interface CompilationOption {
  id: string
  label: string
  description: string
  parameters: CompilationParameter[]
  formula?: string | null
}

export interface CompilationCatalog {
  optimizers: CompilationOption[]
  losses: CompilationOption[]
  metrics: CompilationOption[]
  initializers: string[]
  regularizers: string[]
  constraints: string[]
}

export interface ModelGraph {
  schema_version: number
  nodes: GraphNodeInstance[]
  edges: GraphEdgeInstance[]
  model_configuration: ModelConfiguration
  export_settings: {
    include_comments: boolean
    include_shape_assertions: boolean
    preferred_binary_format: string
  }
  future_training_configuration: Record<string, unknown>
}

export interface ProjectDocument {
  id: string
  name: string
  description: string
  created_at: string
  updated_at: string
  schema_version: number
  execution_backend: ExecutionBackend
  keras_enabled: boolean
  current_revision: number
  graph: ModelGraph
  validation_status: string
  generated_source?: string | null
}

export interface ProjectSummary {
  id: string
  name: string
  description: string
  updated_at: string
  execution_backend: ExecutionBackend
  keras_enabled: boolean
  current_revision: number
  validation_status: string
  node_count: number
  edge_count: number
  has_generated_source: boolean
}

export interface ValidationDiagnostic {
  code: string
  severity: DiagnosticSeverity
  message: string
  node_id?: string | null
  edge_id?: string | null
  parameter_id?: string | null
  execution_backend?: ExecutionBackend | null
  keras_enabled?: boolean | null
  details: Record<string, unknown>
}

export interface TensorSpec {
  shape: Array<number | null | { name: string }>
  dtype: string
  name?: string | null
}

export interface ShapeSummary {
  node_id: string
  outputs: Record<string, TensorSpec>
  parameter_count?: number | null
}

export interface ValidationResult {
  valid: boolean
  diagnostics: ValidationDiagnostic[]
  shape_summaries: ShapeSummary[]
  total_parameters?: number | null
}

export interface CompilationResult {
  valid: boolean
  execution_backend: ExecutionBackend
  keras_enabled: boolean
  diagnostics: ValidationDiagnostic[]
  code?: string | null
}

export interface MetadataResponse {
  name: string
  version: string
  schema_version: number
  runtime_apis?: RuntimeApiStatus[]
  execution_backends: ExecutionBackendStatus[]
  configuration: {
    compilation_catalog: CompilationCatalog
    optimizers: string[]
    losses: string[]
    metrics: string[]
    initializers: string[]
    regularizers: string[]
  }
}

export interface ExecutionBackendStatus {
  id: ExecutionBackend
  installed: boolean
  version?: string | null
  available: boolean
  reason_code?: string | null
  accelerator?: string | null
  device_name?: string | null
  devices?: RuntimeDevice[]
  runtime_apis?: RuntimeApiStatus[]
}

export interface RuntimeApiStatus {
  id: RuntimeApi
  available: boolean
  reason_code?: string | null
}

export interface RuntimeDevice {
  device_id: string
  lease_key: string
  kind: string
  index?: number | null
  name: string
  accelerator: string
  available: boolean
  total_memory_bytes?: number | null
  supported_memory_policies: MemoryPolicyKind[]
}

export interface MemoryPolicy {
  kind: MemoryPolicyKind
  limit_mb?: number | null
  fraction?: number | null
}

export interface WorkerConfiguration {
  api: RuntimeApi
  backend: ExecutionBackend
  device_id: string
  memory_policy: MemoryPolicy
  precision: 'float32' | 'mixed_float16' | 'mixed_bfloat16'
}

export type ExportFormat =
  | 'project_bundle'
  | 'source'
  | 'keras'
  | 'keras_weights'
  | 'tensorflow_saved_model'
  | 'onnx'
  | 'pytorch_exported_program'
  | 'torchscript'
  | 'safetensors'
  | 'tensorflow_checkpoint'
  | 'flax_checkpoint'

export type ExportCategory = 'project' | 'source' | 'native' | 'interchange' | 'checkpoint'
export type ExportLayout = 'file' | 'archive'
export type ExportPayload = 'project' | 'source' | 'full_model' | 'weights_only'
export type ExportConversion = 'canonical' | 'native' | 'keras' | 'bridge'
export type ExportOptionType = 'string' | 'integer' | 'boolean' | 'choice' | 'number'

export interface ExportVerification {
  structural: boolean
  reload: boolean
  inference: boolean
  semantic_equivalence: 'verified' | 'not_claimed' | 'not_applicable'
  notes: string[]
}

export interface ExportArtifactFile {
  path: string
  size_bytes: number
  sha256: string
  media_type: string
}

export interface ExportArtifactManifest {
  format: ExportFormat
  filename: string
  layout: ExportLayout
  payload: ExportPayload
  conversion: ExportConversion
  execution_backend: ExecutionBackend
  runtime_api?: RuntimeApi | null
  project_id?: string | null
  project_revision?: number | null
  graph_sha256?: string | null
  files: ExportArtifactFile[]
  inputs: Array<Record<string, unknown>>
  outputs: Array<Record<string, unknown>>
  tool_versions: Record<string, string>
  verification: ExportVerification
}

export interface ExportOptionDefinition {
  id: string
  label: string
  type: ExportOptionType
  default?: unknown
  choices: unknown[]
  description: string
}

export interface ExportCapability {
  format: ExportFormat
  label: string
  category: ExportCategory
  available: boolean
  reason_code?: string | null
  reason?: string | null
  execution_backend?: ExecutionBackend | null
  runtime_api?: RuntimeApi | null
  conversion: ExportConversion
  layout: ExportLayout
  extension: string
  payload: ExportPayload
  inference_only: boolean
  training_state: 'unavailable' | 'not_preserved'
  compatibility: string
  limitations: string[]
  dependencies: string[]
  options: ExportOptionDefinition[]
}

export interface ExportCapabilitiesResponse {
  project_id: string
  project_revision: number
  execution_backend: ExecutionBackend
  keras_enabled: boolean
  valid: boolean
  diagnostics: Array<Record<string, unknown>>
  formats: ExportCapability[]
}

export interface ExportOptions {
  opset_version: number
  dynamic_batch: boolean
  external_data: 'auto' | 'always' | 'never'
  max_shard_size_gb?: number | null
}

export type JobOperation = 'smoke_inference' | 'model_export'
export type JobState = 'starting' | 'running' | 'stop_requested' | 'stopping' | 'succeeded' | 'terminated' | 'failed'

export interface JobProgress {
  phase: string
  message: string
  percent?: number | null
}

export interface JobDiagnostics {
  process_rss_bytes?: number | null
  framework_allocated_bytes?: number | null
  framework_reserved_bytes?: number | null
  framework_peak_bytes?: number | null
  device_used_bytes?: number | null
  device_total_bytes?: number | null
}

export interface JobError {
  code: string
  message: string
  details?: Record<string, unknown>
}

export interface JobResponse {
  id: string
  operation: JobOperation
  state: JobState
  worker: WorkerConfiguration
  device_lease: {
    device_id: string
    state: 'idle' | 'starting' | 'running' | 'stop_requested' | 'stopping' | 'failed'
    job_id?: string | null
    pid?: number | null
  }
  pid?: number | null
  created_at: string
  started_at?: string | null
  updated_at: string
  finished_at?: string | null
  exit_code?: number | null
  cancellation_requested: boolean
  forced_termination: boolean
  restart_reason: string
  progress: JobProgress
  diagnostics: JobDiagnostics
  result?: Record<string, unknown> | null
  artifact?: ExportArtifactManifest | null
  error?: JobError | null
}

export interface TensorExecutionInfo {
  node_id: string
  shape: number[]
  dtype: string
}

export interface SmokeTestResponse {
  status: 'succeeded'
  execution_backend: ExecutionBackend
  accelerator: string
  device_name: string
  inputs: TensorExecutionInfo[]
  outputs: TensorExecutionInfo[]
  parameter_count: number
  duration_ms: number
}

export interface SmokeTestErrorDetail {
  code: string
  message: string
  backend?: string
  reason_code?: string
  diagnostics?: ValidationDiagnostic[]
}

export const executionBackendLabels: Record<ExecutionBackend, string> = {
  tensorflow: 'Native TensorFlow',
  pytorch: 'Native PyTorch',
  jax: 'Native JAX / Flax',
}

export const executionBackendShortLabels: Record<ExecutionBackend, string> = {
  tensorflow: 'TensorFlow',
  pytorch: 'PyTorch',
  jax: 'JAX / Flax',
}

export function implementationLabel(backend: ExecutionBackend, kerasEnabled: boolean): string {
  return kerasEnabled ? `Keras · ${executionBackendShortLabels[backend]} backend` : executionBackendLabels[backend]
}

export function implementationKey(backend: ExecutionBackend, kerasEnabled: boolean): string {
  return kerasEnabled ? 'keras' : backend
}

export interface EditorNodeData extends Record<string, unknown> {
  manifest: NodeManifest
  parameters: Record<string, unknown>
  label: string
  shapeSummary?: ShapeSummary
}

export type EditorNode = Node<EditorNodeData, 'modelNode'>
export type EditorEdge = Edge

export interface ProjectExportBundle {
  format: 'neuro-weave-project'
  format_version: 3
  exported_at: string
  project: {
    name: string
    description: string
    execution_backend: ExecutionBackend
    keras_enabled: boolean
    graph: ModelGraph
  }
}

export interface KerasArchitectureImportResponse {
  project: ProjectDocument
  warnings: string[]
}

export const emptyModelConfiguration = (): ModelConfiguration => ({
  optimizer: 'adam',
  optimizer_parameters: { learning_rate: 0.001 },
  losses: { __default__: 'categorical_crossentropy' },
  metrics: { __default__: ['accuracy'] },
  default_initializer: 'glorot_uniform',
  default_regularizer: 'none',
  default_constraint: 'none',
  mixed_precision: false,
  random_seed: 42,
})

export const editorNodeFromGraph = (node: GraphNodeInstance, manifest: NodeManifest): EditorNode => ({
  id: node.id,
  type: 'modelNode',
  position: node.position as XYPosition,
  data: {
    manifest,
    parameters: node.parameters,
    label: node.label ?? manifest.display_name,
  },
})
