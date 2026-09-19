export type ExecutionBackend = 'tensorflow' | 'pytorch' | 'jax'
export type RuntimeApi = 'keras' | 'native'
export type RuntimeDiscoveryState = 'unchecked' | 'checking' | 'verified' | 'unavailable' | 'failed'
export type MemoryPolicyKind = 'default' | 'tensorflow_growth' | 'tensorflow_limit_mb' | 'jax_no_preallocation' | 'jax_preallocate_fraction' | 'jax_platform_allocator' | 'pytorch_fraction'
export type DiagnosticSeverity = 'error' | 'warning' | 'compatibility' | 'info'
export type ValidationState = 'not-run' | 'checking' | 'stale' | 'ready' | 'blocked' | 'error'
export type PortKind = 'tensor' | 'tensor_list' | 'scalar' | 'mask'

export interface PortCardinality {
  kind: 'single' | 'parameter'
  parameter_id?: string | null
  minimum: number
  maximum: number
  id_pattern: string
  display_name_pattern: string
}

export interface PortDefinition {
  id: string
  display_name: string
  kind: PortKind
  required: boolean
  accepts_multiple: boolean
  rank?: number | null
  element_kind?: PortKind | null
  description: string
  enabled_when?: Record<string, unknown>
  cardinality?: PortCardinality
}

export type ParameterType =
  | 'string'
  | 'integer'
  | 'number'
  | 'boolean'
  | 'enum'
  | 'shape'
  | 'axis'
  | 'expression'
  | 'list'
  | 'tuple'
  | 'object'
  | 'dtype'
  | 'activation'
  | 'initializer'
  | 'regularizer'
  | 'constraint'

export interface ParameterValueSchema {
  type: ParameterType
  item?: ParameterValueSchema | null
  properties?: Record<string, ParameterValueSchema>
  required_properties?: string[]
}

export interface ParameterDefinition {
  id: string
  display_name: string
  type: ParameterType
  default: unknown
  required: boolean
  constraints: Record<string, unknown>
  options: unknown[]
  description: string
  group?: string
  nullable?: boolean
  ui?: Record<string, unknown>
  value_schema?: ParameterValueSchema | null
  visible_when?: Record<string, unknown>
  enabled_when?: Record<string, unknown>
  required_when?: Record<string, unknown>
  framework_default?: unknown
  editor_default?: unknown
}

export type CapabilityStatus = 'verified' | 'unsupported' | 'unverified'

export interface ParameterCapability {
  supported: boolean
  status: CapabilityStatus
  allowed_values?: unknown[] | null
  constraints: Record<string, unknown>
  mapping?: string | null
  notes?: string | null
  default?: unknown
}

export interface ImplementationMapping {
  supported: boolean
  status: CapabilityStatus
  symbol?: string
  adapter?: string
  notes?: string
  parameters?: Record<string, ParameterCapability>
  variant_symbols?: Record<string, string>
  execution_backends?: Record<string, boolean>
}

export type OperationFamily = 'unary' | 'binary' | 'elementwise' | 'reduction' | 'tensor'

export interface OperationMetadata {
  family: OperationFamily
  shape_behavior: 'preserve' | 'reduce' | 'reshape' | 'transpose' | 'expand_dims' | 'squeeze' | 'flatten' | 'repeat' | 'tile' | 'cast' | 'concatenate' | 'stack' | 'matmul' | 'take' | 'expression'
  output_dtype: 'same' | 'bool' | 'int64' | 'parameter'
  arity: number
  variadic: boolean
  arguments: string[]
  keywords: string[]
}

export interface NodeManifest {
  id: string
  version: number
  display_name: string
  category: string
  description: string
  semantic_role?: 'input' | 'operation' | 'output_adapter'
  task_hint?: string | null
  inputs: PortDefinition[]
  outputs: PortDefinition[]
  parameters: ParameterDefinition[]
  shape_rule: { kind: string; config: Record<string, unknown> }
  operation?: OperationMetadata | null
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
  locked: boolean
}

export interface GraphEdgeInstance {
  id: string
  source_node_id: string
  source_port_id: string
  target_node_id: string
  target_port_id: string
}

export interface OutputBinding {
  node_id: string
  port_id: string
}

export interface KerasCompilationConfiguration {
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

export interface TensorFlowNativeConfiguration {
  jit_compile: boolean
}

export interface PyTorchNativeConfiguration {
  optimizer: string
  optimizer_parameters: Record<string, unknown>
  losses: Record<string, string>
  metrics: Record<string, string[]>
  compile_enabled: boolean
  mode: 'default' | 'reduce-overhead' | 'max-autotune'
  fullgraph: boolean
  dynamic: boolean
}

export interface JaxNativeConfiguration {
  jit_enabled: boolean
}

export interface NativeCompilationConfiguration {
  tensorflow: TensorFlowNativeConfiguration
  pytorch: PyTorchNativeConfiguration
  jax: JaxNativeConfiguration
}

export interface ModelConfiguration {
  keras: KerasCompilationConfiguration
  native: NativeCompilationConfiguration
}

export type CompilationParameterType = 'number' | 'integer' | 'boolean' | 'enum'

export interface CompilationParameter {
  id: string
  label: string
  type: CompilationParameterType
  default: number | boolean | string
  description: string
  options?: string[]
  constraints: Record<string, number | boolean | string>
}

export interface CompilationOption {
  id: string
  label: string
  description: string
  parameters: CompilationParameter[]
  formula?: string | null
  implementations?: Record<string, CompilationImplementation>
  supported_tasks?: Array<'regression' | 'binary_classification' | 'multiclass_classification'>
  target_contract?: 'continuous_same_shape' | 'binary_same_shape' | 'one_hot_same_shape' | 'class_index' | 'probability_distribution_same_shape' | null
  activation_contract?: string[]
  origin?: 'builtin' | 'custom'
  locked?: boolean
  can_edit?: boolean
  can_delete?: boolean
  created_at?: string | null
  updated_at?: string | null
}

export interface CompilationImplementation {
  supported: boolean
  symbol?: string | null
  adapter?: string | null
  parameter_mapping?: Record<string, string>
  notes?: string | null
}

export interface CompilationCatalog {
  optimizers: CompilationOption[]
  losses: CompilationOption[]
  metrics: CompilationOption[]
  initializers: string[]
  regularizers: string[]
  constraints: string[]
}

export interface NativeCompilationCapability {
  backend: ExecutionBackend
  compiler_id: string
  available: boolean
  compiler: string
  reason_code?: string | null
  reason?: string | null
  parameters: CompilationParameter[]
  catalog?: CompilationCatalog
}

export interface CompilationCapabilities {
  keras: CompilationCatalog
  native: Record<ExecutionBackend, NativeCompilationCapability>
}

export interface ModelGraph {
  schema_version: 5
  nodes: GraphNodeInstance[]
  edges: GraphEdgeInstance[]
  output_bindings: OutputBinding[]
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
  input_count: number
  output_count: number
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
  effective_outputs?: OutputReference[]
  output_mode?: 'explicit' | 'inferred'
  model_type?: ModelTypeInference
}

export interface MetricDefinition {
  id: string
  kind: 'metric' | 'loss'
  label: string
  description: string
  formula?: string | null
  origin: 'builtin' | 'custom'
  locked: boolean
  can_edit: boolean
  can_delete: boolean
  supported_tasks: Array<'regression' | 'binary_classification' | 'multiclass_classification'>
  target_contract?: CompilationOption['target_contract']
  created_at?: string | null
  updated_at?: string | null
}

export interface MetricCatalogResponse {
  definitions: MetricDefinition[]
}

export interface ConnectionValidationResult {
  valid: boolean
  diagnostics: ValidationDiagnostic[]
}

export interface OutputReference {
  node_id: string
  port_id: string
  name: string
  explicit: boolean
  task_hint?: string | null
}

export interface ModelTypeInference {
  kind: 'binary_classification' | 'multiclass_classification' | 'multilabel_classification' | 'regression' | 'embedding' | 'sequence' | 'multi_output' | 'generic'
  confidence: 'high' | 'medium' | 'low'
  evidence: string[]
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
    compilation_capabilities: CompilationCapabilities
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
  native_compilers?: Record<string, boolean>
  discovery_state?: RuntimeDiscoveryState
  discovery_source?: 'snapshot' | 'live' | 'none' | string
  last_checked_at?: string | null
}

export interface RuntimeApiStatus {
  id: RuntimeApi
  available: boolean
  reason_code?: string | null
  state?: RuntimeDiscoveryState
  source?: 'snapshot' | 'live' | 'none' | string
  checked_at?: string | null
}

export interface RuntimeDiscoveryResponse {
  backend: ExecutionBackend
  api: RuntimeApi
  state: RuntimeDiscoveryState
  available: boolean
  reason_code?: string | null
  source: 'snapshot' | 'live' | 'none' | string
  checked_at?: string | null
  message?: string | null
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
  compilation?: ModelConfiguration
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

export type JobOperation = 'smoke_inference' | 'tensor_inspection' | 'model_export'
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

export type TensorStorageMode = 'full' | 'sampled' | 'summary_only'
export type TensorDataMode = 'exact' | 'sampled' | 'downsampled'

export interface TensorStatistics {
  mode: TensorDataMode
  count: number
  minimum?: number | null
  maximum?: number | null
  mean?: number | null
  standard_deviation?: number | null
  median?: number | null
  zero_proportion?: number | null
  nan_count: number
  infinite_count: number
}

export interface TensorHistogram {
  mode: TensorDataMode
  bins: number[]
  counts: number[]
}

export interface TensorDescriptor {
  id: string
  storage_id: string
  node_id: string
  direction: 'input' | 'output'
  port_id: string
  item_index?: number | null
  shape: number[]
  dtype: string
  element_count: number
  storage_mode: TensorStorageMode
  value_count: number
  statistics?: TensorStatistics | null
  histogram?: TensorHistogram | null
}

export interface TensorInspectionNode {
  node_id: string
  node_name: string
  node_type: string
  topological_index: number
  depth: number
  predecessors: string[]
  successors: string[]
  inspectable: boolean
  unavailable_reason?: string | null
  inputs: TensorDescriptor[]
  outputs: TensorDescriptor[]
}

export interface TensorInspectionSession {
  id: string
  project_id: string
  project_revision: number
  execution_backend: ExecutionBackend
  runtime_api: RuntimeApi
  worker: WorkerConfiguration
  seed: number
  weight_source: 'initialized_untrained'
  input_description: string
  compiler_mode: 'normal' | 'bypassed_for_inspection'
  created_at: string
  expires_at: string
  nodes: TensorInspectionNode[]
}

export interface TensorInspectionViewRequest {
  tensor_id: string
  fixed_indices?: Record<number, number>
  display_axes?: number[]
  offset?: number
  limit?: number
}

export interface TensorInspectionViewResponse {
  tensor_id: string
  shape: number[]
  dtype: string
  display_axes: number[]
  fixed_indices: Record<number, number>
  values: unknown
  view_shape: number[]
  mode: TensorDataMode
  total_values: number
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
  compilation?: NativeCompilationSummary | null
}

export interface NativeCompilationSummary {
  verified: boolean
  optimizer: { id?: string | null; instantiated: boolean }
  losses: Record<string, unknown>
  metrics: Record<string, unknown>
  training: 'not_run'
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

export type CompatibilityReasonCode =
  | 'missing_implementation'
  | 'unsupported_implementation'
  | 'unverified_implementation'
  | 'backend_excluded'
  | 'unsupported_parameter'
  | 'unverified_parameter'
  | 'unsupported_parameter_option'
  | 'invalid_parameter_value'
  | 'parameter_fixed'
  | 'disabled_parameter'

export interface ImplementationCompatibility {
  available: boolean
  reasonCode: CompatibilityReasonCode | null
  reason: string | null
  mapping?: ImplementationMapping
}

export interface ParameterCompatibility {
  editable: boolean
  valueCompatible: boolean
  reasonCode: CompatibilityReasonCode | null
  reason: string | null
  capability?: ParameterCapability
  effectiveDefault: unknown
}

export function resolveImplementationCompatibility(
  manifest: (Pick<NodeManifest, 'implementations'> & Partial<Pick<NodeManifest, 'display_name'>>) | undefined,
  backend: ExecutionBackend,
  kerasEnabled: boolean,
): ImplementationCompatibility {
  const mapping = manifest?.implementations[implementationKey(backend, kerasEnabled)]
  const label = implementationLabel(backend, kerasEnabled)
  if (!mapping) {
    return {
      available: false,
      reasonCode: 'missing_implementation',
      reason: `${manifest?.display_name ?? 'This node'} has no explicit ${label} adapter.`,
    }
  }
  if (mapping.supported === false || mapping.status === 'unsupported') {
    return {
      available: false,
      reasonCode: 'unsupported_implementation',
      reason: `${manifest?.display_name ?? 'This node'} is not supported by ${label}.`,
      mapping,
    }
  }
  if (mapping.status !== 'verified') {
    return {
      available: false,
      reasonCode: 'unverified_implementation',
      reason: `${manifest?.display_name ?? 'This node'} has not been verified for ${label}.`,
      mapping,
    }
  }
  if (kerasEnabled && mapping.execution_backends?.[backend] === false) {
    return {
      available: false,
      reasonCode: 'backend_excluded',
      reason: `${manifest?.display_name ?? 'This node'} is not available through Keras on the ${executionBackendShortLabels[backend]} backend.`,
      mapping,
    }
  }
  return { available: true, reasonCode: null, reason: null, mapping }
}

export function effectiveParameterDefault(
  definition: ParameterDefinition,
  capability?: ParameterCapability,
): unknown {
  if (capability?.default !== undefined && capability.default !== null) return capability.default
  if (Object.prototype.hasOwnProperty.call(definition, 'framework_default')) return definition.framework_default
  return definition.default
}

/**
 * Validate the structural part of a parameter value without applying the
 * selected implementation's allowed-value mapping. The server performs the
 * authoritative validation; this predicate keeps direct editor mutations
 * from creating values that the editor already knows are malformed.
 */
export function parameterValueMatchesDefinition(
  definition: ParameterDefinition,
  value: unknown,
  capability?: ParameterCapability,
): boolean {
  if (value === undefined) return false
  if (value === null) return Boolean(definition.nullable)
  if (definition.value_schema && !valueMatchesSchema(definition.value_schema, value)) return false

  let typeMatches = true
  if (!definition.value_schema) {
    switch (definition.type) {
      case 'boolean': typeMatches = typeof value === 'boolean'; break
      case 'integer': typeMatches = Number.isSafeInteger(value) && typeof value === 'number'; break
      case 'axis':
        typeMatches = (Number.isSafeInteger(value) && typeof value === 'number')
          || (Array.isArray(value) && value.every((item) => Number.isSafeInteger(item) && typeof item === 'number'))
        break
      case 'number': typeMatches = typeof value === 'number' && Number.isFinite(value); break
      case 'enum': typeMatches = !definition.options.length || definition.options.some((option) => sameValue(option, value)); break
      case 'string':
      case 'expression':
      case 'dtype':
      case 'activation': typeMatches = typeof value === 'string'; break
      case 'initializer':
      case 'regularizer':
      case 'constraint': typeMatches = typeof value === 'string' || isRecord(value); break
      case 'object': typeMatches = isRecord(value); break
      case 'shape':
        typeMatches = Array.isArray(value) && value.every((item) => (
          item === null
          || Number.isSafeInteger(item) && typeof item === 'number'
          || Array.isArray(item)
        )); break
      case 'list':
      case 'tuple': typeMatches = Array.isArray(value); break
    }
  }
  if (!typeMatches) return false

  if (definition.options.length && !isRecord(value)
    && !definition.options.some((option) => sameValue(option, value))) return false

  const constraints = { ...definition.constraints, ...(capability?.constraints ?? {}) }
  if (typeof value === 'number' && !Number.isFinite(value)) return false
  if (typeof value === 'number') {
    if (typeof constraints['min'] === 'number' && value < constraints['min']) return false
    if (typeof constraints['max'] === 'number' && value > constraints['max']) return false
  }
  if (Array.isArray(value)) {
    if (typeof constraints['length'] === 'number' && value.length !== constraints['length']) return false
    if (typeof constraints['min_length'] === 'number' && value.length < constraints['min_length']) return false
    if (typeof constraints['max_length'] === 'number' && value.length > constraints['max_length']) return false
    const numericItems = value.filter((item): item is number => typeof item === 'number' && !Number.isNaN(item))
    const itemMin = constraints['item_min']
    const itemMax = constraints['item_max']
    if (typeof itemMin === 'number' && numericItems.some((item) => item < itemMin)) return false
    if (typeof itemMax === 'number' && numericItems.some((item) => item > itemMax)) return false
    if (definition.type === 'shape' && value.some((item) => item === 0)) return false
  }
  return true
}

export function resolveParameterCompatibility(
  manifest: (Pick<NodeManifest, 'implementations'> & Partial<Pick<NodeManifest, 'display_name'>>) | undefined,
  definition: ParameterDefinition,
  value: unknown,
  parameters: Record<string, unknown>,
  backend: ExecutionBackend,
  kerasEnabled: boolean,
): ParameterCompatibility {
  const implementation = resolveImplementationCompatibility(manifest, backend, kerasEnabled)
  const capability = implementation.mapping?.parameters?.[definition.id]
  const effectiveDefault = effectiveParameterDefault(definition, capability)
  if (!implementation.available) {
    return {
      editable: false,
      valueCompatible: false,
      reasonCode: implementation.reasonCode,
      reason: implementation.reason,
      capability,
      effectiveDefault,
    }
  }

  const label = implementationLabel(backend, kerasEnabled)
  const unsupported = capability?.supported === false || capability?.status === 'unsupported'
  if (capability && !unsupported && capability.status !== 'verified') {
    return {
      editable: false,
      valueCompatible: false,
      reasonCode: 'unverified_parameter',
      reason: `${definition.display_name} has not been verified for ${label}.`,
      capability,
      effectiveDefault,
    }
  }

  const allowedValues = capability?.allowed_values
  const allowed = !allowedValues || allowedValues.some((item) => sameValue(item, value))
  if (!parameterValueMatchesDefinition(definition, value, capability)) {
    return {
      editable: true,
      valueCompatible: false,
      reasonCode: 'invalid_parameter_value',
      reason: `${definition.display_name} has an invalid value for ${label}.`,
      capability,
      effectiveDefault,
    }
  }
  const enabled = conditionMatches(definition.enabled_when, parameters)
  const valueCompatible = !unsupported || (
    sameValue(value, effectiveDefault) && allowed
  )
  const conditionCompatible = enabled || sameValue(value, effectiveDefault)
  if (unsupported) {
    return {
      editable: false,
      valueCompatible: valueCompatible && conditionCompatible,
      reasonCode: 'unsupported_parameter',
      reason: `${definition.display_name} is not supported for ${label}.`,
      capability,
      effectiveDefault,
    }
  }
  if (!enabled) {
    return {
      editable: false,
      valueCompatible: conditionCompatible && allowed,
      reasonCode: 'disabled_parameter',
      reason: `${definition.display_name} is disabled by the current node configuration.`,
      capability,
      effectiveDefault,
    }
  }
  if (!allowed) {
    return {
      editable: !allowedValues || allowedValues.length > 1,
      valueCompatible: false,
      reasonCode: 'unsupported_parameter_option',
      reason: `${definition.display_name} has no supported ${label} mapping for this value.`,
      capability,
      effectiveDefault,
    }
  }
  if (allowedValues?.length === 1) {
    return {
      editable: false,
      valueCompatible: true,
      reasonCode: 'parameter_fixed',
      reason: `Fixed to ${String(allowedValues[0])} by this backend.`,
      capability,
      effectiveDefault,
    }
  }
  return {
    editable: true,
    valueCompatible: true,
    reasonCode: null,
    reason: null,
    capability,
    effectiveDefault,
  }
}

export function conditionMatches(
  condition: Record<string, unknown> | undefined,
  parameters: Record<string, unknown>,
): boolean {
  if (!condition) return true
  return Object.entries(condition).every(([parameterId, expected]) => {
    const actual = parameters[parameterId]
    if (typeof expected === 'object' && expected !== null && !Array.isArray(expected)) {
      const rule = expected as Record<string, unknown>
      if ('equals' in rule && !sameValue(actual, rule['equals'])) return false
      if ('not_equals' in rule && sameValue(actual, rule['not_equals'])) return false
      if ('in' in rule && Array.isArray(rule['in']) && !rule['in'].some((item) => sameValue(actual, item))) return false
      if ('not_in' in rule && Array.isArray(rule['not_in']) && rule['not_in'].some((item) => sameValue(actual, item))) return false
      return true
    }
    if (Array.isArray(expected)) return expected.some((item) => sameValue(actual, item))
    return sameValue(actual, expected)
  })
}

function sameValue(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function valueMatchesSchema(schema: ParameterValueSchema, value: unknown): boolean {
  if (value === null) return false
  if (schema.type === 'object') {
    if (!isRecord(value)) return false
    if ((schema.required_properties ?? []).some((key) => !Object.prototype.hasOwnProperty.call(value, key))) return false
    return Object.entries(schema.properties ?? {}).every(([key, propertySchema]) => (
      !Object.prototype.hasOwnProperty.call(value, key) || valueMatchesSchema(propertySchema, value[key])
    ))
  }
  if (schema.type === 'list' || schema.type === 'tuple' || schema.type === 'shape') {
    return Array.isArray(value) && (!schema.item || value.every((item) => valueMatchesSchema(schema.item as ParameterValueSchema, item)))
  }
  if (schema.type === 'integer' || schema.type === 'axis') {
    return typeof value === 'number' && Number.isSafeInteger(value)
  }
  if (schema.type === 'number') return typeof value === 'number' && Number.isFinite(value)
  if (schema.type === 'boolean') return typeof value === 'boolean'
  if (schema.type === 'string' || schema.type === 'enum' || schema.type === 'expression' || schema.type === 'dtype' || schema.type === 'activation') return typeof value === 'string'
  if (schema.type === 'initializer' || schema.type === 'regularizer' || schema.type === 'constraint') return typeof value === 'string' || isRecord(value)
  return true
}

export interface ProjectExportBundle {
  format: 'neuro-weave-project'
  format_version: 6
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
  keras: {
    optimizer: 'adam',
    optimizer_parameters: { learning_rate: 0.001 },
    losses: { __default__: 'categorical_crossentropy' },
    metrics: { __default__: ['accuracy'] },
    default_initializer: 'glorot_uniform',
    default_regularizer: 'none',
    default_constraint: 'none',
    mixed_precision: false,
    random_seed: 42,
  },
  native: {
    tensorflow: { jit_compile: false },
    pytorch: {
      optimizer: 'adam',
      optimizer_parameters: { learning_rate: 0.001 },
      losses: {},
      metrics: {},
      compile_enabled: false,
      mode: 'default',
      fullgraph: false,
      dynamic: false,
    },
    jax: { jit_enabled: false },
  },
})
