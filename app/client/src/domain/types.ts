import type { Edge, Node, XYPosition } from '@xyflow/react'

export type CodeTarget = 'keras' | 'tensorflow' | 'pytorch' | 'jax'
export type ExecutionBackend = 'tensorflow' | 'pytorch' | 'jax'
export type DiagnosticSeverity = 'error' | 'warning' | 'compatibility' | 'info'
export type PortKind = 'tensor' | 'tensor_list' | 'scalar' | 'mask'

export interface PortDefinition {
  id: string
  display_name: string
  kind: PortKind
  required: boolean
  accepts_multiple: boolean
  rank?: number | null
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
  frameworks: Record<string, { supported: boolean; symbol?: string; notes?: string }>
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
  code_target: CodeTarget
  current_revision: number
  graph: ModelGraph
  validation_status: string
  generated_code_snapshots: Record<string, string>
}

export interface ProjectSummary {
  id: string
  name: string
  description: string
  updated_at: string
  code_target: CodeTarget
  current_revision: number
  validation_status: string
  node_count: number
  edge_count: number
  compiled_targets: CodeTarget[]
}

export interface ValidationDiagnostic {
  code: string
  severity: DiagnosticSeverity
  message: string
  node_id?: string | null
  edge_id?: string | null
  code_target?: CodeTarget | null
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
  code_target: CodeTarget
  diagnostics: ValidationDiagnostic[]
  code?: string | null
}

export interface MetadataResponse {
  name: string
  version: string
  schema_version: number
  code_targets: CodeTarget[]
  execution_backends: ExecutionBackendStatus[]
  configuration: {
    optimizers: string[]
    losses: string[]
    metrics: string[]
    initializers: string[]
    regularizers: string[]
  }
}

export interface ExecutionBackendStatus {
  id: ExecutionBackend
  available: boolean
  reason_code?: string | null
  accelerator?: string | null
  device_name?: string | null
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

export const codeTargetLabels: Record<CodeTarget, string> = {
  keras: 'Portable Keras',
  tensorflow: 'Native TensorFlow',
  pytorch: 'Native PyTorch',
  jax: 'Native JAX / Flax',
}

export const codeTargetShortLabels: Record<CodeTarget, string> = {
  keras: 'Keras',
  tensorflow: 'TensorFlow',
  pytorch: 'PyTorch',
  jax: 'JAX / Flax',
}

export interface EditorNodeData extends Record<string, unknown> {
  manifest: NodeManifest
  parameters: Record<string, unknown>
  label: string
  diagnostics: ValidationDiagnostic[]
  shapeSummary?: ShapeSummary
}

export type EditorNode = Node<EditorNodeData, 'modelNode'>
export type EditorEdge = Edge

export interface ProjectExportBundle {
  format: 'neuro-weave-project'
  format_version: 2
  exported_at: string
  project: {
    name: string
    description: string
    code_target: CodeTarget
    graph: ModelGraph
  }
}

export const emptyModelConfiguration = (): ModelConfiguration => ({
  optimizer: 'adam',
  optimizer_parameters: { learning_rate: 0.001 },
  losses: {},
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
    diagnostics: [],
  },
})
