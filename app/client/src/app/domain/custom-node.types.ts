export type CustomNodeKind = 'mathematical' | 'composite' | 'repository'

export interface MathematicalNodeSpec {
  type: 'mathematical'
  formula: string
  expression?: string | null
  input_symbols: ['x']
}

export interface CompositeNodeComponent {
  definition_id: string
  definition_version: number
  label?: string | null
  parameters: Record<string, unknown>
}

export interface CompositeNodeSpec {
  type: 'composite'
  connection_mode: 'sequential'
  components: CompositeNodeComponent[]
}

export interface RepositoryNodeSpec {
  type: 'repository'
  provider: 'huggingface'
  repository_id: string
  revision: string
  artifact_type: 'model' | 'layer'
  entrypoint?: string | null
  resolved_commit?: string | null
  library_name?: string | null
  pipeline_tag?: string | null
}

export type CustomNodeSpec = MathematicalNodeSpec | CompositeNodeSpec | RepositoryNodeSpec

export interface CustomNodeDefinition {
  id: string
  version: number
  display_name: string
  category: string
  description: string
  kind: CustomNodeKind
  spec: CustomNodeSpec
  origin: 'custom'
  locked: false
  can_edit: true
  can_delete: true
  executable_template: boolean
  created_at: string
  updated_at: string
}

export interface CustomNodeCatalogResponse {
  definitions: CustomNodeDefinition[]
}

export interface CustomNodeDefinitionMutation {
  display_name: string
  category: string
  description: string
  spec: CustomNodeSpec
}
