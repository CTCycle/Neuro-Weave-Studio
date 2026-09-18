export type DatasetType = 'tabular' | 'images' | 'text' | 'time_series' | 'array' | 'composed'
export type DatasetValidationStatus = 'valid' | 'invalid'
export type DatasetAvailabilityStatus = 'available' | 'unavailable'

export interface TableSource {
  type: 'table'
  id: string
  path: string
  format: 'csv' | 'tsv' | 'parquet'
  has_header: boolean
  delimiter: ',' | '\t' | ';' | '|' | null
}

export interface ImageFolderSource {
  type: 'image_folder'
  id: string
  path: string
  recursive: boolean
  label_mode: 'none' | 'parent_folder'
}

export interface TextSource {
  type: 'text'
  id: string
  path: string
  path_kind: 'file' | 'folder'
  recursive: boolean
}

export interface ArraySource {
  type: 'array'
  id: string
  path: string
  format: 'npy' | 'npz'
}

export type DatasetSource = TableSource | ImageFolderSource | TextSource | ArraySource

export interface ColumnBinding {
  type: 'columns'
  source_id: string
  columns: string[]
}

export interface ImageBinding {
  type: 'image'
  source_id: string
}

export interface TextBinding {
  type: 'text'
  source_id: string
}

export interface ArrayBinding {
  type: 'array'
  source_id: string
  keys: string[]
}

export interface FolderLabelBinding {
  type: 'folder_label'
  source_id: string
}

export type DatasetBinding = ColumnBinding | ImageBinding | TextBinding | ArrayBinding | FolderLabelBinding

export interface DatasetRoleDefinition {
  bindings: DatasetBinding[]
}

export interface FileTableRelationship {
  type: 'file_table'
  file_source_id: string
  file_key: 'filename' | 'stem' | 'relative_path'
  table_source_id: string
  table_key_column: string
}

export interface TimeSeriesStructure {
  timestamp_column: string | null
  group_column: string | null
}

export type DatasetSubsetDefinition =
  | { type: 'all' }
  | { type: 'first_n'; count: number }
  | { type: 'range'; start: number; end: number }

export type DatasetSplitStrategy = 'random' | 'sequential'

export interface DatasetSplitDefinition {
  strategy: DatasetSplitStrategy
  train_fraction: number
  validation_fraction: number
  test_fraction: number
  seed: number | null
}

export interface DatasetDefinitionMutation {
  name: string
  dataset_type: DatasetType
  sources: DatasetSource[]
  input_definition: DatasetRoleDefinition
  output_definition: DatasetRoleDefinition
  metadata_definition: DatasetRoleDefinition
  relationships: FileTableRelationship[]
  sequence_definition: TimeSeriesStructure | null
  subset_definition: DatasetSubsetDefinition
  split_definition: DatasetSplitDefinition | null
}

export interface DatasetInspectionRequest {
  dataset_type: DatasetType
  sources: DatasetSource[]
}

export interface ColumnMetadata {
  name: string
  dtype: string
  nullable: boolean
}

export interface ArrayMetadata {
  key: string
  shape: number[]
  dtype: string
}

export interface TableInspection {
  type: 'table'
  source_id: string
  path: string
  format: 'csv' | 'tsv' | 'parquet'
  sample_count: number
  columns: ColumnMetadata[]
  inference_sample_size: number
}

export interface ImageFolderInspection {
  type: 'image_folder'
  source_id: string
  path: string
  sample_count: number
  extensions: string[]
  class_names: string[]
  class_counts: Record<string, number>
}

export interface TextInspection {
  type: 'text'
  source_id: string
  path: string
  sample_count: number
  extensions: string[]
}

export interface ArrayInspection {
  type: 'array'
  source_id: string
  path: string
  format: 'npy' | 'npz'
  sample_count: number
  arrays: ArrayMetadata[]
}

export type DatasetSourceInspection = TableInspection | ImageFolderInspection | TextInspection | ArrayInspection

export interface DatasetInspection {
  dataset_type: DatasetType
  sample_count: number
  sources: DatasetSourceInspection[]
  warnings: string[]
}

export interface DatasetValidationFinding {
  code: string
  message: string
  source_id: string | null
  details: Record<string, unknown>
}

export interface DatasetValidationMetadata {
  status: DatasetValidationStatus
  checked_at: string
  errors: DatasetValidationFinding[]
  warnings: DatasetValidationFinding[]
}

export interface DatasetValidationResult {
  valid: boolean
  sample_count: number
  selected_sample_count: number
  inspection: DatasetInspection | null
  validation: DatasetValidationMetadata
}

export interface DatasetAvailabilityState {
  status: DatasetAvailabilityStatus
  unavailable_source_ids: string[]
}

export interface DatasetSummary {
  id: string
  name: string
  dataset_type: DatasetType
  format: string
  primary_path: string
  sample_count: number
  selected_sample_count: number
  input_summary: string
  output_summary: string
  validation_status: DatasetValidationStatus
  availability: DatasetAvailabilityState
  created_at: string
  updated_at: string
}

export interface DatasetDocument extends DatasetDefinitionMutation {
  id: string
  format: string
  sample_count: number
  selected_sample_count: number
  inspection: DatasetInspection
  validation: DatasetValidationMetadata
  availability: DatasetAvailabilityState
  created_at: string
  updated_at: string
}
