// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import type {
  ArrayBinding,
  ColumnBinding,
  DatasetBinding,
  DatasetDefinitionMutation,
  DatasetDocument,
  DatasetInspection,
  DatasetRoleDefinition,
  DatasetSource,
  DatasetType,
  FolderLabelBinding,
  ImageBinding,
  TextBinding,
} from '../../domain/dataset.types'

export type FieldRole = 'input' | 'output' | 'metadata' | 'unused'
export type DatasetFieldKind = 'column' | 'array' | 'image' | 'text' | 'folder_label'

export interface DatasetField {
  key: string
  sourceId: string
  value: string
  label: string
  kind: DatasetFieldKind
  dtype?: string
  structural: boolean
}

export interface DatasetFieldOptions {
  structuralKeys?: ReadonlySet<string>
}

export function buildDatasetFields(result: DatasetInspection | null, options: DatasetFieldOptions = {}): DatasetField[] {
  if (!result) return []
  const structuralKeys = options.structuralKeys ?? new Set<string>()
  const fields: DatasetField[] = []
  for (const source of result.sources) {
    if (source.type === 'table') {
      for (const column of source.columns) {
        const key = `${source.source_id}::${column.name}`
        fields.push({ key, sourceId: source.source_id, value: column.name, label: column.name, kind: 'column', dtype: column.dtype, structural: structuralKeys.has(key) })
      }
    }
    if (source.type === 'array') {
      for (const array of source.arrays) {
        const key = `${source.source_id}::${array.key}`
        fields.push({ key, sourceId: source.source_id, value: array.key, label: `${array.key} · [${array.shape.join(' × ')}]`, kind: 'array', dtype: array.dtype, structural: structuralKeys.has(key) })
      }
    }
    if (source.type === 'image_folder') {
      fields.push({ key: `${source.source_id}::__image`, sourceId: source.source_id, value: '__image', label: 'Image samples', kind: 'image', structural: false })
      if (source.class_names.length) fields.push({ key: `${source.source_id}::__label`, sourceId: source.source_id, value: '__label', label: 'Folder labels', kind: 'folder_label', structural: false })
    }
    if (source.type === 'text') fields.push({ key: `${source.source_id}::__text`, sourceId: source.source_id, value: '__text', label: 'Text samples', kind: 'text', structural: false })
  }
  return fields
}

export function bindingKeys(binding: DatasetBinding): string[] {
  if (binding.type === 'columns') return binding.columns.map((column) => `${binding.source_id}::${column}`)
  if (binding.type === 'array') return binding.keys.map((key) => `${binding.source_id}::${key}`)
  if (binding.type === 'image') return [`${binding.source_id}::__image`]
  if (binding.type === 'text') return [`${binding.source_id}::__text`]
  return [`${binding.source_id}::__label`]
}

export function defaultFieldRole(field: DatasetField, datasetType: DatasetType): FieldRole {
  if (field.structural) return 'unused'
  if (field.kind === 'folder_label') return 'output'
  if (field.kind === 'image' || field.kind === 'text') return 'input'
  if (datasetType === 'tabular' || datasetType === 'time_series' || datasetType === 'array') return 'input'
  return 'unused'
}

export function initialRoleAssignments(fields: DatasetField[], datasetType: DatasetType): Record<string, FieldRole> {
  return Object.fromEntries(fields.map((field) => [field.key, defaultFieldRole(field, datasetType)]))
}

export function roleAssignmentsFromDocument(fields: DatasetField[], datasetType: DatasetType, document: DatasetDocument): Record<string, FieldRole> {
  const assignments = initialRoleAssignments(fields, datasetType)
  const structuralKeys = new Set(fields.filter((field) => field.structural).map((field) => field.key))
  for (const [role, definition] of [['input', document.input_definition], ['output', document.output_definition], ['metadata', document.metadata_definition]] as const) {
    for (const binding of definition.bindings) {
      for (const key of bindingKeys(binding)) {
        if (assignments[key] !== undefined && !structuralKeys.has(key)) assignments[key] = role
      }
    }
  }
  return assignments
}

export function bindingsFor(fields: DatasetField[], sources: DatasetSource[], assignments: Record<string, FieldRole>, role: FieldRole): DatasetBinding[] {
  const selected = fields.filter((field) => assignments[field.key] === role && !field.structural)
  const bindings: DatasetBinding[] = []
  for (const source of sources) {
    const sourceFields = selected.filter((field) => field.sourceId === source.id)
    if (!sourceFields.length) continue
    if (source.type === 'table') {
      const columns = sourceFields.filter((field) => field.kind === 'column').map((field) => field.value)
      if (columns.length) bindings.push({ type: 'columns', source_id: source.id, columns } satisfies ColumnBinding)
    } else if (source.type === 'array') {
      const keys = sourceFields.filter((field) => field.kind === 'array').map((field) => field.value)
      if (keys.length) bindings.push({ type: 'array', source_id: source.id, keys } satisfies ArrayBinding)
    } else if (source.type === 'image_folder') {
      for (const field of sourceFields) {
        bindings.push(field.kind === 'folder_label' ? { type: 'folder_label', source_id: source.id } satisfies FolderLabelBinding : { type: 'image', source_id: source.id } satisfies ImageBinding)
      }
    } else if (sourceFields.some((field) => field.kind === 'text')) {
      bindings.push({ type: 'text', source_id: source.id } satisfies TextBinding)
    }
  }
  return bindings
}

export function roleDefinition(fields: DatasetField[], sources: DatasetSource[], assignments: Record<string, FieldRole>, role: FieldRole): DatasetRoleDefinition {
  return { bindings: bindingsFor(fields, sources, assignments, role) }
}

export function cloneDatasetSources(sources: DatasetSource[]): DatasetSource[] {
  return sources.map((source) => ({ ...source }))
}

export function definitionWithRoles(
  document: DatasetDocument,
  sources: DatasetSource[],
  fields: DatasetField[],
  assignments: Record<string, FieldRole>,
): DatasetDefinitionMutation {
  return {
    name: document.name,
    dataset_type: document.dataset_type,
    sources,
    input_definition: roleDefinition(fields, sources, assignments, 'input'),
    output_definition: roleDefinition(fields, sources, assignments, 'output'),
    metadata_definition: roleDefinition(fields, sources, assignments, 'metadata'),
    relationships: document.relationships.map((relationship) => ({ ...relationship })),
    sequence_definition: document.sequence_definition ? { ...document.sequence_definition } : null,
    subset_definition: { ...document.subset_definition },
    split_definition: document.split_definition ? { ...document.split_definition } : null,
  }
}

export function bindingSummary(bindings: DatasetBinding[]): string {
  return bindings.map((binding) => {
    if (binding.type === 'columns') return `${binding.columns.length} column${binding.columns.length === 1 ? '' : 's'}`
    if (binding.type === 'array') return `${binding.keys.length} array${binding.keys.length === 1 ? '' : 's'}`
    if (binding.type === 'image') return 'image'
    if (binding.type === 'text') return 'text'
    return 'folder label'
  }).join(' + ') || 'None'
}
