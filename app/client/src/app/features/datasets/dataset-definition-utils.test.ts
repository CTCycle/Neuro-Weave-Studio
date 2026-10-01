// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { describe, expect, it } from 'vitest'
import type { DatasetDocument, DatasetInspection } from '../../domain/dataset.types'
import { bindingsFor, buildDatasetFields, definitionWithRoles, initialRoleAssignments, roleAssignmentsFromDocument } from './dataset-definition-utils'

const inspection: DatasetInspection = {
  dataset_type: 'tabular',
  sample_count: 2,
  sources: [{
    type: 'table',
    source_id: 'table',
    path: 'patients.csv',
    format: 'csv',
    sample_count: 2,
    columns: [
      { name: 'age', dtype: 'int64', nullable: false },
      { name: 'weight', dtype: 'int64', nullable: false },
    ],
    inference_sample_size: 2,
  }],
  warnings: [],
}

const document = {
  id: 'dataset-1',
  name: 'Patients',
  dataset_type: 'tabular',
  sources: [{ type: 'table', id: 'table', path: 'patients.csv', format: 'csv', has_header: true, delimiter: ',' }],
  input_definition: { bindings: [{ type: 'columns', source_id: 'table', columns: ['weight'] }] },
  output_definition: { bindings: [{ type: 'columns', source_id: 'table', columns: ['age'] }] },
  metadata_definition: { bindings: [] },
  relationships: [],
  sequence_definition: null,
  subset_definition: { type: 'all' },
  split_definition: null,
  format: 'CSV',
  sample_count: 2,
  selected_sample_count: 2,
  inspection,
  validation: { status: 'valid', checked_at: '2026-09-17T00:00:00Z', errors: [], warnings: [] },
  availability: { status: 'available', unavailable_source_ids: [] },
  created_at: '2026-09-17T00:00:00Z',
  updated_at: '2026-09-17T00:00:00Z',
} satisfies DatasetDocument

describe('dataset definition utilities', () => {
  it('builds structural-aware fields and conservative tabular defaults', () => {
    const fields = buildDatasetFields(inspection, { structuralKeys: new Set(['table::age']) })
    const assignments = initialRoleAssignments(fields, 'tabular')

    expect(fields.map((field) => [field.key, field.structural])).toEqual([
      ['table::age', true],
      ['table::weight', false],
    ])
    expect(assignments).toEqual({ 'table::age': 'unused', 'table::weight': 'input' })
  })

  it('hydrates persisted roles and builds complete typed bindings', () => {
    const fields = buildDatasetFields(inspection)
    const assignments = roleAssignmentsFromDocument(fields, 'tabular', document)

    expect(bindingsFor(fields, document.sources, assignments, 'input')).toEqual([{ type: 'columns', source_id: 'table', columns: ['weight'] }])
    expect(bindingsFor(fields, document.sources, assignments, 'output')).toEqual([{ type: 'columns', source_id: 'table', columns: ['age'] }])
    expect(definitionWithRoles(document, document.sources, fields, assignments)).toMatchObject({
      input_definition: { bindings: [{ type: 'columns', source_id: 'table', columns: ['weight'] }] },
      output_definition: { bindings: [{ type: 'columns', source_id: 'table', columns: ['age'] }] },
      split_definition: null,
    })
  })
})
