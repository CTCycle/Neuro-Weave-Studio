// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { describe, expect, it } from 'vitest'
import '@angular/compiler'
import { ApiError } from '../../core/api/api-error'
import type { DatasetValidationResult } from '../../domain/dataset.types'
import { extractDatasetValidation } from './dataset-wizard.component'

const validation: DatasetValidationResult = {
  valid: false,
  sample_count: 1,
  selected_sample_count: 1,
  inspection: null,
  validation: {
    status: 'invalid',
    checked_at: '2026-09-17T09:00:00Z',
    errors: [{
      code: 'missing_columns',
      message: 'Columns do not exist: missing',
      source_id: 'table',
      details: { examples: ['missing'] },
    }],
    warnings: [],
  },
}

describe('dataset registration validation errors', () => {
  it('retains the complete validation result from a registration 422', () => {
    const error = new ApiError(422, 'Dataset definition is invalid', {
      code: 'dataset_definition_invalid',
      validation,
    })

    expect(extractDatasetValidation(error)).toEqual(validation)
    expect(extractDatasetValidation(error)?.validation.errors[0].details).toEqual({ examples: ['missing'] })
  })

  it('does not treat unrelated or malformed errors as validation state', () => {
    expect(extractDatasetValidation(new ApiError(422, 'Other failure', { code: 'inspection_failed', validation }))).toBeNull()
    expect(extractDatasetValidation(new ApiError(422, 'Invalid definition', { code: 'dataset_definition_invalid' }))).toBeNull()
    expect(extractDatasetValidation(new ApiError(500, 'Invalid definition', { code: 'dataset_definition_invalid', validation }))).toBeNull()
  })
})
