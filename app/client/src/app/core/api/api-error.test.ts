import { describe, expect, it } from 'vitest'
import { apiErrorMessage } from './api-error'

describe('apiErrorMessage', () => {
  it('preserves nested server diagnostics alongside the top-level message', () => {
    expect(apiErrorMessage({ message: 'Graph cannot be compiled', diagnostics: [{ code: 'invalid_filename', message: 'Filename stems cannot contain path separators' }] }, 'fallback'))
      .toBe('Graph cannot be compiled (Filename stems cannot contain path separators)')
  })

  it('falls back when a server payload has no usable message', () => {
    expect(apiErrorMessage({ diagnostics: [{ code: 'unknown' }, null] }, 'fallback')).toBe('fallback')
  })
})
