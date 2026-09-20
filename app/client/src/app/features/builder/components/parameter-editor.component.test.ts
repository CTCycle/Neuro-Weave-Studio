import { describe, expect, it } from 'vitest'
import { parameterEditorReadOnly } from './parameter-editor.component'

describe('parameter editor state', () => {
  it('keeps invalid values editable so the user can repair them', () => {
    expect(parameterEditorReadOnly(false, 'invalid')).toBe(false)
    expect(parameterEditorReadOnly(false, 'fixed')).toBe(true)
    expect(parameterEditorReadOnly(false, 'backend-inapplicable')).toBe(true)
    expect(parameterEditorReadOnly(true, 'invalid')).toBe(true)
  })
})
