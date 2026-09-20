import { describe, expect, it } from 'vitest'
import { draftSnapshot } from './builder-draft-snapshot'

describe('builder draft snapshots', () => {
  it('does not mark a semantically identical graph dirty because object keys were reordered', () => {
    const left = { schema_version: 6, nodes: [{ id: 'node-1', parameters: { units: 16, activation: 'relu' } }], edges: [], output_bindings: [] }
    const right = { output_bindings: [], edges: [], nodes: [{ parameters: { activation: 'relu', units: 16 }, id: 'node-1' }], schema_version: 6 }

    expect(draftSnapshot('Model', '', 'pytorch', true, left)).toBe(draftSnapshot('Model', '', 'pytorch', true, right))
  })
})
