import { describe, expect, it } from 'vitest'
import type { ParameterDefinition } from '../../../domain/types'
import { displayParameterValue, parseParameterValue } from './parameter-value'

const definition = (type: ParameterDefinition['type'], overrides: Partial<ParameterDefinition> = {}): ParameterDefinition => ({
  id: 'value', display_name: 'Value', type, default: 0, required: true, constraints: {}, options: [], description: '', ...overrides,
})

describe('parameter value parser', () => {
  it('parses nested shape values and nullable markers', () => {
    expect(parseParameterValue(definition('shape'), '[1, ?, [3, 4]]')).toEqual({ valid: true, value: [1, null, [3, 4]] })
    expect(displayParameterValue([1, null, [3, 4]])).toBe('[1, ?, [3, 4]]')
  })

  it('enforces integer bounds and backend capability restrictions', () => {
    const integer = definition('integer', { constraints: { min: 1, max: 8 } })
    expect(parseParameterValue(integer, '0').valid).toBe(false)
    expect(parseParameterValue(integer, '4', { supported: true, constraints: { max: 3 } }).valid).toBe(false)
    expect(parseParameterValue(integer, '4', { supported: true, constraints: {} }).value).toBe(4)
  })

  it('allows optional empty values and rejects malformed sequences', () => {
    expect(parseParameterValue(definition('string', { required: false }), '').valid).toBe(true)
    expect(parseParameterValue(definition('tuple'), '[1, nope]').valid).toBe(false)
  })

  it('round-trips list-valued axes and typed enum choices', () => {
    expect(parseParameterValue(definition('axis', { default: [-1] }), '[-2, -1]')).toEqual({ valid: true, value: [-2, -1] })
    expect(parseParameterValue(definition('enum', { options: [1, 2], default: 1 }), '2')).toEqual({ valid: true, value: 2 })
  })
})
