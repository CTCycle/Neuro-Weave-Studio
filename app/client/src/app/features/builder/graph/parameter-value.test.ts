import { describe, expect, it } from 'vitest'
import type { ParameterDefinition } from '../../../domain/types'
import { displayParameterValue, isFiniteParameter, parameterOptions, parseParameterValue } from './parameter-value'

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

  it('treats every finite parameter type as a typed choice', () => {
    for (const type of ['enum', 'activation', 'dtype', 'initializer', 'regularizer', 'constraint'] as const) {
      const choice = definition(type, { default: 'relu', options: ['relu', 'tanh'] })
      expect(isFiniteParameter(choice)).toBe(true)
      expect(parseParameterValue(choice, 'tanh')).toEqual({ valid: true, value: 'tanh' })
      expect(parseParameterValue(choice, 'not-supported').valid).toBe(false)
    }
    expect(parameterOptions(definition('boolean'))).toEqual([true, false])
    expect(parseParameterValue(definition('boolean', { default: true }), 'false')).toEqual({ valid: true, value: false })
    expect(parseParameterValue(definition('boolean', { default: false }), 'true')).toEqual({ valid: true, value: true })
  })

  it('does not turn a finite choice into arbitrary JSON or text', () => {
    const initializer = definition('initializer', { default: 'zeros', options: ['zeros', 'ones'] })
    expect(parseParameterValue(initializer, '{"class_name":"Custom"}').valid).toBe(false)
    expect(isFiniteParameter(definition('object'))).toBe(false)
    expect(isFiniteParameter(definition('string'))).toBe(false)
    expect(isFiniteParameter(definition('string', { options: ['a', 'b'] }))).toBe(true)
  })
})
