import { describe, expect, it } from 'vitest'
import type { ImplementationMapping, NodeManifest, ParameterDefinition } from './types'
import { conditionMatches, parameterValueMatchesDefinition, resolveImplementationCompatibility, resolveParameterCompatibility } from './types'

const parameter: ParameterDefinition = {
  id: 'mode',
  display_name: 'Mode',
  type: 'enum',
  default: 'auto',
  required: false,
  constraints: {},
  options: ['auto', 'fast'],
  description: '',
}

const manifest = (mapping: ImplementationMapping | undefined): NodeManifest => ({
  id: 'layer.test',
  version: 1,
  display_name: 'Test layer',
  category: 'test',
  description: '',
  semantic_role: 'operation',
  inputs: [],
  outputs: [],
  parameters: [parameter],
  shape_rule: { kind: 'operation', config: {} },
  implementations: mapping ? { pytorch: mapping } : {},
  ui: { icon: 'Box', accent: 'cyan', width: 250, height: 120 },
  deprecated: false,
  documentation_ref: '',
})

const verifiedMapping = (overrides: Partial<ImplementationMapping> = {}): ImplementationMapping => ({
  supported: true,
  status: 'verified',
  ...overrides,
})

describe('profile compatibility resolver', () => {
  it.each([
    ['missing_implementation', undefined, undefined],
    ['unsupported_implementation', false, 'verified'],
    ['unsupported_implementation', true, 'unsupported'],
    ['unverified_implementation', true, 'unverified'],
  ] as const)('treats %s mappings as unavailable', (reasonCode, supported, status) => {
    const mapping = supported === undefined
      ? undefined
      : verifiedMapping({ supported, status: status ?? 'verified' })
    const result = resolveImplementationCompatibility(manifest(mapping), 'pytorch', false)

    expect(result.available).toBe(false)
    expect(result.reasonCode).toBe(reasonCode)
    expect(result.reason).toContain('Test layer')
  })

  it('treats a Keras execution-backend exclusion as unavailable', () => {
    const result = resolveImplementationCompatibility(
      {
        ...manifest(undefined),
        implementations: { keras: verifiedMapping({ execution_backends: { pytorch: false } }) },
      },
      'pytorch',
      true,
    )

    expect(result).toMatchObject({
      available: false,
      reasonCode: 'backend_excluded',
    })
  })

  it('keeps an unsupported parameter default visible but blocks non-default values', () => {
    const node = manifest(verifiedMapping({
      parameters: {
        mode: {
          supported: false,
          status: 'verified',
          constraints: {},
        },
      },
    }))

    const defaultValue = resolveParameterCompatibility(node, parameter, 'auto', { mode: 'auto' }, 'pytorch', false)
    const nonDefaultValue = resolveParameterCompatibility(node, parameter, 'fast', { mode: 'fast' }, 'pytorch', false)

    expect(defaultValue).toMatchObject({ editable: false, valueCompatible: true, reasonCode: 'unsupported_parameter' })
    expect(nonDefaultValue).toMatchObject({ editable: false, valueCompatible: false, reasonCode: 'unsupported_parameter' })
  })

  it('keeps parameters editable when the verified node has no target-specific capability entry', () => {
    const result = resolveParameterCompatibility(manifest(verifiedMapping()), parameter, 'auto', { mode: 'auto' }, 'pytorch', false)

    expect(result).toMatchObject({ editable: true, valueCompatible: true, reasonCode: null })
  })

  it('treats an unverified parameter mapping as unavailable for every value', () => {
    const node = manifest(verifiedMapping({
      parameters: {
        mode: {
          supported: true,
          status: 'unverified',
          constraints: {},
        },
      },
    }))
    const result = resolveParameterCompatibility(node, parameter, 'auto', { mode: 'auto' }, 'pytorch', false)

    expect(result).toMatchObject({
      editable: false,
      valueCompatible: false,
      reasonCode: 'unverified_parameter',
    })
  })

  it('keeps a supported parameter editable while exposing a fixed-value restriction', () => {
    const node = manifest(verifiedMapping({
      parameters: {
        mode: {
          supported: true,
          status: 'verified',
          allowed_values: ['auto'],
          constraints: {},
        },
      },
    }))

    expect(resolveParameterCompatibility(node, parameter, 'auto', { mode: 'auto' }, 'pytorch', false)).toMatchObject({
      editable: false,
      valueCompatible: true,
      reasonCode: 'parameter_fixed',
    })
    expect(resolveParameterCompatibility(node, parameter, 'fast', { mode: 'fast' }, 'pytorch', false)).toMatchObject({
      editable: false,
      valueCompatible: false,
      reasonCode: 'unsupported_parameter_option',
    })
  })

  it('reports malformed stored values without making repair controls unavailable', () => {
    const node = manifest(verifiedMapping())
    const result = resolveParameterCompatibility(node, parameter, 'unknown', { mode: 'unknown' }, 'pytorch', false)

    expect(parameterValueMatchesDefinition(parameter, 'unknown')).toBe(false)
    expect(result).toMatchObject({
      editable: true,
      valueCompatible: false,
      reasonCode: 'invalid_parameter_value',
    })
  })

  it('evaluates all supported conditional rule forms consistently', () => {
    expect(conditionMatches({ mode: { equals: 'fast' } }, { mode: 'fast' })).toBe(true)
    expect(conditionMatches({ mode: { not_equals: 'auto' } }, { mode: 'fast' })).toBe(true)
    expect(conditionMatches({ mode: { in: ['fast', 'safe'] } }, { mode: 'safe' })).toBe(true)
    expect(conditionMatches({ mode: { not_in: ['auto', 'safe'] } }, { mode: 'fast' })).toBe(true)
    expect(conditionMatches({ mode: ['fast', 'safe'] }, { mode: 'auto' })).toBe(false)
  })
})
