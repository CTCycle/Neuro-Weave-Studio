import { describe, expect, it } from 'vitest'
import { materializePorts, materializedOutputs } from './ports'
import type { NodeManifest, PortDefinition } from './types'

const dynamicOutput: PortDefinition = {
  id: 'output',
  display_name: 'Output',
  kind: 'tensor',
  required: true,
  accepts_multiple: false,
  description: 'Split output',
  cardinality: {
    kind: 'parameter',
    parameter_id: 'sections',
    minimum: 1,
    maximum: 4,
    id_pattern: 'output_{index}',
    display_name_pattern: 'Output {index}',
  },
}

const splitManifest = {
  inputs: [],
  outputs: [dynamicOutput],
} as NodeManifest

describe('declarative port materialization', () => {
  it('expands parameter-driven outputs into stable canonical ports', () => {
    expect(materializedOutputs(splitManifest, { sections: 3 }).map((port) => ({ id: port.id, label: port.display_name }))).toEqual([
      { id: 'output_0', label: 'Output 0' },
      { id: 'output_1', label: 'Output 1' },
      { id: 'output_2', label: 'Output 2' },
    ])
  })

  it('rejects values outside the declared cardinality contract', () => {
    expect(() => materializePorts([dynamicOutput], { sections: 0 })).toThrow(/between 1 and 4/)
    expect(() => materializePorts([dynamicOutput], { sections: 1.5 })).toThrow(/integer/)
    expect(() => materializePorts([dynamicOutput], { sections: 5 })).toThrow(/between 1 and 4/)
  })
})
