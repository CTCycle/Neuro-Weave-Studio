import type { NodeManifest, PortDefinition } from './types'

/**
 * Expand a manifest's declarative parameter-cardinality ports for one node
 * instance. Graph-facing callers must use this instead of raw manifest port
 * arrays so dynamic ports have one identity everywhere in the editor.
 */
export function materializePorts(
  definitions: readonly PortDefinition[],
  parameters: Record<string, unknown> = {},
): PortDefinition[] {
  const materialized: PortDefinition[] = []
  for (const definition of definitions) {
    const cardinality = definition.cardinality
    if (!cardinality || cardinality.kind === 'single') {
      materialized.push(definition)
      continue
    }
    const parameterId = cardinality.parameter_id
    const value = parameterId ? parameters[parameterId] : undefined
    if (typeof value !== 'number' || !Number.isInteger(value) || value < cardinality.minimum || value > cardinality.maximum) {
      throw new Error(
        `Port cardinality parameter '${parameterId ?? 'unknown'}' must be an integer between ${cardinality.minimum} and ${cardinality.maximum}`,
      )
    }
    for (let index = 0; index < value; index += 1) {
      materialized.push({
        ...definition,
        id: formatPattern(cardinality.id_pattern, definition, index),
        display_name: formatPattern(cardinality.display_name_pattern, definition, index),
        cardinality: { ...cardinality, kind: 'single' },
      })
    }
  }
  return materialized
}

export function materializedInputs(
  manifest: NodeManifest,
  parameters: Record<string, unknown> = {},
): PortDefinition[] {
  return materializePorts(manifest.inputs, parameters)
}

export function materializedOutputs(
  manifest: NodeManifest,
  parameters: Record<string, unknown> = {},
): PortDefinition[] {
  return materializePorts(manifest.outputs, parameters)
}

function formatPattern(
  pattern: string,
  definition: PortDefinition,
  index: number,
): string {
  return pattern
    .replaceAll('{id}', definition.id)
    .replaceAll('{display_name}', definition.display_name)
    .replaceAll('{index}', String(index))
}
