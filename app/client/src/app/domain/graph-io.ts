import type { GraphNodeInstance, ModelGraph, NodeManifest, OutputBinding, PortDefinition } from './types'

export interface GraphOutputSummary {
  node_id: string
  port_id: string
  name: string
  task_hint?: string | null
  explicit: boolean
}

export interface GraphIoSummary {
  inputCount: number
  outputCount: number
  inputs: GraphNodeInstance[]
  outputs: GraphOutputSummary[]
}

export function summarizeGraphIo(graph: ModelGraph, manifests: NodeManifest[]): GraphIoSummary {
  const catalog = new Map(manifests.map((manifest) => [`${manifest.id}@${manifest.version}`, manifest]))
  const manifestFor = (node: GraphNodeInstance): NodeManifest | undefined => catalog.get(`${node.definition_id}@${node.definition_version}`)
  const inputs = graph.nodes.filter((node) => manifestFor(node)?.semantic_role === 'input')
  const bindings = graph.output_bindings.length ? graph.output_bindings : inferredBindings(graph, manifestFor)
  const outputs = bindings.map((binding) => resolveOutput(binding, graph, manifestFor, graph.output_bindings.length > 0))
  return { inputCount: inputs.length, outputCount: outputs.length, inputs, outputs }
}

export function outputSelectors(output: GraphOutputSummary, graph: ModelGraph): string[] {
  const node = graph.nodes.find((item) => item.id === output.node_id)
  return [...new Set([
    output.name,
    output.node_id,
    `${output.node_id}:${output.port_id}`,
    node?.label ?? '',
  ].filter(Boolean))]
}

function inferredBindings(
  graph: ModelGraph,
  manifestFor: (node: GraphNodeInstance) => NodeManifest | undefined,
): OutputBinding[] {
  const consumed = new Set(graph.edges.map((edge) => `${edge.source_node_id}:${edge.source_port_id}`))
  const bindings: OutputBinding[] = []
  for (const node of graph.nodes) {
    const manifest = manifestFor(node)
    if (!manifest || manifest.semantic_role === 'input') continue
    for (const port of tensorOutputPorts(manifest, node.parameters)) {
      if (!consumed.has(`${node.id}:${port.id}`)) bindings.push({ node_id: node.id, port_id: port.id })
    }
  }
  return bindings
}

function resolveOutput(
  binding: OutputBinding,
  graph: ModelGraph,
  manifestFor: (node: GraphNodeInstance) => NodeManifest | undefined,
  explicit: boolean,
): GraphOutputSummary {
  const node = graph.nodes.find((item) => item.id === binding.node_id)
  const manifest = node ? manifestFor(node) : undefined
  const port = manifest?.outputs.find((item) => item.id === binding.port_id)
  const enabledPorts = manifest && node ? tensorOutputPorts(manifest, node.parameters) : []
  const base = node?.label || node?.id || binding.node_id
  return {
    node_id: binding.node_id,
    port_id: binding.port_id,
    name: enabledPorts.length > 1 ? `${base}_${port?.id ?? binding.port_id}` : base,
    task_hint: manifest?.task_hint ?? null,
    explicit,
  }
}

function tensorOutputPorts(manifest: NodeManifest, parameters: Record<string, unknown>): PortDefinition[] {
  return manifest.outputs.filter((port) =>
    ['tensor', 'tensor_list', 'mask', 'scalar'].includes(port.kind)
    && conditionMatches(port.enabled_when, parameters),
  )
}

function conditionMatches(condition: Record<string, unknown> | undefined, parameters: Record<string, unknown>): boolean {
  if (!condition) return true
  return Object.entries(condition).every(([parameterId, expected]) => {
    const actual = parameters[parameterId]
    if (typeof expected === 'object' && expected !== null && !Array.isArray(expected)) {
      const rule = expected as Record<string, unknown>
      if ('equals' in rule && !sameValue(actual, rule['equals'])) return false
      if ('not_equals' in rule && sameValue(actual, rule['not_equals'])) return false
      if ('in' in rule && Array.isArray(rule['in']) && !rule['in'].some((item) => sameValue(actual, item))) return false
      if ('not_in' in rule && Array.isArray(rule['not_in']) && rule['not_in'].some((item) => sameValue(actual, item))) return false
      return true
    }
    if (Array.isArray(expected)) return expected.some((item) => sameValue(actual, item))
    return sameValue(actual, expected)
  })
}

function sameValue(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}
