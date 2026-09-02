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
  const outputs = uniqueOutputNames(bindings
    .filter((binding) => isEffectiveOutput(binding, graph, manifestFor))
    .map((binding) => resolveOutput(binding, graph, manifestFor, graph.output_bindings.length > 0)))
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
    task_hint: taskHintForOutput(graph, node, manifestFor),
    explicit,
  }
}

function taskHintForOutput(
  graph: ModelGraph,
  outputNode: GraphNodeInstance | undefined,
  manifestFor: (node: GraphNodeInstance) => NodeManifest | undefined,
): string | null {
  const manifest = outputNode ? manifestFor(outputNode) : undefined
  const taskHint = manifest?.task_hint ?? null
  if (taskHint !== 'classification' || !outputNode) return taskHint

  const head = terminalHead(graph, outputNode, manifestFor)
  const width = outputWidth(head)
  if (width === 1) return 'binary_classification'
  if (width !== null && width >= 2) return 'multiclass_classification'
  return null
}

function terminalHead(
  graph: ModelGraph,
  outputNode: GraphNodeInstance,
  manifestFor: (node: GraphNodeInstance) => NodeManifest | undefined,
): GraphNodeInstance | undefined {
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]))
  const incoming = new Map(
    graph.edges
      .filter((edge) => edge.target_port_id === 'input')
      .map((edge) => [edge.target_node_id, edge.source_node_id]),
  )
  let current: GraphNodeInstance | undefined = outputNode
  const visited = new Set<string>()
  while (current && !visited.has(current.id)) {
    visited.add(current.id)
    const manifest = manifestFor(current)
    if (current.id !== outputNode.id || manifest?.semantic_role !== 'output_adapter') return current
    const sourceId = incoming.get(current.id)
    current = sourceId ? nodes.get(sourceId) : undefined
  }
  return current
}

function outputWidth(node: GraphNodeInstance | undefined): number | null {
  if (!node) return null
  if (node.definition_id === 'layer.activation') return null
  if (!['layer.dense', 'conv.conv1d', 'conv.conv2d', 'layer.embedding'].includes(node.definition_id)) return null
  const value = node.parameters['units'] ?? node.parameters['filters']
  return typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : null
}

function uniqueOutputNames(outputs: GraphOutputSummary[]): GraphOutputSummary[] {
  const counts = new Map<string, number>()
  for (const output of outputs) counts.set(output.name, (counts.get(output.name) ?? 0) + 1)
  const used = new Set<string>()
  return outputs.map((output) => {
    const base = (counts.get(output.name) ?? 0) > 1
      ? `${output.name}_${output.node_id}`
      : output.name
    let name = base
    let suffix = 2
    while (used.has(name)) name = `${base}_${suffix++}`
    used.add(name)
    return { ...output, name }
  })
}

function isEffectiveOutput(
  binding: OutputBinding,
  graph: ModelGraph,
  manifestFor: (node: GraphNodeInstance) => NodeManifest | undefined,
): boolean {
  const node = graph.nodes.find((item) => item.id === binding.node_id)
  const manifest = node ? manifestFor(node) : undefined
  const port = manifest?.outputs.find((item) => item.id === binding.port_id)
  return Boolean(
    node
    && manifest
    && manifest.semantic_role !== 'input'
    && port
    && ['tensor', 'tensor_list', 'mask', 'scalar'].includes(port.kind)
    && conditionMatches(port.enabled_when, node.parameters),
  )
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
