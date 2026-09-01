import type { EditorEdge, EditorNode } from '../../../domain/editor.types'

export interface GraphConnection {
  source: string
  sourceHandle: string
  target: string
  targetHandle: string
}

export type GraphConnectionRejection =
  | 'missing_endpoint'
  | 'missing_node'
  | 'missing_port'
  | 'disabled_port'
  | 'incompatible_kind'
  | 'duplicate'
  | 'input_at_capacity'
  | 'cycle'

export type GraphConnectionValidation =
  | { valid: true }
  | { valid: false; reason: GraphConnectionRejection; message: string }

const rejectionMessages: Record<GraphConnectionRejection, string> = {
  missing_endpoint: 'Choose an output and an input port to create a connection.',
  missing_node: 'The selected node is no longer available.',
  missing_port: 'The selected port is no longer available.',
  disabled_port: 'This port is disabled by the current node configuration.',
  incompatible_kind: 'These ports carry incompatible tensor or value types.',
  duplicate: 'This connection already exists.',
  input_at_capacity: 'This input already has its maximum number of connections.',
  cycle: 'Connection blocked: this edge would create a cycle.',
}

function rejected(reason: GraphConnectionRejection): GraphConnectionValidation {
  return { valid: false, reason, message: rejectionMessages[reason] }
}

export function validateGraphConnection(
  nodes: readonly EditorNode[],
  edges: readonly EditorEdge[],
  connection: GraphConnection,
): GraphConnectionValidation {
  if (!connection.source || !connection.target || !connection.sourceHandle || !connection.targetHandle) return rejected('missing_endpoint')
  const sourceNode = nodes.find((node) => node.id === connection.source)
  const targetNode = nodes.find((node) => node.id === connection.target)
  if (!sourceNode || !targetNode) return rejected('missing_node')
  const sourcePort = sourceNode?.data.manifest.outputs.find((port) => port.id === connection.sourceHandle)
  const targetPort = targetNode?.data.manifest.inputs.find((port) => port.id === connection.targetHandle)
  if (!sourcePort || !targetPort) return rejected('missing_port')
  if (!conditionMatches(sourcePort.enabled_when, sourceNode.data.parameters)
    || !conditionMatches(targetPort.enabled_when, targetNode.data.parameters)) return rejected('disabled_port')
  const expectedKind = targetPort.kind === 'tensor_list' ? targetPort.element_kind ?? 'tensor' : targetPort.kind
  if (sourcePort.kind !== expectedKind) return rejected('incompatible_kind')
  if (edges.some((edge) => edge.source === connection.source
    && (edge.sourceHandle ?? 'output') === (connection.sourceHandle ?? 'output')
    && edge.target === connection.target
    && (edge.targetHandle ?? 'input') === (connection.targetHandle ?? 'input'))) return rejected('duplicate')
  if (!targetPort.accepts_multiple && edges.some((edge) => edge.target === connection.target
    && (edge.targetHandle ?? 'input') === connection.targetHandle)) return rejected('input_at_capacity')
  if (wouldCreateCycle(nodes, edges, connection.source, connection.target)) return rejected('cycle')
  return { valid: true }
}

export function isGraphConnectionValid(
  nodes: readonly EditorNode[],
  edges: readonly EditorEdge[],
  connection: GraphConnection,
): boolean {
  return validateGraphConnection(nodes, edges, connection).valid
}

/**
 * Returns semantic target candidates for Foblex connection snapping.
 * Cycle candidates remain available so the completed gesture reaches the
 * store and can report the specific cycle rejection to the user.
 */
export function connectionCandidates(
  nodes: readonly EditorNode[],
  edges: readonly EditorEdge[],
  source: string,
  sourceHandle: string,
): GraphConnection[] {
  const candidates: GraphConnection[] = []
  for (const node of nodes) {
    for (const port of node.data.manifest.inputs) {
      const connection = { source, sourceHandle, target: node.id, targetHandle: port.id }
      const result = validateGraphConnection(nodes, edges, connection)
      if (result.valid || result.reason === 'cycle') candidates.push(connection)
    }
  }
  return candidates
}

export function wouldCreateCycle(
  nodes: readonly EditorNode[],
  edges: readonly EditorEdge[],
  source: string,
  target: string,
): boolean {
  if (source === target) return true
  const adjacency = new Map<string, string[]>()
  for (const node of nodes) adjacency.set(node.id, [])
  for (const edge of edges) adjacency.get(edge.source)?.push(edge.target)
  const stack = [target]
  const visited = new Set<string>()
  while (stack.length) {
    const current = stack.pop() as string
    if (current === source) return true
    if (visited.has(current)) continue
    visited.add(current)
    stack.push(...(adjacency.get(current) ?? []))
  }
  return false
}

function conditionMatches(
  condition: Record<string, unknown> | undefined,
  parameters: Record<string, unknown>,
): boolean {
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
