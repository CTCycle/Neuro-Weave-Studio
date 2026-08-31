import type { EditorEdge, EditorNode } from '../../../domain/editor.types'

export interface GraphConnection {
  source: string
  sourceHandle: string
  target: string
  targetHandle: string
}

export function isGraphConnectionValid(
  nodes: readonly EditorNode[],
  edges: readonly EditorEdge[],
  connection: GraphConnection,
): boolean {
  if (!connection.source || !connection.target || !connection.sourceHandle || !connection.targetHandle) return false
  if (wouldCreateCycle(nodes, edges, connection.source, connection.target)) return false
  const sourceNode = nodes.find((node) => node.id === connection.source)
  const targetNode = nodes.find((node) => node.id === connection.target)
  const sourcePort = sourceNode?.data.manifest.outputs.find((port) => port.id === connection.sourceHandle)
  const targetPort = targetNode?.data.manifest.inputs.find((port) => port.id === connection.targetHandle)
  if (!sourcePort || !targetPort) return false
  const expectedKind = targetPort.kind === 'tensor_list' ? targetPort.element_kind ?? 'tensor' : targetPort.kind
  if (sourcePort.kind !== expectedKind) return false
  return targetPort.accepts_multiple || !edges.some((edge) => edge.target === connection.target && edge.targetHandle === connection.targetHandle)
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
