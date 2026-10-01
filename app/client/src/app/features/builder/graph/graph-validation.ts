// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import type { EditorEdge, EditorNode } from '../../../domain/editor.types'
import { materializedInputs, materializedOutputs } from '../../../domain/ports'
import { conditionMatches, resolveImplementationCompatibility, type ExecutionBackend } from '../../../domain/types'

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
  | 'unsupported_profile'
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
  unsupported_profile: 'One of these nodes is unavailable for the current execution profile.',
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
  executionBackend?: ExecutionBackend,
  kerasEnabled?: boolean,
): GraphConnectionValidation {
  if (!connection.source || !connection.target || !connection.sourceHandle || !connection.targetHandle) return rejected('missing_endpoint')
  const sourceNode = nodes.find((node) => node.id === connection.source)
  const targetNode = nodes.find((node) => node.id === connection.target)
  if (!sourceNode || !targetNode) return rejected('missing_node')
  let sourcePort: ReturnType<typeof materializedOutputs>[number] | undefined
  let targetPort: ReturnType<typeof materializedInputs>[number] | undefined
  try {
    sourcePort = materializedOutputs(sourceNode.data.manifest, sourceNode.data.parameters)
      .find((port) => port.id === connection.sourceHandle)
    targetPort = materializedInputs(targetNode.data.manifest, targetNode.data.parameters)
      .find((port) => port.id === connection.targetHandle)
  } catch {
    return rejected('missing_port')
  }
  if (!sourcePort || !targetPort) return rejected('missing_port')
  if (!conditionMatches(sourcePort.enabled_when, sourceNode.data.parameters)
    || !conditionMatches(targetPort.enabled_when, targetNode.data.parameters)) return rejected('disabled_port')
  const expectedKind = targetPort.kind === 'tensor_list' ? targetPort.element_kind ?? 'tensor' : targetPort.kind
  const kindMatches = sourcePort.kind === expectedKind
    || (sourcePort.kind === 'tensor_list' && targetPort.kind === 'tensor_list')
  if (!kindMatches) return rejected('incompatible_kind')
  if (executionBackend !== undefined && kerasEnabled !== undefined
    && (!resolveImplementationCompatibility(sourceNode.data.manifest, executionBackend, kerasEnabled).available
      || !resolveImplementationCompatibility(targetNode.data.manifest, executionBackend, kerasEnabled).available)) return rejected('unsupported_profile')
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
  executionBackend?: ExecutionBackend,
  kerasEnabled?: boolean,
): boolean {
  return validateGraphConnection(nodes, edges, connection, executionBackend, kerasEnabled).valid
}

/**
 * Returns cheap semantic candidates for Foblex connection snapping. The server
 * remains authoritative for inferred shape and final connection validation.
 * Cycle-compatible targets remain visible so a drop can reach the authoritative
 * server check and return its cycle diagnostic to the user.
 */
export function connectionCandidates(
  nodes: readonly EditorNode[],
  edges: readonly EditorEdge[],
  source: string,
  sourceHandle: string,
  executionBackend?: ExecutionBackend,
  kerasEnabled?: boolean,
): GraphConnection[] {
  const candidates: GraphConnection[] = []
  for (const node of nodes) {
    let inputs: ReturnType<typeof materializedInputs>
    try {
      inputs = materializedInputs(node.data.manifest, node.data.parameters)
    } catch {
      inputs = []
    }
    for (const port of inputs) {
      const connection = { source, sourceHandle, target: node.id, targetHandle: port.id }
      const result = validateGraphConnection(
        nodes,
        edges,
        connection,
        executionBackend,
        kerasEnabled,
      )
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
