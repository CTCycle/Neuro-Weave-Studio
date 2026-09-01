import type { Position } from '../../../domain/types'
import type { GraphConnection } from './graph-validation'

export interface GraphNodeSize {
  width: number
  height: number
}

// ReactFlow rendered the shared model-node surface at 230px wide by roughly
// 103px high. Keeping this geometry in the adapter makes Foblex's measured
// boxes and fit calculations use the same visual contract.
export const GRAPH_NODE_SIZE: GraphNodeSize = { width: 230, height: 103 }

export interface ParsedConnector {
  nodeId: string
  direction: 'input' | 'output'
  portId: string
}

export function connectorId(nodeId: string, direction: ParsedConnector['direction'], portId: string): string {
  return `${nodeId}|${direction}|${portId}`
}

export function parseConnector(value: string): ParsedConnector | null {
  const [nodeId, direction, ...port] = value.split('|')
  if ((direction !== 'input' && direction !== 'output') || !nodeId || !port.length) return null
  return { nodeId, direction, portId: port.join('|') }
}

export function normalizeConnection(sourceId: string, targetId: string): GraphConnection | null {
  const source = parseConnector(sourceId)
  const target = parseConnector(targetId)
  if (!source || !target || source.direction !== 'output' || target.direction !== 'input') return null
  return { source: source.nodeId, sourceHandle: source.portId, target: target.nodeId, targetHandle: target.portId }
}

export function portOffset(index: number, count: number): number {
  return ((index + 1) / (count + 1)) * 100
}

export function graphFitPadding(nodes: readonly { position: Position }[], ratio: number, size: GraphNodeSize = GRAPH_NODE_SIZE): Position {
  if (!nodes.length) return { x: 0, y: 0 }
  const minX = Math.min(...nodes.map((node) => node.position.x))
  const maxX = Math.max(...nodes.map((node) => node.position.x))
  const minY = Math.min(...nodes.map((node) => node.position.y))
  const maxY = Math.max(...nodes.map((node) => node.position.y))
  return { x: (maxX - minX + size.width) * ratio, y: (maxY - minY + size.height) * ratio }
}

export function flowPositionFromDrop(event: DragEvent, canvas: HTMLElement): Position {
  const bounds = canvas.getBoundingClientRect()
  const flow = canvas.querySelector('f-flow')
  const viewport = canvas.querySelector('f-canvas')
  const flowBounds = flow instanceof HTMLElement ? flow.getBoundingClientRect() : bounds
  const transform = viewport instanceof HTMLElement && typeof window !== 'undefined' ? window.getComputedStyle(viewport).transform : ''
  const matrix = transform.match(/^matrix\(([^)]+)\)$/)?.[1].split(',').map(Number)
  const scale = matrix?.[0] && Number.isFinite(matrix[0]) ? matrix[0] : 1
  const translateX = matrix?.[4] && Number.isFinite(matrix[4]) ? matrix[4] : 0
  const translateY = matrix?.[5] && Number.isFinite(matrix[5]) ? matrix[5] : 0
  return { x: (event.clientX - flowBounds.left - translateX) / scale, y: (event.clientY - flowBounds.top - translateY) / scale }
}
