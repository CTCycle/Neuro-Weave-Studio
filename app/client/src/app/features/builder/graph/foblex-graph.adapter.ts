import type { Position } from '../../../domain/types'
import type { GraphConnection } from './graph-validation'

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

export function flowPositionFromDrop(event: DragEvent, canvas: HTMLElement): Position {
  const bounds = canvas.getBoundingClientRect()
  return { x: event.clientX - bounds.left, y: event.clientY - bounds.top }
}
