import type { EditorEdge, EditorNode } from '../../../domain/editor.types'

export interface LayoutPosition {
  x: number
  y: number
}

export function layeredLayout(nodes: readonly EditorNode[], edges: readonly EditorEdge[]): Map<string, LayoutPosition> {
  const incoming = new Map(nodes.map((node) => [node.id, 0]))
  const outgoing = new Map(nodes.map((node) => [node.id, [] as string[]]))
  for (const edge of edges) {
    incoming.set(edge.target, (incoming.get(edge.target) ?? 0) + 1)
    outgoing.get(edge.source)?.push(edge.target)
  }
  const queue = nodes.filter((node) => incoming.get(node.id) === 0).map((node) => node.id)
  const layer = new Map(queue.map((id) => [id, 0]))
  while (queue.length) {
    const source = queue.shift() as string
    for (const target of outgoing.get(source) ?? []) {
      layer.set(target, Math.max(layer.get(target) ?? 0, (layer.get(source) ?? 0) + 1))
      incoming.set(target, (incoming.get(target) ?? 1) - 1)
      if (incoming.get(target) === 0) queue.push(target)
    }
  }
  const rows = new Map<number, number>()
  return new Map(nodes.map((node) => {
    const column = layer.get(node.id) ?? 0
    const row = rows.get(column) ?? 0
    rows.set(column, row + 1)
    return [node.id, { x: column * 360, y: row * 180 }]
  }))
}
