import type { NodeManifest, Position, ShapeSummary } from './types'

export interface EditorNodeData extends Record<string, unknown> {
  manifest: NodeManifest
  parameters: Record<string, unknown>
  label: string
  shapeSummary?: ShapeSummary
}

export interface EditorNode {
  id: string
  type: 'modelNode'
  position: Position
  data: EditorNodeData
  selected?: boolean
}

export interface EditorEdge {
  id: string
  source: string
  target: string
  sourceHandle?: string
  targetHandle?: string
  type?: 'smoothstep'
  selected?: boolean
}
