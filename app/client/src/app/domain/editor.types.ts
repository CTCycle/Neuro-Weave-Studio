import type { NodeManifest, Position } from './types'

export interface EditorNodeData extends Record<string, unknown> {
  manifest: NodeManifest
  parameters: Record<string, unknown>
  label: string
}

export interface EditorNode {
  id: string
  type: 'modelNode'
  position: Position
  data: EditorNodeData
  locked: boolean
  selected?: boolean
}

export interface EditorEdge {
  id: string
  source: string
  target: string
  sourceHandle?: string
  targetHandle?: string
  type?: 'bezier'
  selected?: boolean
}
