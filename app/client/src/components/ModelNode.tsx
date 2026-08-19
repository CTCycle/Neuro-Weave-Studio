import { memo } from 'react'
import { Handle, Position } from '@xyflow/react'
import type { EditorNodeData } from '../domain/types'

const portTop = (index: number, total: number) => `${((index + 1) / (total + 1)) * 100}%`

export const ModelNode = memo(({ data, selected }: { data: EditorNodeData; selected?: boolean }) => {
  const firstOutput = data.shapeSummary ? Object.values(data.shapeSummary.outputs)[0] : undefined
  const shape = firstOutput?.shape.map((value) => value === null ? '?' : typeof value === 'object' ? value.name : value).join(' × ')
  return (
    <div className={`model-node ${selected ? 'is-selected' : ''} accent-${data.manifest.ui.accent}`}>
      {data.manifest.inputs.map((port, index) => (
        <Handle
          key={port.id}
          id={port.id}
          type="target"
          position={Position.Left}
          style={{ top: portTop(index, data.manifest.inputs.length) }}
          title={`${port.display_name}: ${port.kind}`}
          className={`port port-${port.kind}`}
        />
      ))}
      <div className="model-node__eyebrow">{data.manifest.category}</div>
      <div className="model-node__title">{data.label}</div>
      <div className="model-node__id">{data.manifest.id}</div>
      <div className="model-node__footer">
        <span>{shape || 'Shape pending'}</span>
      </div>
      {data.manifest.outputs.map((port, index) => (
        <Handle
          key={port.id}
          id={port.id}
          type="source"
          position={Position.Right}
          style={{ top: portTop(index, data.manifest.outputs.length) }}
          title={`${port.display_name}: ${port.kind}`}
          className={`port port-${port.kind}`}
        />
      ))}
    </div>
  )
})
ModelNode.displayName = 'ModelNode'
