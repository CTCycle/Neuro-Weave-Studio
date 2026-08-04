import type { ParameterDefinition } from '../domain/types'
import { useGraphStore } from '../state/graphStore'

function coerce(definition: ParameterDefinition, raw: string | boolean): unknown {
  if (definition.type === 'boolean') return Boolean(raw)
  if (definition.type === 'integer') return Number.parseInt(String(raw), 10)
  if (definition.type === 'number' || definition.type === 'axis') return Number(String(raw))
  if (definition.type === 'shape') {
    return String(raw).split(',').map((item) => {
      const value = item.trim()
      if (value === '?' || value.toLowerCase() === 'none' || value === '') return null
      return Number.parseInt(value, 10)
    })
  }
  return raw
}

function displayValue(value: unknown): string {
  if (Array.isArray(value)) return value.map((item) => item === null ? '?' : Array.isArray(item) ? item.join(':') : String(item)).join(', ')
  return value === null || value === undefined ? '' : String(value)
}

export function PropertiesInspector() {
  const selectedNodeId = useGraphStore((state) => state.selectedNodeId)
  const node = useGraphStore((state) => state.nodes.find((item) => item.id === selectedNodeId))
  const updateNode = useGraphStore((state) => state.updateNode)
  if (!node) {
    return <aside className="inspector-panel panel empty-state"><div><span className="eyebrow">INSPECTOR</span><h2>No node selected</h2><p>Select a node to edit its label and parameters.</p></div></aside>
  }
  const updateParameter = (definition: ParameterDefinition, value: string | boolean) => {
    updateNode(node.id, { parameters: { ...node.data.parameters, [definition.id]: coerce(definition, value) } })
  }
  return (
    <aside className="inspector-panel panel">
      <div className="panel-heading"><div><span className="eyebrow">INSPECTOR</span><h2>{node.data.manifest.display_name}</h2></div></div>
      <label className="field"><span>Node label</span><input value={node.data.label} onChange={(event) => updateNode(node.id, { label: event.target.value })} /></label>
      <div className="inspector-description">{node.data.manifest.description}</div>
      <div className="field-stack">
        {node.data.manifest.parameters.map((definition) => {
          const value = node.data.parameters[definition.id] ?? definition.default
          if (definition.type === 'boolean') {
            return <label key={definition.id} className="toggle-field"><span>{definition.display_name}</span><input type="checkbox" checked={Boolean(value)} onChange={(event) => updateParameter(definition, event.target.checked)} /></label>
          }
          if (definition.type === 'enum') {
            return <label key={definition.id} className="field"><span>{definition.display_name}</span><select value={displayValue(value)} onChange={(event) => updateParameter(definition, event.target.value)}>{definition.options.map((option) => <option key={String(option)} value={String(option)}>{String(option)}</option>)}</select></label>
          }
          return <label key={definition.id} className="field"><span>{definition.display_name}</span><input type={definition.type === 'number' || definition.type === 'integer' || definition.type === 'axis' ? 'number' : 'text'} value={displayValue(value)} onChange={(event) => updateParameter(definition, event.target.value)} /></label>
        })}
      </div>
      {node.data.shapeSummary && <div className="summary-card"><strong>Inferred output</strong>{Object.entries(node.data.shapeSummary.outputs).map(([port, spec]) => <div key={port}><span>{port}</span><code>{spec.shape.map((value) => value === null ? '?' : typeof value === 'object' ? value.name : value).join(' × ')}</code></div>)}<div><span>Parameters</span><code>{node.data.shapeSummary.parameter_count?.toLocaleString() ?? 'unknown'}</code></div></div>}
      {node.data.diagnostics.length > 0 && <div className="node-diagnostics">{node.data.diagnostics.map((item, index) => <div key={`${item.code}-${index}`} className={`diagnostic ${item.severity}`}><strong>{item.severity}</strong><span>{item.message}</span></div>)}</div>}
    </aside>
  )
}
