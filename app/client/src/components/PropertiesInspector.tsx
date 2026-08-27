import { useEffect, useMemo, useRef, useState } from 'react'
import type { ExecutionBackend, ParameterCapability, ParameterDefinition } from '../domain/types'
import { implementationKey } from '../domain/types'
import { useGraphStore } from '../state/graphStore'

type DraftValue = string | boolean

function splitSequence(text: string): string[] {
  const values: string[] = []
  let depth = 0
  let start = 0
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index]
    if (character === '[' || character === '(') depth += 1
    if (character === ']' || character === ')') depth -= 1
    if (character === ',' && depth === 0) {
      values.push(text.slice(start, index).trim())
      start = index + 1
    }
  }
  values.push(text.slice(start).trim())
  return values.filter(Boolean)
}

function parseSequence(text: string): unknown[] | undefined {
  const trimmed = text.trim()
  if (!trimmed) return []
  const unwrapped = (trimmed.startsWith('[') && trimmed.endsWith(']')) || (trimmed.startsWith('(') && trimmed.endsWith(')'))
    ? trimmed.slice(1, -1).trim()
    : trimmed
  const values = splitSequence(unwrapped)
  const parsed: unknown[] = []
  for (const value of values) {
    if ((value.startsWith('[') && value.endsWith(']')) || (value.startsWith('(') && value.endsWith(')'))) {
      const nested = parseSequence(value)
      if (!nested) return undefined
      parsed.push(nested)
    } else if (value === '?' || value.toLowerCase() === 'none' || value.toLowerCase() === 'null') {
      parsed.push(null)
    } else if (/^-?\d+$/.test(value)) {
      parsed.push(Number(value))
    } else {
      return undefined
    }
  }
  return parsed
}

function sequenceItemsValid(
  value: unknown,
  definition: ParameterDefinition,
  capability?: ParameterCapability,
): boolean {
  if (!Array.isArray(value)) return false
  const constraints = { ...definition.constraints, ...(capability?.constraints ?? {}) }
  const length = constraints.length
  const minLength = constraints.min_length
  const maxLength = constraints.max_length
  if ((typeof length === 'number' && value.length !== length)
    || (typeof minLength === 'number' && value.length < minLength)
    || (typeof maxLength === 'number' && value.length > maxLength)) return false
  const itemMin = constraints.item_min
  const itemMax = constraints.item_max
  const valid = (item: unknown): boolean => {
    if (item === null) return true
    if (Array.isArray(item)) return item.every(valid)
    return typeof item === 'number' && Number.isSafeInteger(item)
      && (typeof itemMin !== 'number' || item >= itemMin)
      && (typeof itemMax !== 'number' || item <= itemMax)
  }
  return value.every(valid)
}

function parseValue(
  definition: ParameterDefinition,
  raw: DraftValue,
  capability?: ParameterCapability,
): { valid: boolean; value?: unknown } {
  if (definition.type === 'boolean') {
    const value = Boolean(raw)
    return { valid: capabilityAllowsValue(capability, value), value }
  }
  const text = String(raw).trim()
  if (!text) {
    if (definition.nullable) return { valid: capabilityAllowsValue(capability, null), value: null }
    if (!definition.required) return { valid: true }
    return { valid: false }
  }
  if (definition.type === 'integer' || definition.type === 'axis') {
    if (!/^-?\d+$/.test(text)) return { valid: false }
    const value = Number(text)
    return { valid: Number.isSafeInteger(value) && withinConstraints(definition, value, capability) && capabilityAllowsValue(capability, value), value }
  }
  if (definition.type === 'number') {
    const value = Number(text)
    return { valid: Number.isFinite(value) && withinConstraints(definition, value, capability) && capabilityAllowsValue(capability, value), value }
  }
  if (definition.type === 'shape' || definition.type === 'list' || definition.type === 'tuple') {
    const parsed = parseSequence(text)
    return { valid: parsed !== undefined && sequenceItemsValid(parsed, definition, capability) && capabilityAllowsValue(capability, parsed), value: parsed }
  }
  if (definition.type === 'enum' && definition.options.length > 0 && !definition.options.some((item) => String(item) === text)) return { valid: false }
  if (!capabilityAllowsValue(capability, raw)) return { valid: false }
  return { valid: true, value: raw }
}

function capabilityAllowsValue(capability: ParameterCapability | undefined, value: unknown): boolean {
  return !capability?.allowed_values || capability.allowed_values.some((item) => JSON.stringify(item) === JSON.stringify(value))
}

function withinConstraints(
  definition: ParameterDefinition,
  value: number,
  capability?: ParameterCapability,
): boolean {
  const constraints = { ...definition.constraints, ...(capability?.constraints ?? {}) }
  const minimum = constraints.min
  const maximum = constraints.max
  return (typeof minimum !== 'number' || value >= minimum) && (typeof maximum !== 'number' || value <= maximum)
}

function displayValue(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map((item) => item === null ? '?' : displayValue(item)).join(', ')}]`
  return value === null || value === undefined ? '' : String(value)
}

function helpfulDescription(definition: ParameterDefinition): string | null {
  const description = definition.description?.trim()
  if (!description) return null
  const normalized = description.toLowerCase().replace(/[.!?]+$/, '')
  const label = definition.display_name.trim().toLowerCase()
  const genericDescriptions = new Set([
    `configure ${label}`,
    `configure the ${label} used by this layer`,
  ])
  return genericDescriptions.has(normalized) ? null : description
}

export function PropertiesInspector({ onClose, isOpen = true, executionBackend = 'pytorch', kerasEnabled = true }: { onClose?: () => void; isOpen?: boolean; executionBackend?: ExecutionBackend; kerasEnabled?: boolean }) {
  const selectedNodeId = useGraphStore((state) => state.selectedNodeId)
  const node = useGraphStore((state) => state.nodes.find((item) => item.id === selectedNodeId))
  const updateNode = useGraphStore((state) => state.updateNode)
  const inspectorRef = useRef<HTMLElement>(null)
  const [drafts, setDrafts] = useState<Record<string, DraftValue>>({})
  const [invalid, setInvalid] = useState<Record<string, boolean>>({})
  useEffect(() => {
    const inspector = inspectorRef.current
    if (inspector && typeof inspector.scrollTo === 'function') inspector.scrollTo({ top: 0 })
    setDrafts({})
    setInvalid({})
  }, [selectedNodeId, isOpen])
  const grouped = useMemo(() => {
    const groups = new Map<string, ParameterDefinition[]>()
    for (const definition of node?.data.manifest.parameters ?? []) {
      const group = definition.group || 'General'
      groups.set(group, [...(groups.get(group) ?? []), definition])
    }
    return [...groups.entries()]
  }, [node])
  if (!node) {
    return <aside ref={inspectorRef} className="inspector-panel panel empty-state"><div><span className="eyebrow">INSPECTOR</span><h2>No node selected</h2><p>Select a node to edit its label and parameters.</p></div></aside>
  }
  const targetMapping = node.data.manifest.implementations[implementationKey(executionBackend, kerasEnabled)]
  const updateParameter = (definition: ParameterDefinition, raw: DraftValue) => {
    setDrafts((state) => ({ ...state, [definition.id]: raw }))
    const capability = targetMapping?.parameters?.[definition.id]
    const parsed = parseValue(definition, raw, capability)
    setInvalid((state) => ({ ...state, [definition.id]: !parsed.valid }))
    if (!parsed.valid) return
    const parameters = { ...node.data.parameters }
    if (parsed.value === undefined) delete parameters[definition.id]
    else parameters[definition.id] = parsed.value
    updateNode(node.id, { parameters })
  }
  return (
    <aside ref={inspectorRef} className="inspector-panel panel">
      <div className="panel-heading"><div><span className="eyebrow">INSPECTOR</span><h2>{node.data.manifest.display_name}</h2></div>{onClose && <button className="panel-close" onClick={onClose} aria-label="Close inspector">×</button>}</div>
      <label className="field"><span>Node label <small>(layer name)</small></span><input value={node.data.label} onChange={(event) => updateNode(node.id, { label: event.target.value })} /></label>
      <div className="inspector-description">{node.data.manifest.description}</div>
      <div className="field-stack">
        {grouped.map(([group, definitions]) => <details className="parameter-group" key={group} open><summary>{group}<span>{definitions.length}</span></summary>{definitions.map((definition) => {
          const value = drafts[definition.id] ?? displayValue(node.data.parameters[definition.id] ?? definition.default)
          const help = helpfulDescription(definition)
          const capability = targetMapping?.parameters?.[definition.id]
          const restrictedToSingleValue = capability?.allowed_values?.length === 1
          const disabled = targetMapping?.supported === false || capability?.supported === false || restrictedToSingleValue
          const options = definition.options.filter((option) => !capability?.allowed_values || capability.allowed_values.some((allowed) => String(allowed) === String(option)))
          const bounds = { min: typeof definition.constraints.min === 'number' ? definition.constraints.min : undefined, max: typeof definition.constraints.max === 'number' ? definition.constraints.max : undefined, step: typeof definition.constraints.step === 'number' ? definition.constraints.step : definition.type === 'integer' || definition.type === 'axis' ? 1 : undefined }
          if (definition.type === 'boolean') return <label key={definition.id} className="toggle-field"><span>{definition.display_name}{help && <small>{help}</small>}</span><input type="checkbox" checked={Boolean(node.data.parameters[definition.id] ?? definition.default)} disabled={disabled} onChange={(event) => updateParameter(definition, event.target.checked)} /></label>
          if (definition.type === 'enum') return <label key={definition.id} className="field"><span>{definition.display_name}{help && <small>{help}</small>}</span><select value={String(value)} disabled={disabled} onChange={(event) => updateParameter(definition, event.target.value)} aria-invalid={invalid[definition.id]}>{definition.nullable && <option value="">None</option>}{options.map((option) => <option key={String(option)} value={String(option)}>{String(option)}</option>)}</select>{invalid[definition.id] && <em className="field-error">Choose a supported option.</em>}</label>
          return <label key={definition.id} className="field"><span>{definition.display_name}{help && <small>{help}</small>}</span><input type={definition.type === 'number' || definition.type === 'integer' || definition.type === 'axis' ? 'number' : 'text'} value={String(value)} disabled={disabled} min={bounds.min} max={bounds.max} step={bounds.step} onChange={(event) => updateParameter(definition, event.target.value)} aria-invalid={invalid[definition.id]} />{invalid[definition.id] && <em className="field-error">Enter a complete value within the supported range.</em>}</label>
        })}</details>)}
      </div>
    </aside>
  )
}
