import type { ParameterCapability, ParameterDefinition } from '../../../domain/types'

export type ParameterDraftValue = string | boolean

export interface ParsedParameterValue {
  valid: boolean
  value?: unknown
}

export function parseParameterValue(
  definition: ParameterDefinition,
  raw: ParameterDraftValue,
  capability?: ParameterCapability,
): ParsedParameterValue {
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
  return { valid: capabilityAllowsValue(capability, raw), value: raw }
}

export function displayParameterValue(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map((item) => item === null ? '?' : displayParameterValue(item)).join(', ')}]`
  return value === null || value === undefined ? '' : String(value)
}

export function helpfulParameterDescription(definition: ParameterDefinition): string | null {
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

function sequenceItemsValid(value: unknown, definition: ParameterDefinition, capability?: ParameterCapability): boolean {
  if (!Array.isArray(value)) return false
  const constraints = { ...definition.constraints, ...(capability?.constraints ?? {}) }
  const length = constraints['length']
  const minLength = constraints['min_length']
  const maxLength = constraints['max_length']
  if ((typeof length === 'number' && value.length !== length)
    || (typeof minLength === 'number' && value.length < minLength)
    || (typeof maxLength === 'number' && value.length > maxLength)) return false
  const itemMin = constraints['item_min']
  const itemMax = constraints['item_max']
  const valid = (item: unknown): boolean => {
    if (item === null) return true
    if (Array.isArray(item)) return item.every(valid)
    return typeof item === 'number' && Number.isSafeInteger(item)
      && (typeof itemMin !== 'number' || item >= itemMin)
      && (typeof itemMax !== 'number' || item <= itemMax)
  }
  return value.every(valid)
}

function capabilityAllowsValue(capability: ParameterCapability | undefined, value: unknown): boolean {
  return !capability?.allowed_values || capability.allowed_values.some((item) => JSON.stringify(item) === JSON.stringify(value))
}

function withinConstraints(definition: ParameterDefinition, value: number, capability?: ParameterCapability): boolean {
  const constraints = { ...definition.constraints, ...(capability?.constraints ?? {}) }
  const minimum = constraints['min']
  const maximum = constraints['max']
  return (typeof minimum !== 'number' || value >= minimum) && (typeof maximum !== 'number' || value <= maximum)
}
