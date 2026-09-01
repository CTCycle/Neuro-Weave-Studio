import type { ParameterCapability, ParameterDefinition } from '../../../domain/types'

export type ParameterDraftValue = string | boolean | number | null

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
  if (definition.type === 'axis' && (text.startsWith('[') || text.startsWith('('))) {
    const value = parseSequence(text, false)
    return { valid: value !== undefined && sequenceItemsValid(value, definition, capability) && capabilityAllowsValue(capability, value), value }
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
    const parsed = parseSequence(text, definition.type === 'list')
    return { valid: parsed !== undefined && sequenceItemsValid(parsed, definition, capability) && capabilityAllowsValue(capability, parsed), value: parsed }
  }
  if (definition.type === 'object' || definition.type === 'initializer' || definition.type === 'regularizer' || definition.type === 'constraint') {
    const value = parseJsonObject(text)
    if (value === undefined) {
      if (definition.type !== 'object' && typeof raw === 'string') return { valid: capabilityAllowsValue(capability, raw), value: raw }
      return { valid: false }
    }
    return { valid: capabilityAllowsValue(capability, value), value }
  }
  if (definition.type === 'enum' || definition.type === 'activation' || definition.type === 'dtype') {
    const option = definition.options.find((item) => String(item) === text)
    if (definition.options.length > 0 && option === undefined) return { valid: false }
    const value = option === undefined ? raw : option
    return { valid: capabilityAllowsValue(capability, value), value }
  }
  return { valid: capabilityAllowsValue(capability, raw), value: raw }
}

export function displayParameterValue(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map((item) => item === null ? '?' : displayParameterValue(item)).join(', ')}]`
  if (typeof value === 'object' && value !== null) return JSON.stringify(value)
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

function parseSequence(text: string, allowStrings: boolean): unknown[] | undefined {
  const trimmed = text.trim()
  if (!trimmed) return []
  const unwrapped = (trimmed.startsWith('[') && trimmed.endsWith(']')) || (trimmed.startsWith('(') && trimmed.endsWith(')'))
    ? trimmed.slice(1, -1).trim()
    : trimmed
  const values = splitSequence(unwrapped)
  const parsed: unknown[] = []
  for (const value of values) {
    if ((value.startsWith('[') && value.endsWith(']')) || (value.startsWith('(') && value.endsWith(')'))) {
      const nested = parseSequence(value, allowStrings)
      if (!nested) return undefined
      parsed.push(nested)
    } else if (value === '?' || value.toLowerCase() === 'none' || value.toLowerCase() === 'null') {
      parsed.push(null)
    } else if (/^-?\d+(?:\.\d+)?$/.test(value)) {
      parsed.push(Number(value))
    } else if (allowStrings && ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'")))) {
      parsed.push(value.slice(1, -1))
    } else if (allowStrings && (value.toLowerCase() === 'true' || value.toLowerCase() === 'false')) {
      parsed.push(value.toLowerCase() === 'true')
    } else {
      return undefined
    }
  }
  return parsed
}

function parseJsonObject(text: string): unknown | undefined {
  try {
    const parsed: unknown = JSON.parse(text)
    return parsed !== null && typeof parsed === 'object' ? parsed : undefined
  } catch {
    return undefined
  }
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
    if (definition.type === 'list' && (typeof item === 'string' || typeof item === 'boolean')) return true
    return typeof item === 'number' && Number.isFinite(item)
      && (definition.type === 'list' || Number.isSafeInteger(item))
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
