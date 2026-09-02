export function isEffectivelyEmptyObject(value: unknown): boolean {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const entries = Object.entries(value as Record<string, unknown>)
  if (!entries.length) return true
  return entries.every(([, item]) => {
    if (item === null || item === undefined || item === '') return true
    if (Array.isArray(item)) return item.length === 0
    if (typeof item === 'object') return isEffectivelyEmptyObject(item)
    return false
  })
}
