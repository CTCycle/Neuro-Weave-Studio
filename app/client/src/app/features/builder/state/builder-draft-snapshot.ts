// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

export function draftSnapshot(
  name: string,
  description: string,
  executionBackend: string,
  kerasEnabled: boolean,
  graph: unknown,
): string {
  return JSON.stringify(canonicalize({ name, description, executionBackend, kerasEnabled, graph }))
}

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map((item) => canonicalize(item))
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, canonicalize(item)]),
    )
  }
  return value
}
