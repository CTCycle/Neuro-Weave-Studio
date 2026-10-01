// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
    public readonly detail: unknown = null,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export function apiErrorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === 'string') return detail
  if (typeof detail !== 'object' || detail === null) return fallback

  const payload = detail as Record<string, unknown>
  const message = typeof payload['message'] === 'string' ? payload['message'] : null
  const diagnostics = diagnosticMessages(payload['diagnostics'])
  if (message && diagnostics.length) return `${message} (${diagnostics.join(' ')})`
  if (message) return message
  if (diagnostics.length) return diagnostics.join(' ')
  return fallback
}

function diagnosticMessages(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => typeof item === 'object' && item !== null && 'message' in item ? (item as { message?: unknown }).message : null)
    .filter((message): message is string => typeof message === 'string' && message.length > 0)
}
