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
  if (typeof detail === 'object' && detail !== null && 'message' in detail) {
    const message = (detail as { message?: unknown }).message
    if (typeof message === 'string') return message
  }
  return typeof detail === 'string' ? detail : fallback
}
