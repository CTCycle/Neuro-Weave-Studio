import type { JobResponse } from '../../domain/types'

export const terminalJobStates = new Set<JobResponse['state']>(['succeeded', 'terminated', 'failed'])

export interface JobPollingOptions {
  intervalMs: number
  timeoutMs: number
  isDestroyed?: () => boolean
}

/**
 * Poll a submitted job until the server reports a terminal state or the
 * bounded wait expires. The caller remains responsible for presenting a
 * timeout as an in-progress/unknown result rather than as a worker failure.
 */
export async function pollJobUntilTerminal(
  initial: JobResponse,
  getJob: (jobId: string) => Promise<JobResponse>,
  onUpdate: (job: JobResponse) => void,
  options: JobPollingOptions,
): Promise<JobResponse> {
  let current = initial
  const deadline = Date.now() + options.timeoutMs
  while (!terminalJobStates.has(current.state)) {
    if (options.isDestroyed?.()) return current
    const remaining = deadline - Date.now()
    if (remaining <= 0) return current
    await delay(Math.min(options.intervalMs, remaining))
    if (options.isDestroyed?.()) return current
    current = await getJob(current.id)
    onUpdate(current)
  }
  return current
}

function delay(durationMs: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, durationMs))
}
