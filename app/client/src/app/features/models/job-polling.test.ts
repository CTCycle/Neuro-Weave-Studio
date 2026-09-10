import { describe, expect, it } from 'vitest'
import type { JobResponse } from '../../domain/types'
import { pollJobUntilTerminal } from './job-polling'

const job = (state: JobResponse['state']): JobResponse => ({ id: 'job-1', state } as JobResponse)

describe('job polling', () => {
  it('returns a terminal submission without an extra request', async () => {
    let requests = 0
    const result = await pollJobUntilTerminal(job('succeeded'), async () => { requests += 1; return job('failed') }, () => undefined, { intervalMs: 0, timeoutMs: 10 })

    expect(result.state).toBe('succeeded')
    expect(requests).toBe(0)
  })

  it('returns the terminal response and publishes each polled state', async () => {
    const states: JobResponse['state'][] = []
    const responses = [job('running'), job('succeeded')]
    const result = await pollJobUntilTerminal(job('starting'), async () => responses.shift() as JobResponse, (current) => states.push(current.state), { intervalMs: 0, timeoutMs: 100 })

    expect(result.state).toBe('succeeded')
    expect(states).toEqual(['running', 'succeeded'])
  })

  it('leaves an in-progress job non-terminal when the bounded wait expires', async () => {
    const result = await pollJobUntilTerminal(job('starting'), async () => job('running'), () => undefined, { intervalMs: 1, timeoutMs: 2 })

    expect(result.state).toBe('running')
  })
})
