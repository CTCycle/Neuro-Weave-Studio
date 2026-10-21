// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { afterEach, describe, expect, it, vi } from 'vitest'
import { RuntimeApiConfigService } from './runtime-api-config.service'

describe('RuntimeApiConfigService', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('loads the API base URL from the uncached runtime document', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ api_base_url: 'https://api.example.test/v2/' }),
    }))
    const service = new RuntimeApiConfigService()

    await service.load()

    expect(fetch).toHaveBeenCalledWith('/runtime-config.json', { cache: 'no-store' })
    expect(service.apiBaseUrl).toBe('https://api.example.test/v2')
  })

  it('keeps same-origin /api when the runtime document is unavailable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('runtime config unavailable')))
    const service = new RuntimeApiConfigService()

    await service.load()

    expect(service.apiBaseUrl).toBe('/api')
  })
})
