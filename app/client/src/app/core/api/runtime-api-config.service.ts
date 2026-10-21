// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { Injectable } from '@angular/core'

const DEFAULT_API_BASE_URL = '/api'

interface RuntimeConfigurationDocument {
  api_base_url?: unknown
}

@Injectable({ providedIn: 'root' })
export class RuntimeApiConfigService {
  private configuredApiBaseUrl = DEFAULT_API_BASE_URL

  get apiBaseUrl(): string {
    return this.configuredApiBaseUrl
  }

  async load(): Promise<void> {
    try {
      const response = await fetch('/runtime-config.json', { cache: 'no-store' })
      if (!response.ok) return

      const configuration = await response.json() as RuntimeConfigurationDocument
      if (typeof configuration.api_base_url === 'string') {
        this.configuredApiBaseUrl = this.normalizeApiBaseUrl(configuration.api_base_url)
      }
    } catch {
      // Keep same-origin /api for static hosts that do not expose runtime config.
    }
  }

  private normalizeApiBaseUrl(value: string): string {
    const candidate = value.trim()
    if (!candidate || /\s/.test(candidate)) throw new Error('Runtime API base URL is invalid.')

    if (candidate.startsWith('/') && !candidate.startsWith('//')) {
      if (candidate.includes('?') || candidate.includes('#')) throw new Error('Runtime API base URL is invalid.')
      return candidate.replace(/\/+$/, '') || '/'
    }

    const url = new URL(candidate)
    if (
      (url.protocol !== 'http:' && url.protocol !== 'https:')
      || !url.host
      || url.username
      || url.password
      || url.search
      || url.hash
    ) {
      throw new Error('Runtime API base URL is invalid.')
    }
    return candidate.replace(/\/+$/, '')
  }
}
