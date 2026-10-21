// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import '@angular/compiler'
import { HttpClient } from '@angular/common/http'
import { Injector, runInInjectionContext } from '@angular/core'
import { describe, expect, it, vi } from 'vitest'
import { of } from 'rxjs'
import { CustomNodeApiService } from './custom-node-api.service'
import { ApiService } from './api.service'
import { RuntimeApiConfigService } from './runtime-api-config.service'

const defaultRuntimeConfig = { provide: RuntimeApiConfigService, useValue: { apiBaseUrl: '/api' } }

describe('ApiService dataset request timeouts', () => {
  it('keeps the 10 second default and opts dataset operations into the longer timeout', async () => {
    const request = vi.fn(() => of({}))
    const injector = Injector.create({
      providers: [{ provide: HttpClient, useValue: { request } }, defaultRuntimeConfig],
    })
    const service = runInInjectionContext(injector, () => new ApiService())

    await service.listProjects()
    await service.inspectDataset({} as never)
    await service.validateDataset({} as never)
    await service.createDataset({} as never)
    await service.updateDataset('dataset-1', {} as never)
    await service.revalidateDataset('dataset-1')

    expect(request.mock.calls[0][2]).toEqual(expect.objectContaining({ timeout: 10_000 }))
    for (const call of request.mock.calls.slice(1)) {
      expect(call[2]).toEqual(expect.objectContaining({ timeout: 60_000 }))
    }
  })

  it('gives whole-graph validation time to infer shapes and keeps connector checks responsive', async () => {
    const request = vi.fn(() => of({}))
    const injector = Injector.create({
      providers: [{ provide: HttpClient, useValue: { request } }, defaultRuntimeConfig],
    })
    const service = runInInjectionContext(injector, () => new ApiService())

    await service.validate({} as never, 'pytorch', true)
    await service.validateConnection({} as never, {} as never, 'pytorch', true)

    expect(request.mock.calls[0][2]).toEqual(expect.objectContaining({ timeout: 60_000 }))
    expect(request.mock.calls[1][2]).toEqual(expect.objectContaining({ timeout: 10_000 }))
  })

  it('extends timeouts for architecture imports and artifact downloads', async () => {
    const post = vi.fn(() => of({}))
    const get = vi.fn(() => of(new Blob()))
    const injector = Injector.create({
      providers: [{ provide: HttpClient, useValue: { post, get } }, defaultRuntimeConfig],
    })
    const service = runInInjectionContext(injector, () => new ApiService())

    await service.importKerasArchitecture(new Blob(['keras']), 'pytorch')
    await service.jobArtifact('job-1')

    expect(post.mock.calls[0][2]).toEqual(expect.objectContaining({ timeout: 60_000 }))
    expect(get.mock.calls[0][1]).toEqual(expect.objectContaining({ timeout: 60_000 }))
  })

  it('uses the runtime API base URL for general and custom-node API clients', async () => {
    const request = vi.fn(() => of({ definitions: [] }))
    const injector = Injector.create({
      providers: [
        { provide: HttpClient, useValue: { request } },
        { provide: RuntimeApiConfigService, useValue: { apiBaseUrl: 'https://api.example.test/v2' } },
      ],
    })
    const api = runInInjectionContext(injector, () => new ApiService())
    const customNodes = runInInjectionContext(injector, () => new CustomNodeApiService())

    await api.listProjects()
    await customNodes.list()

    expect(request.mock.calls[0][1]).toBe('https://api.example.test/v2/projects')
    expect(request.mock.calls[1][1]).toBe('https://api.example.test/v2/nodes/custom')
  })
})
