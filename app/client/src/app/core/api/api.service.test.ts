import '@angular/compiler'
import { HttpClient } from '@angular/common/http'
import { Injector, runInInjectionContext } from '@angular/core'
import { describe, expect, it, vi } from 'vitest'
import { of } from 'rxjs'
import { ApiService } from './api.service'

describe('ApiService dataset request timeouts', () => {
  it('keeps the 10 second default and opts dataset operations into the longer timeout', async () => {
    const request = vi.fn(() => of({}))
    const injector = Injector.create({
      providers: [{ provide: HttpClient, useValue: { request } }],
    })
    const service = runInInjectionContext(injector, () => new ApiService())

    await service.listProjects()
    await service.inspectDataset({} as never)
    await service.validateDataset({} as never)
    await service.createDataset({} as never)
    await service.updateDataset('dataset-1', {} as never)

    expect(request.mock.calls[0][2]).toEqual(expect.objectContaining({ timeout: 10_000 }))
    for (const call of request.mock.calls.slice(1)) {
      expect(call[2]).toEqual(expect.objectContaining({ timeout: 60_000 }))
    }
  })

  it('extends timeouts for architecture imports and artifact downloads', async () => {
    const post = vi.fn(() => of({}))
    const get = vi.fn(() => of(new Blob()))
    const injector = Injector.create({
      providers: [{ provide: HttpClient, useValue: { post, get } }],
    })
    const service = runInInjectionContext(injector, () => new ApiService())

    await service.importKerasArchitecture(new Blob(['keras']), 'pytorch')
    await service.jobArtifact('job-1')

    expect(post.mock.calls[0][2]).toEqual(expect.objectContaining({ timeout: 60_000 }))
    expect(get.mock.calls[0][1]).toEqual(expect.objectContaining({ timeout: 60_000 }))
  })
})
