import { inject } from '@angular/core'
import type { ResolveFn } from '@angular/router'
import { AppResourcesService } from '../state/app-resources.service'

export const builderResourcesResolver: ResolveFn<boolean> = async () => {
  const resources = inject(AppResourcesService)
  await Promise.all([
    resources.loadCatalog(),
    resources.loadRuntimeMetadata(),
  ])
  return true
}
