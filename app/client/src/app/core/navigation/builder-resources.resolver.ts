// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { inject } from '@angular/core'
import type { ResolveFn } from '@angular/router'
import { AppResourcesService } from '../state/app-resources.service'

export const builderResourcesResolver: ResolveFn<boolean> = async () => {
  const resources = inject(AppResourcesService)
  await Promise.all([
    resources.loadCatalog(),
    resources.loadRuntimeMetadata(),
  ])
  if (resources.hasError('catalog') || resources.hasError('runtime')) {
    throw new Error('The Builder requires the node catalog and runtime metadata. Retry when the API is reachable.')
  }
  await resources.ensureBuilderRuntime()
  return true
}
