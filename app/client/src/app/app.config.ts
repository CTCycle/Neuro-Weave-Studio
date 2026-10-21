// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { provideHttpClient } from '@angular/common/http'
import { inject, provideAppInitializer, provideZonelessChangeDetection } from '@angular/core'
import type { ApplicationConfig } from '@angular/core'
import { provideRouter, withComponentInputBinding } from '@angular/router'
import { RuntimeApiConfigService } from './core/api/runtime-api-config.service'
import { routes } from './app.routes'

export const appConfig: ApplicationConfig = {
  providers: [
    provideZonelessChangeDetection(),
    provideRouter(routes, withComponentInputBinding()),
    provideHttpClient(),
    provideAppInitializer(() => inject(RuntimeApiConfigService).load()),
  ],
}
