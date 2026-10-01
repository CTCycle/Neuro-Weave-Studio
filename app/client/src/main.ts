// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { bootstrapApplication } from '@angular/platform-browser'
import { AppComponent } from './app/app'
import { appConfig } from './app/app.config'

performance.mark('nw:bootstrap:start')
bootstrapApplication(AppComponent, appConfig).then(() => {
  performance.mark('nw:bootstrap:complete')
}).catch((error: unknown) => {
  console.error(error)
  const message = error instanceof Error ? error.message : String(error)
  const diagnostic = document.createElement('pre')
  diagnostic.setAttribute('role', 'alert')
  diagnostic.textContent = `NeuroWeave could not start. ${message}`
  diagnostic.style.cssText = 'margin:2rem;padding:1rem;color:#ef6a72;background:#111a22;border:1px solid #ef6a72;white-space:pre-wrap;font:14px/1.5 monospace;'
  document.body.replaceChildren(diagnostic)
})
