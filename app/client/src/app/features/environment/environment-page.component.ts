import { Component, inject } from '@angular/core'
import { LucideCpu, LucideInfo } from '@lucide/angular'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { BackendStatusPanelComponent } from './backend-status-panel.component'

@Component({
  selector: 'nw-environment-page',
  standalone: true,
  imports: [LucideCpu, LucideInfo, BackendStatusPanelComponent],
  template: `
    <div class="environment-page"><div class="environment-header"><div><span class="eyebrow">RUNTIME ENVIRONMENT</span><h1>Execution backends</h1><p>Configure the API, device, and memory policy preference stored for each execution backend.</p></div><div class="environment-summary"><svg lucideCpu [size]="16"></svg><strong>{{ resources.runtimeReady() ? operationalCount() + ' operational' : 'Reading runtime…' }}</strong></div></div><div class="environment-note panel"><svg lucideInfo [size]="16"></svg><span>Availability comes from isolated worker discovery. Matching environment snapshots are reused until packages, hardware, or backend configuration changes. Each model uses the preference for its persisted backend.</span></div><nw-backend-status-panel />@if (resources.hasError('runtime')) { <section class="page-state panel" role="alert"><h2>Runtime metadata unavailable</h2><p>The environment request failed while the rest of the application remains available.</p><button class="secondary" type="button" (click)="resources.retryRuntimeMetadata()">Retry runtime metadata</button></section> } @else if (!resources.backendStatuses().length && resources.runtimeReady()) { <section class="page-state panel"><h2>No execution metadata returned</h2><p>The backend did not report any operational profiles.</p></section> }</div>
  `,
})
export class EnvironmentPageComponent {
  readonly resources = inject(AppResourcesService)
  constructor() { void this.resources.loadRuntimeMetadata() }
  operationalCount(): number { return this.resources.backendStatuses().filter((status) => status.available).length }
  availableApis(status: { runtime_apis?: Array<{ id: string; available: boolean }> }): string { return status.runtime_apis?.filter((api) => api.available).map((api) => api.id).join(' · ') ?? '' }
}
