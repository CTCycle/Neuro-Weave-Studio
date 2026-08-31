import { Component, inject } from '@angular/core'
import { LucideCpu, LucideInfo } from '@lucide/angular'
import { AppResourcesService } from '../../core/state/app-resources.service'

@Component({
  selector: 'nw-environment-page',
  standalone: true,
  imports: [LucideCpu, LucideInfo],
  template: `
    <div class="environment-page"><div class="environment-header"><div><span class="eyebrow">RUNTIME ENVIRONMENT</span><h1>Execution backends</h1><p>Configure the API, device, and memory policy preference stored for each execution backend.</p></div><div class="environment-summary"><svg lucideCpu [size]="16"></svg><strong>{{ resources.runtimeReady() ? operationalCount() + ' operational' : 'Reading runtime…' }}</strong></div></div><div class="environment-note panel"><svg lucideInfo [size]="16"></svg><span>Availability comes from isolated worker discovery. Matching environment snapshots are reused until packages, hardware, or backend configuration changes. Each model uses the preference for its persisted backend.</span></div><div class="backend-status-grid">@for (status of resources.backendStatuses(); track status.id) { <section class="backend-status-card panel" [class.is-available]="status.available"><div class="section-heading"><div><span class="eyebrow">EXECUTION BACKEND</span><h2>{{ status.id }}</h2></div><span class="status-indicator" [class.valid]="status.available">{{ status.available ? 'Operational' : 'Unavailable' }}</span></div><dl class="detail-list"><div><dt>Installed</dt><dd>{{ status.installed ? 'Yes' : 'No' }}{{ status.version ? ' · ' + status.version : '' }}</dd></div><div><dt>Accelerator</dt><dd>{{ status.accelerator || 'cpu' }}</dd></div><div><dt>Device</dt><dd>{{ status.device_name || status.devices?.[0]?.name || 'No device reported' }}</dd></div><div><dt>Runtime API</dt><dd>{{ availableApis(status) || 'None available' }}</dd></div></dl>@if (!status.available) { <p class="muted">{{ status.reason_code || 'Runtime unavailable for this host.' }}</p> } </section> } @if (!resources.backendStatuses().length && resources.runtimeReady()) { <section class="page-state panel"><h2>No execution metadata returned</h2><p>The backend did not report any operational profiles.</p></section> }</div></div>
  `,
})
export class EnvironmentPageComponent {
  readonly resources = inject(AppResourcesService)
  operationalCount(): number { return this.resources.backendStatuses().filter((status) => status.available).length }
  availableApis(status: { runtime_apis?: Array<{ id: string; available: boolean }> }): string { return status.runtime_apis?.filter((api) => api.available).map((api) => api.id).join(' · ') ?? '' }
}
