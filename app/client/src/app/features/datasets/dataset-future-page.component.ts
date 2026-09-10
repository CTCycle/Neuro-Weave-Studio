import { Component, inject } from '@angular/core'
import { ActivatedRoute } from '@angular/router'
import { LucideInfo } from '@lucide/angular'
import { DatasetWorkspaceState } from './dataset-workspace.state'

@Component({
  selector: 'nw-dataset-future-page',
  standalone: true,
  imports: [LucideInfo],
  template: `
    @if (state.dataset(); as dataset) {
      <section class="future-page panel"><span class="eyebrow">{{ mode === 'processing' ? 'PROCESSING' : 'SPLIT' }} FOUNDATION</span><h2>{{ mode === 'processing' ? 'Processing is prepared for a later phase' : 'Split configuration is prepared for a later phase' }}</h2><p>{{ mode === 'processing' ? 'This workspace will expose transformations appropriate to ' + dataset.dataset_type + ' datasets after the metadata definition is stable.' : 'Train, validation, and test materialization will be added after dataset definitions can feed a future training workflow.' }}</p><div class="future-note"><svg lucideInfo [size]="16"></svg><span>No files are modified, no split is generated, and no processing state is persisted by this foundation.</span></div></section>
    }
  `,
  styles: [`
    .future-page{display:grid;gap:10px;max-width:760px;padding:20px}.future-page h2{margin:0}.future-page p{margin:0;color:var(--muted);font-size:10px;line-height:1.5}.future-note{display:flex;gap:8px;padding:11px;border:1px solid var(--border);color:var(--muted);font-size:9px;line-height:1.45}.future-note svg{flex:0 0 auto;color:var(--cyan)}
  `],
})
export class DatasetFuturePageComponent {
  readonly state = inject(DatasetWorkspaceState)
  readonly mode = (inject(ActivatedRoute).snapshot.data['mode'] as 'processing' | 'split' | undefined) ?? 'processing'
}
