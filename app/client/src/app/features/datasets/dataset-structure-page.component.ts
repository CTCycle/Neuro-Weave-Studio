import { CommonModule } from '@angular/common'
import { Component, inject } from '@angular/core'
import { DatasetWorkspaceState } from './dataset-workspace.state'
import type { DatasetBinding } from '../../domain/dataset.types'

@Component({
  selector: 'nw-dataset-structure-page',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (state.dataset(); as dataset) {
      <div class="structure-page workspace-section">
        @for (source of dataset.inspection.sources; track source.source_id) {
          <section class="structure-panel panel">
            <header><div><span class="eyebrow">{{ source.type }}</span><h2>{{ source.source_id }}</h2></div><code>{{ source.path }}</code></header>
            @if (source.type === 'table') { <div class="schema-table"><div class="schema-row schema-header"><span>Column</span><span>Type</span><span>Nullable</span><span>Role</span></div>@for (column of source.columns; track column.name) { <div class="schema-row"><strong>{{ column.name }}</strong><span>{{ column.dtype }}</span><span>{{ column.nullable ? 'Yes' : 'No' }}</span><span>{{ roleFor(dataset, source.source_id, column.name) }}</span></div> }</div><p class="structure-note">Dtype inference sampled {{ source.inference_sample_size | number }} row{{ source.inference_sample_size === 1 ? '' : 's' }}.</p> }
            @if (source.type === 'image_folder') { <div class="structure-stats"><div><span>Samples</span><strong>{{ source.sample_count | number }}</strong></div><div><span>Extensions</span><strong>{{ source.extensions.join(', ') || 'None' }}</strong></div><div><span>Classes</span><strong>{{ source.class_names.join(', ') || 'Unlabeled' }}</strong></div></div>@if (source.class_names.length) { <div class="class-list">@for (className of source.class_names; track className) { <span>{{ className }} · {{ source.class_counts[className] | number }}</span> }</div> } }
            @if (source.type === 'text') { <div class="structure-stats"><div><span>Samples</span><strong>{{ source.sample_count | number }}</strong></div><div><span>Extensions</span><strong>{{ source.extensions.join(', ') || 'None' }}</strong></div></div> }
            @if (source.type === 'array') { <div class="schema-table"><div class="schema-row schema-header"><span>Key</span><span>Shape</span><span>Dtype</span><span>Samples</span></div>@for (array of source.arrays; track array.key) { <div class="schema-row"><strong>{{ array.key }}</strong><span>[{{ array.shape.join(' × ') }}]</span><span>{{ array.dtype }}</span><span>{{ source.sample_count | number }}</span></div> }</div> }
          </section>
        }
        @if (dataset.relationships.length) { <section class="structure-panel panel"><span class="eyebrow">RELATIONSHIP</span><h2>File-to-table mapping</h2><div class="relationship-summary"><span>{{ dataset.relationships[0].file_source_id }}</span><strong>{{ dataset.relationships[0].file_key }}</strong><span>→</span><span>{{ dataset.relationships[0].table_source_id }}.{{ dataset.relationships[0].table_key_column }}</span></div></section> }
        @if (dataset.sequence_definition) { <section class="structure-panel panel"><span class="eyebrow">SEQUENCE STRUCTURE</span><div class="relationship-summary"><span>Timestamp</span><strong>{{ dataset.sequence_definition.timestamp_column || 'None' }}</strong><span>Group</span><strong>{{ dataset.sequence_definition.group_column || 'None' }}</strong></div></section> }
      </div>
    }
  `,
  styles: [`
    .workspace-section{display:grid;gap:14px}.structure-panel{min-width:0;padding:15px}.structure-panel header{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;margin-bottom:13px}.structure-panel h2{margin:5px 0 0}.structure-panel header code{max-width:55%;overflow-wrap:anywhere;color:var(--muted);font-size:9px}.schema-table{display:grid;gap:3px}.schema-row{display:grid;grid-template-columns:minmax(140px,1.2fr) minmax(100px,1fr) minmax(70px,.6fr) minmax(100px,.8fr);gap:10px;align-items:center;padding:8px;border-bottom:1px solid var(--border);font-size:9px}.schema-row span{color:var(--muted);overflow-wrap:anywhere}.schema-header{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase}.schema-header span{color:var(--muted)}.structure-note{margin:10px 0 0;color:var(--muted);font-size:9px}.structure-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.structure-stats>div{display:grid;gap:4px;padding:10px;border:1px solid var(--border);border-radius:5px;background:#0c141b}.structure-stats span{color:var(--muted);font-size:8px}.structure-stats strong{overflow-wrap:anywhere;font-size:10px}.class-list{display:flex;flex-wrap:wrap;gap:6px;margin-top:11px}.class-list span{padding:6px 8px;border:1px solid var(--border);border-radius:4px;color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace}.relationship-summary{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:10px;font-size:10px}.relationship-summary span{color:var(--muted)}@media(max-width:720px){.structure-panel header{display:grid}.structure-panel header code{max-width:none}.schema-row{grid-template-columns:1fr 1fr}.schema-header span:nth-child(n+3),.schema-row span:nth-child(n+3){display:none}.structure-stats{grid-template-columns:1fr}}
  `],
})
export class DatasetStructurePageComponent {
  readonly state = inject(DatasetWorkspaceState)

  roleFor(dataset: { input_definition: { bindings: DatasetBinding[] }; output_definition: { bindings: DatasetBinding[] }; metadata_definition: { bindings: DatasetBinding[] } }, sourceId: string, key: string): string {
    const has = (bindings: DatasetBinding[]): boolean => bindings.some((binding) => binding.type === 'columns' && binding.source_id === sourceId && binding.columns.includes(key))
    if (has(dataset.input_definition.bindings)) return 'Input'
    if (has(dataset.output_definition.bindings)) return 'Output'
    if (has(dataset.metadata_definition.bindings)) return 'Metadata'
    return 'Unused'
  }
}
