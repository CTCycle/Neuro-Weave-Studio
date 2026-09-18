import { CommonModule } from '@angular/common'
import { Component, effect, inject, signal } from '@angular/core'
import { LucideCheck, LucideSave } from '@lucide/angular'
import { extractDatasetValidation } from './dataset-wizard.component'
import { DatasetWorkspaceState } from './dataset-workspace.state'
import type { DatasetDefinitionMutation, DatasetDocument, FileTableRelationship } from '../../domain/dataset.types'
import { bindingSummary, bindingsFor, buildDatasetFields, definitionWithRoles, roleAssignmentsFromDocument, type DatasetField, type FieldRole } from './dataset-definition-utils'

@Component({
  selector: 'nw-dataset-io-page',
  standalone: true,
  imports: [CommonModule, LucideCheck, LucideSave],
  template: `
    @if (state.dataset(); as dataset) {
      <div class="workspace-section io-page">
        <section class="io-header panel">
          <div><span class="eyebrow">MODEL BOUNDARY</span><h2>Configure I/O</h2><p>Assign inspected fields to Input, Output, Metadata, or Unused. Structural columns stay outside the model boundary.</p></div>
          <button class="primary" type="button" [disabled]="busy()" (click)="save()"><svg lucideSave [size]="14"></svg>{{ busy() ? 'Saving…' : 'Save I/O configuration' }}</button>
        </section>
        @if (notice()) { <p class="success-notice" role="status"><svg lucideCheck [size]="14"></svg>{{ notice() }}</p> }
        @if (error()) { <p class="page-error" role="alert">{{ error() }}</p> }
        @if (validation(); as result) {
          <section class="inline-validation panel"><strong>Definition was not saved</strong>@for (finding of result.validation.errors; track finding.code + finding.message) { <p><b>{{ finding.code }}</b> · {{ finding.message }}</p> }</section>
        }

        <section class="field-panel panel">
          <header><div><span class="eyebrow">INSPECTED FIELDS</span><h3>{{ fields().length }} available field{{ fields().length === 1 ? '' : 's' }}</h3></div><span class="field-legend">Input · Output / target · Metadata · Unused</span></header>
          @if (!fields().length) { <p class="muted">No inspection metadata is available for this dataset.</p> } @else {
            <div class="role-list">
              @for (field of fields(); track field.key) {
                <div class="role-row" [class.structural]="field.structural">
                  <div class="role-field"><strong>{{ field.label }}</strong><span>{{ sourceLabel(field.sourceId) }}{{ fieldDetails(field) }}</span></div>
                  <select [value]="role(field.key)" [disabled]="field.structural" [attr.aria-label]="'Role for ' + field.label" (change)="setRole(field.key, $any($event.target).value)"><option value="input">Input</option><option value="output">Output / target</option><option value="metadata">Metadata</option><option value="unused">Unused</option></select>
                </div>
              }
            </div>
          }
        </section>

        @if (relationshipRequired()) {
          <section class="structure-panel panel"><span class="eyebrow">SOURCE RELATIONSHIP</span><h3>File-to-table mapping</h3><p class="muted">The identifier column is structural and cannot also be a model field.</p><div class="option-grid"><label><span>File identifier</span><select [value]="relationshipKey()" (change)="setRelationshipKey($any($event.target).value)"><option value="filename">Full filename</option><option value="stem">Filename stem</option><option value="relative_path">Relative path</option></select></label><label><span>Table identifier column</span><select [value]="relationshipColumn()" (change)="setRelationshipColumn($any($event.target).value)">@for (column of tableColumns(); track column) { <option [value]="column">{{ column }}</option> }</select></label></div></section>
        }
        @if (dataset.dataset_type === 'time_series') {
          <section class="structure-panel panel"><span class="eyebrow">SEQUENCE STRUCTURE</span><h3>Time-series columns</h3><p class="muted">Timestamp and group columns are persisted structure, not model I/O.</p><div class="option-grid"><label><span>Timestamp column</span><select [value]="timestampColumn()" (change)="setTimestampColumn($any($event.target).value)"><option value="">None</option>@for (column of tableColumns(); track column) { <option [value]="column">{{ column }}</option> }</select></label><label><span>Group / sequence column</span><select [value]="groupColumn()" (change)="setGroupColumn($any($event.target).value)"><option value="">None</option>@for (column of tableColumns(); track column) { <option [value]="column">{{ column }}</option> }</select></label></div></section>
        }
        <section class="io-summary panel"><div><span class="eyebrow">INPUT</span><strong>{{ summary('input') }}</strong></div><div><span class="eyebrow">OUTPUT</span><strong>{{ summary('output') }}</strong></div><div><span class="eyebrow">METADATA</span><strong>{{ summary('metadata') }}</strong></div></section>
      </div>
    }
  `,
  styles: [`
    .workspace-section{display:grid;gap:14px}.io-header{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;padding:16px}.io-header h2{margin:5px 0}.io-header p{margin:0;color:var(--muted);font-size:10px;line-height:1.45}.io-header button{display:inline-flex;align-items:center;gap:7px;flex:0 0 auto}.field-panel{padding:15px}.field-panel header{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin-bottom:10px}.field-panel h3,.structure-panel h3{margin:5px 0 0;font-size:14px}.field-legend,.muted{color:var(--muted);font-size:9px}.role-list{display:grid;gap:4px}.role-row{display:grid;grid-template-columns:minmax(0,1fr) 170px;align-items:center;gap:12px;padding:9px;border-bottom:1px solid var(--border)}.role-row.structural{background:rgba(229,183,95,.04)}.role-field{display:grid;gap:3px;min-width:0}.role-field strong{font-size:10px;overflow-wrap:anywhere}.role-field span{color:var(--muted);font-size:8px}.role-row select,.option-grid select{height:31px;border:1px solid var(--border);border-radius:5px;background:#091117;color:var(--text);padding:0 8px;font:inherit}.structure-panel{display:grid;gap:7px;padding:15px}.structure-panel p{margin:0}.option-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.option-grid label{display:grid;gap:5px;color:var(--muted);font-size:9px}.success-notice,.page-error{display:flex;align-items:center;gap:7px;margin:0;padding:9px;border-radius:5px;font-size:10px}.success-notice{border:1px solid rgba(99,202,143,.45);background:rgba(99,202,143,.07);color:var(--success,#63ca8f)}.page-error{border:1px solid rgba(239,106,114,.45);background:rgba(239,106,114,.08);color:var(--error)}.inline-validation{display:grid;gap:5px;padding:11px;border-color:rgba(239,106,114,.45);font-size:9px}.inline-validation p{margin:0;color:var(--muted)}.inline-validation b{color:var(--error)}.io-summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;padding:13px}.io-summary>div{display:grid;gap:5px;padding:9px;border-right:1px solid var(--border)}.io-summary>div:last-child{border-right:0}.io-summary strong{font-size:11px}@media(max-width:700px){.io-header{display:grid}.field-panel header{display:grid}.role-row{grid-template-columns:1fr}.option-grid,.io-summary{grid-template-columns:1fr}.io-summary>div{border-right:0;border-bottom:1px solid var(--border)}.io-summary>div:last-child{border-bottom:0}}
  `],
})
export class DatasetIoPageComponent {
  readonly state = inject(DatasetWorkspaceState)
  readonly busy = signal(false)
  readonly error = signal<string | null>(null)
  readonly notice = signal<string | null>(null)
  readonly validation = signal<ReturnType<typeof extractDatasetValidation>>(null)
  readonly assignments = signal<Record<string, FieldRole>>({})
  readonly relationshipKey = signal<FileTableRelationship['file_key']>('filename')
  readonly relationshipColumn = signal('')
  readonly timestampColumn = signal('')
  readonly groupColumn = signal('')
  readonly fields = signal<DatasetField[]>([])
  private loadedId: string | null = null

  constructor() {
    effect(() => {
      const dataset = this.state.dataset()
      if (dataset && dataset.id !== this.loadedId) this.hydrate(dataset)
    })
  }

  role(key: string): FieldRole { return this.assignments()[key] ?? 'unused' }
  fieldDetails(field: DatasetField): string { return `${field.dtype ? ` · ${field.dtype}` : ''}${field.structural ? ' · structural key' : ''}` }
  setRole(key: string, value: string): void { const field = this.fields().find((item) => item.key === key); if (field?.structural) return; this.assignments.update((current) => ({ ...current, [key]: value as FieldRole })); this.notice.set(null); this.validation.set(null) }
  sourceLabel(sourceId: string): string { const source = this.state.dataset()?.sources.find((item) => item.id === sourceId); if (!source) return sourceId; if (source.type === 'table') return `${source.format.toUpperCase()} table`; if (source.type === 'image_folder') return 'Image folder'; if (source.type === 'text') return source.path_kind === 'folder' ? 'Text folder' : 'Text file'; return `${source.format.toUpperCase()} array` }
  tableColumns(): string[] { return this.state.dataset()?.inspection.sources.filter((source): source is Extract<typeof source, { type: 'table' }> => source.type === 'table').flatMap((source) => source.columns.map((column) => column.name)) ?? [] }
  relationshipRequired(): boolean { const dataset = this.state.dataset(); return Boolean(dataset && (dataset.dataset_type === 'composed' || (dataset.dataset_type === 'images' && dataset.sources.some((source) => source.type === 'table')))) }
  setRelationshipKey(value: string): void { this.relationshipKey.set(value as FileTableRelationship['file_key']); this.notice.set(null) }
  setRelationshipColumn(value: string): void { this.relationshipColumn.set(value); this.markStructural(value); this.notice.set(null) }
  setTimestampColumn(value: string): void { this.timestampColumn.set(value); this.markStructural(value); this.notice.set(null) }
  setGroupColumn(value: string): void { this.groupColumn.set(value); this.markStructural(value); this.notice.set(null) }
  summary(role: FieldRole): string { const dataset = this.state.dataset(); return dataset ? bindingSummary(bindingsFor(this.fields(), dataset.sources, this.assignments(), role)) || 'None' : 'None' }

  async save(): Promise<void> {
    const dataset = this.state.dataset()
    if (!dataset || this.busy()) return
    this.busy.set(true); this.error.set(null); this.notice.set(null); this.validation.set(null)
    try {
      const saved = await this.state.updateDefinition(this.buildDefinition(dataset))
      this.hydrate(saved)
      this.notice.set('I/O configuration saved.')
    } catch (reason) {
      const validation = extractDatasetValidation(reason)
      if (validation) this.validation.set(validation)
      this.error.set(validation ? 'The stored definition was not changed. Review the findings below.' : reason instanceof Error ? reason.message : String(reason))
    } finally { this.busy.set(false) }
  }

  private hydrate(dataset: DatasetDocument): void { this.loadedId = dataset.id; this.relationshipKey.set(dataset.relationships[0]?.file_key ?? 'filename'); this.relationshipColumn.set(dataset.relationships[0]?.table_key_column ?? ''); this.timestampColumn.set(dataset.sequence_definition?.timestamp_column ?? ''); this.groupColumn.set(dataset.sequence_definition?.group_column ?? ''); const fields = buildDatasetFields(dataset.inspection, { structuralKeys: this.structuralKeys(dataset) }); this.fields.set(fields); this.assignments.set(roleAssignmentsFromDocument(fields, dataset.dataset_type, dataset)) }
  private structuralKeys(dataset: DatasetDocument): Set<string> { const tableId = dataset.sources.find((source) => source.type === 'table')?.id ?? ''; return new Set([this.relationshipColumn(), this.timestampColumn(), this.groupColumn()].filter(Boolean).map((column) => `${tableId}::${column}`)) }
  private markStructural(column: string): void { const dataset = this.state.dataset(); if (!dataset || !column) return; const tableId = dataset.sources.find((source) => source.type === 'table')?.id ?? ''; this.assignments.update((current) => ({ ...current, [`${tableId}::${column}`]: 'unused' })); this.fields.set(buildDatasetFields(dataset.inspection, { structuralKeys: this.structuralKeys(dataset) })) }
  private buildDefinition(dataset: DatasetDocument): DatasetDefinitionMutation { const payload = definitionWithRoles(dataset, dataset.sources, this.fields(), this.assignments()); const relationship = dataset.relationships[0]; if (this.relationshipRequired() && relationship) payload.relationships = [{ ...relationship, file_key: this.relationshipKey(), table_key_column: this.relationshipColumn() }]; payload.sequence_definition = dataset.dataset_type === 'time_series' ? { timestamp_column: this.timestampColumn() || null, group_column: this.groupColumn() || null } : null; return payload }
}
