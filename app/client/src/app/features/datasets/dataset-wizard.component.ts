// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { CommonModule } from '@angular/common'
import { Component, EventEmitter, Input, Output, computed, inject, signal } from '@angular/core'
import type { OnChanges, SimpleChanges } from '@angular/core'
import { LucideAlertTriangle, LucideCheck, LucideFilePlus2, LucidePlus, LucideX } from '@lucide/angular'
import { ApiError } from '../../core/api/api-error'
import { ApiService } from '../../core/api/api.service'
import { FilesystemBrowserComponent } from '../../shared/components/filesystem-browser.component'
import type {
  ArraySource,
  DatasetDefinitionMutation,
  DatasetDocument,
  DatasetInspection,
  DatasetSource,
  DatasetSourceInspection,
  DatasetType,
  FileTableRelationship,
  ImageFolderSource,
  TableSource,
  TextSource,
  DatasetValidationResult,
} from '../../domain/dataset.types'
import { DATASET_CAPABILITIES, arrayFormatFromPath, datasetCapability, sourceExtensions, sourceMode, tableFormatFromPath } from '../../domain/dataset-capabilities'
import { ModalGuardDirective } from '../../shared/components/modal-guard.directive'
import { bindingSummary, bindingsFor, buildDatasetFields, defaultFieldRole, type FieldRole } from './dataset-definition-utils'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

export function extractDatasetValidation(reason: unknown): DatasetValidationResult | null {
  if (!(reason instanceof ApiError) || reason.status !== 422 || !isRecord(reason.detail)) return null
  if (reason.detail['code'] !== 'dataset_definition_invalid' || !isRecord(reason.detail['validation'])) return null
  const result = reason.detail['validation']
  const metadata = isRecord(result) ? result['validation'] : null
  if (
    !isRecord(result)
    || typeof result['valid'] !== 'boolean'
    || typeof result['sample_count'] !== 'number'
    || typeof result['selected_sample_count'] !== 'number'
    || (result['inspection'] !== null && !isRecord(result['inspection']))
    || !isRecord(metadata)
    || typeof metadata['status'] !== 'string'
    || typeof metadata['checked_at'] !== 'string'
    || !Array.isArray(metadata['errors'])
    || !Array.isArray(metadata['warnings'])
  ) return null
  return result as unknown as DatasetValidationResult
}

@Component({
  selector: 'nw-dataset-wizard',
  standalone: true,
  imports: [CommonModule, LucideAlertTriangle, LucideCheck, LucideFilePlus2, LucidePlus, LucideX, FilesystemBrowserComponent, ModalGuardDirective],
  template: `
    @if (open) {
      <div class="dialog-backdrop" nwModalGuard (click)="close()">
        <section class="dataset-wizard panel" role="dialog" aria-modal="true" aria-labelledby="dataset-wizard-title" (click)="$event.stopPropagation()" (keydown.escape)="close()">
          <header class="wizard-header">
            <div><span class="eyebrow">NEW DATASET · STEP {{ step() + 1 }} OF {{ steps.length }}</span><h2 id="dataset-wizard-title">{{ steps[step()] }}</h2></div>
            <button class="icon-button" type="button" aria-label="Close dataset wizard" (click)="close()"><svg lucideX [size]="16"></svg></button>
          </header>
          <div class="wizard-progress" aria-hidden="true">@for (label of steps; track label; let index = $index) { <span [class.active]="index <= step()"></span> }</div>

          <div class="wizard-body">
            @if (step() === 0) {
              <div class="wizard-intro"><span class="eyebrow">DATASET AND SOURCES</span><p>Choose the organization and point NeuroWeave at external data. The selected filename determines unambiguous table and array formats.</p></div>
              <div class="dataset-type-grid">
                @for (type of datasetTypes; track type.id) {
                  <button type="button" class="dataset-type-option" [class.selected]="type.id === datasetType()" [class.disabled]="!type.enabled" [attr.aria-pressed]="type.id === datasetType()" [disabled]="!type.enabled" (click)="selectType(type.id)">
                    <span class="dataset-type-icon">{{ type.enabled ? '◇' : '…' }}</span><strong>{{ type.label }}</strong><span>{{ type.description }}</span><small>{{ type.sourceLabel }}</small>
                  </button>
                }
              </div>
              <div class="wizard-fields">
                <label><span>Dataset name</span><input [value]="name()" placeholder="e.g. Patient outcomes" (input)="setName($any($event.target).value)" /></label>
                @if (datasetType() === 'composed') { <div class="inline-choice"><span>Primary modality</span><button type="button" class="choice-button" [class.active]="composedMode() === 'images'" [attr.aria-pressed]="composedMode() === 'images'" (click)="setComposedMode('images')">Images + table</button><button type="button" class="choice-button" [class.active]="composedMode() === 'text'" [attr.aria-pressed]="composedMode() === 'text'" (click)="setComposedMode('text')">Text + table</button></div> }
                @if (datasetType() === 'text') { <div class="inline-choice"><span>Interpret text as</span><button type="button" class="choice-button" [class.active]="textTableMode()" [attr.aria-pressed]="textTableMode()" (click)="setTextTableMode(true)">A table</button><button type="button" class="choice-button" [class.active]="!textTableMode()" [attr.aria-pressed]="!textTableMode()" (click)="setTextTableMode(false)">Text files</button></div> }
                <div class="source-stack">
                  @for (source of sources(); track source.id; let index = $index) {
                    <div class="source-card">
                      <div class="source-card-header"><div><span class="eyebrow">SOURCE {{ index + 1 }}</span><strong>{{ sourceTitle(source) }}</strong></div>@if (canRemoveSource()) { <button class="icon-button" type="button" [attr.aria-label]="'Remove ' + sourceTitle(source)" (click)="removeSource(index)"><svg lucideX [size]="14"></svg></button> }</div>
                      <label class="path-field"><span>{{ sourcePathLabel(source) }}</span><div><input readonly [value]="source.path" placeholder="No path selected" /><button class="secondary" type="button" (click)="openBrowser(index)">Browse…</button></div></label>
                      @if (source.type === 'table') {
                        <div class="detected-format"><span>FORMAT</span><strong>{{ source.path ? source.format.toUpperCase() : 'Selected filename will determine format' }}</strong></div>
                        @if (source.path && source.format !== 'parquet') { <div class="option-grid"><label><span>Header row</span><select [value]="source.has_header ? 'yes' : 'no'" (change)="updateTable(index, { has_header: $any($event.target).value === 'yes' })"><option value="yes">First row contains column names</option><option value="no">No header row</option></select></label><label><span>Delimiter</span><select [value]="source.delimiter ?? defaultDelimiter(source.format)" (change)="updateTable(index, { delimiter: $any($event.target).value })"><option value=",">Comma</option><option value="\t">Tab</option><option value=";">Semicolon</option><option value="|">Pipe</option></select></label></div> }
                      }
                      @if (source.type === 'image_folder') { <div class="option-grid"><label><span>Folder labels</span><select [value]="source.label_mode" (change)="updateImage(index, { label_mode: $any($event.target).value })"><option value="none">No folder labels</option><option value="parent_folder">Parent folder names</option></select></label><label class="check-field"><input type="checkbox" [checked]="source.recursive" (change)="updateImage(index, { recursive: $any($event.target).checked })" /><span>Scan subfolders recursively</span></label></div> }
                      @if (source.type === 'text') { <div class="option-grid"><label><span>Text source</span><select [value]="source.path_kind" (change)="updateText(index, { path_kind: $any($event.target).value })"><option value="file">One text file</option><option value="folder">Text folder</option></select></label><label class="check-field"><input type="checkbox" [checked]="source.recursive" (change)="updateText(index, { recursive: $any($event.target).checked })" /><span>Scan subfolders recursively</span></label></div> }
                      @if (source.type === 'array') { <div class="detected-format"><span>FORMAT</span><strong>{{ source.path ? source.format.toUpperCase() : 'Selected filename will determine format' }}</strong><small>NPY/NPZ headers are inspected without loading array contents.</small></div> }
                    </div>
                  }
                </div>
                @if (datasetType() === 'images') { <button class="secondary add-source" type="button" (click)="toggleImageTable()"><svg lucidePlus [size]="14"></svg>{{ hasLabelTable() ? 'Remove label / metadata table' : 'Add label / metadata table' }}</button> }
                @if (datasetType() === 'array' && sources().length < 3) { <button class="secondary add-source" type="button" (click)="addArraySource()"><svg lucidePlus [size]="14"></svg>Add array source</button> }
              </div>
              @if (error()) { <p class="wizard-error" role="alert">{{ error() }}</p> }
            }

            @if (step() === 1) {
              @if (!inspection()) { <section class="wizard-empty"><span class="eyebrow">INSPECTION REQUIRED</span><h3>Inspect the selected sources first</h3><p>NeuroWeave reads bounded schema and count metadata before it can confirm this registration.</p><button class="secondary" type="button" (click)="goTo(0)">Back to sources</button></section> }
              @else {
                <div class="wizard-intro"><span class="eyebrow">INSPECT AND CONFIRM</span><p>These conservative defaults make the definition valid without guessing targets. Detailed field roles remain editable in the dataset workspace.</p></div>
                <div class="inspection-summary"><div><span>AVAILABLE SAMPLES</span><strong>{{ inspection()!.sample_count | number }}</strong></div><div><span>SOURCES INSPECTED</span><strong>{{ inspection()!.sources.length }}</strong></div><div><span>SCHEMA</span><strong>{{ fieldCount(inspection()!) }} fields</strong></div></div>
                <div class="source-summary-list">@for (source of inspection()!.sources; track source.source_id) { <div><span>{{ source.source_id }}</span><strong>{{ inspectedSourceLabel(source) }}</strong><code>{{ source.path }}</code></div> }</div>
                <div class="confirm-summary"><div><span class="eyebrow">INITIAL MODEL INPUT</span><strong>{{ roleSummary('input') }}</strong><small>All table columns, array keys, image samples, and text samples are treated as inputs. Folder labels are outputs only when parent-folder labels are enabled.</small></div><div><span class="eyebrow">INITIAL MODEL OUTPUT</span><strong>{{ roleSummary('output') }}</strong><small>No column is promoted to a target based on its name.</small></div></div>
                @if (hasRelationship()) { <div class="relationship-card"><span class="eyebrow">REQUIRED SOURCE RELATIONSHIP</span><p>Choose the table identifier that resolves each file to exactly one table row.</p><div class="option-grid"><label><span>File identifier</span><select [value]="relationshipKey()" (change)="setRelationshipKey($any($event.target).value)"><option value="filename">Full filename</option><option value="stem">Filename stem</option><option value="relative_path">Relative path</option></select></label><label><span>Table identifier column</span><select [value]="relationshipColumn()" (change)="setRelationshipColumn($any($event.target).value)">@for (column of tableColumns(); track column) { <option [value]="column">{{ column }}</option> }</select></label></div></div> }
                @if (datasetType() === 'time_series') { <div class="relationship-card"><span class="eyebrow">OPTIONAL SEQUENCE STRUCTURE</span><div class="option-grid"><label><span>Timestamp column</span><select [value]="timestampColumn()" (change)="setTimestampColumn($any($event.target).value)"><option value="">None</option>@for (column of tableColumns(); track column) { <option [value]="column">{{ column }}</option> }</select></label><label><span>Group / sequence column</span><select [value]="groupColumn()" (change)="setGroupColumn($any($event.target).value)"><option value="">None</option>@for (column of tableColumns(); track column) { <option [value]="column">{{ column }}</option> }</select></label></div></div> }
                @if (roleError()) { <p class="wizard-error" role="alert">{{ roleError() }}</p> }
                @if (validation(); as result) { <div class="validation-result" [class.valid]="result.valid" [class.invalid]="!result.valid"><div class="validation-result-title">@if (result.valid) { <svg lucideCheck class="success-icon" [size]="16"></svg> } @else { <svg lucideAlertTriangle class="error-icon" [size]="16"></svg> }<strong>{{ result.valid ? 'Definition is valid' : 'Definition needs attention' }}</strong></div>@for (finding of result.validation.errors; track finding.code + finding.message) { <p><strong>{{ finding.code }}</strong> · {{ finding.message }}</p> }@for (finding of result.validation.warnings; track finding.code + finding.message) { <p class="warning-line"><strong>Warning</strong> · {{ finding.message }}</p> }</div> }
              }
            }

            @if (step() === 2) {
              <section class="wizard-success"><span class="eyebrow">READY TO REGISTER</span><div class="success-mark"><svg lucideCheck [size]="22"></svg></div><h3>Register this dataset definition</h3><p>The final request is re-inspected by the backend before saving. NeuroWeave stores metadata and configuration only; the external data remains where it is.</p><div class="final-summary"><span>{{ name() }}</span><strong>{{ inspection()?.sample_count | number }} available samples · {{ datasetTypeLabel() }}</strong><small>Initial inputs: {{ roleSummary('input') }} · outputs: {{ roleSummary('output') }}</small></div>@if (validation(); as result) { <div class="validation-result valid"><div class="validation-result-title"><svg lucideCheck class="success-icon" [size]="16"></svg><strong>Confirmed against the current source</strong></div><p>{{ result.selected_sample_count | number }} samples will be registered with the default all-samples population.</p></div> }</section>
              @if (error()) { <p class="wizard-error" role="alert">{{ error() }}</p> }
            }
          </div>

          <footer class="dialog-actions wizard-actions"><button class="secondary" type="button" (click)="step() === 0 ? close() : goTo(step() - 1)" [disabled]="busy()">{{ step() === 0 ? 'Cancel' : 'Back' }}</button>@if (step() === 0) { <button class="primary" type="button" [disabled]="busy() || !sourcesReady()" (click)="inspectAndContinue()">{{ busy() ? 'Inspecting…' : 'Inspect sources' }}</button> } @else if (step() === 1) { <button class="primary" type="button" [disabled]="busy() || !canConfirm()" (click)="confirmDefinition()">{{ busy() ? 'Checking…' : 'Inspect & confirm' }}</button> } @else { <button class="primary" type="button" [disabled]="busy() || !canRegister()" (click)="register()"><svg lucideFilePlus2 [size]="15"></svg>{{ busy() ? 'Registering…' : 'Register dataset' }}</button> }</footer>
        </section>
      </div>
      <nw-filesystem-browser [open]="browserOpen()" [title]="browserTitle()" [selectionMode]="browserMode()" [extensions]="browserExtensions()" (closed)="browserOpen.set(false)" (selected)="acceptPath($event)" />
    }
  `,
  styles: [`
    :host{display:block}.dataset-wizard{width:min(900px,calc(100vw - 32px));max-height:calc(100vh - 32px);display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;overflow:hidden;box-shadow:0 24px 70px rgba(0,0,0,.55)}.wizard-header{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;padding:18px 20px;border-bottom:1px solid var(--border)}.wizard-header h2{margin:5px 0 0}.wizard-progress{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;padding:8px 20px}.wizard-progress span{height:2px;background:var(--border)}.wizard-progress span.active{background:var(--cyan)}.wizard-body{min-height:0;overflow:auto;padding:18px 20px 22px;scrollbar-gutter:stable}.wizard-intro{display:grid;gap:6px;margin-bottom:16px}.wizard-intro p,.wizard-empty p,.wizard-success p,.detected-format small,.confirm-summary small{margin:0;color:var(--muted);font-size:10px;line-height:1.5}.dataset-type-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin-bottom:18px}.dataset-type-option{min-width:0;display:grid;grid-template-columns:auto 1fr;gap:5px 9px;padding:12px;border:1px solid var(--border);border-radius:6px;background:#0c141b;color:var(--text);text-align:left}.dataset-type-option:hover,.dataset-type-option.selected{border-color:var(--cyan);background:rgba(101,201,212,.07)}.dataset-type-option.disabled{opacity:.5}.dataset-type-option strong{font-size:11px}.dataset-type-option>span:not(.dataset-type-icon),.dataset-type-option small{grid-column:2;color:var(--muted);font-size:9px;line-height:1.4}.dataset-type-icon{grid-row:span 3;color:var(--cyan);font-size:16px}.wizard-fields,.source-stack{display:grid;gap:12px}.wizard-fields>label,.option-grid label,.inline-number{display:grid;gap:5px;color:var(--muted);font-size:9px}.wizard-fields input,.wizard-fields select,.option-grid select,.option-grid input[type=number],.inline-number input{min-width:0;height:32px;border:1px solid var(--border);border-radius:5px;background:#091117;color:var(--text);padding:0 9px;font:inherit}.source-card{display:grid;gap:11px;padding:13px;border:1px solid var(--border);border-radius:6px;background:#0d151c}.source-card-header{display:flex;justify-content:space-between;gap:10px}.source-card-header>div{display:grid;gap:4px}.path-field>div{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:7px}.path-field input{font-family:'IBM Plex Mono',Consolas,monospace;color:var(--muted)}.option-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.check-field{display:flex!important;align-items:end;gap:7px;padding-bottom:8px}.check-field input{width:14px;height:14px}.inline-choice{display:flex;align-items:center;gap:7px;flex-wrap:wrap;color:var(--muted);font-size:9px}.choice-button{border:1px solid var(--border);border-radius:5px;background:transparent;color:var(--muted);padding:7px 9px;font-size:9px}.choice-button.active{border-color:var(--cyan);background:rgba(101,201,212,.1);color:var(--text)}.add-source{justify-self:start;display:inline-flex;align-items:center;gap:6px}.detected-format{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase}.detected-format strong{color:var(--text)}.detected-format small{flex-basis:100%;font:inherit;text-transform:none}.inspection-summary,.sample-count-banner{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-bottom:14px}.inspection-summary>div,.sample-count-banner>div{display:grid;gap:5px;padding:11px;border:1px solid var(--border);border-radius:5px;background:#0c141b}.inspection-summary span,.sample-count-banner span,.source-summary-list span,.final-summary small{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase}.inspection-summary strong,.sample-count-banner strong{font-size:16px}.source-summary-list,.confirm-summary{display:grid;gap:7px;margin-bottom:14px}.source-summary-list>div{display:grid;grid-template-columns:120px minmax(0,1fr);gap:6px;padding:9px;border-bottom:1px solid var(--border);font-size:9px}.source-summary-list strong{font-size:10px}.source-summary-list code{grid-column:2;overflow-wrap:anywhere;color:var(--muted);font-size:9px}.confirm-summary>div,.relationship-card{display:grid;gap:6px;padding:12px;border:1px solid var(--border);border-radius:5px;background:#0c141b}.confirm-summary strong{font-size:12px}.relationship-card{margin-top:12px}.relationship-card p{margin:0;color:var(--muted);font-size:9px}.wizard-error{margin:0;padding:9px;border:1px solid rgba(239,106,114,.45);border-radius:5px;background:rgba(239,106,114,.08);color:var(--error);font-size:10px}.wizard-empty,.wizard-success{display:grid;justify-items:center;gap:10px;padding:34px 12px;text-align:center}.wizard-empty h3,.wizard-success h3{margin:0}.success-mark{display:grid;place-items:center;width:48px;height:48px;border:1px solid rgba(99,202,143,.5);border-radius:50%;color:var(--success,#63ca8f)}.final-summary{display:grid;gap:6px;width:min(480px,100%);padding:13px;text-align:left;border:1px solid var(--border);background:#0c141b;border-radius:5px}.final-summary span{font-size:12px}.final-summary strong{font-size:10px}.final-summary small{line-height:1.4;text-transform:none}.validation-result{display:grid;gap:6px;margin-top:12px;padding:11px;border:1px solid var(--border);border-radius:5px;background:rgba(239,106,114,.06);font-size:9px}.validation-result.valid{border-color:rgba(99,202,143,.45);background:rgba(99,202,143,.06)}.validation-result p{margin:0;color:var(--muted);line-height:1.4}.validation-result-title{display:flex;align-items:center;gap:7px}.success-icon{color:var(--success,#63ca8f)}.error-icon{color:var(--error)}.warning-line{color:var(--warning)!important}.wizard-actions{padding:13px 20px;border-top:1px solid var(--border)}.wizard-actions .primary{display:inline-flex;align-items:center;gap:7px}@media(max-width:680px){.dataset-wizard{width:calc(100vw - 16px);max-height:calc(100vh - 16px)}.wizard-header,.wizard-body,.wizard-actions{padding-left:14px;padding-right:14px}.dataset-type-grid,.option-grid{grid-template-columns:1fr}.inspection-summary{grid-template-columns:1fr}.source-summary-list>div{grid-template-columns:1fr}.source-summary-list code{grid-column:1}.wizard-actions{gap:7px}.wizard-actions button{flex:1}}
  `],
})
export class DatasetWizardComponent implements OnChanges {
  @Input() open = false
  @Output() readonly closed = new EventEmitter<void>()
  @Output() readonly saved = new EventEmitter<DatasetDocument>()

  private readonly api = inject(ApiService)
  readonly steps = ['Dataset and sources', 'Inspect and confirm', 'Register']
  readonly datasetTypes = DATASET_CAPABILITIES
  readonly step = signal(0)
  readonly datasetType = signal<DatasetType>('tabular')
  readonly composedMode = signal<'images' | 'text'>('images')
  readonly textTableMode = signal(false)
  readonly name = signal('')
  readonly sources = signal<DatasetSource[]>([])
  readonly inspection = signal<DatasetInspection | null>(null)
  readonly validation = signal<DatasetValidationResult | null>(null)
  readonly roleAssignments = signal<Record<string, FieldRole>>({})
  readonly relationshipKey = signal<FileTableRelationship['file_key']>('filename')
  readonly relationshipColumn = signal('')
  readonly timestampColumn = signal('')
  readonly groupColumn = signal('')
  readonly browserOpen = signal(false)
  readonly browserTarget = signal(0)
  readonly busy = signal(false)
  readonly error = signal<string | null>(null)

  readonly fields = computed(() => buildDatasetFields(this.inspection(), { structuralKeys: this.structuralKeys() }))

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['open']?.currentValue === true && changes['open']?.previousValue !== true) this.reset()
  }

  selectType(type: string): void {
    if (!DATASET_CAPABILITIES.some((item) => item.id === type && item.enabled) || type === 'auto_detect') return
    const next = type as DatasetType
    if (next === this.datasetType() && this.sources().length) return
    this.datasetType.set(next)
    this.textTableMode.set(false)
    this.composedMode.set('images')
    this.resetSources(next)
    this.clearInspection()
  }

  setTextTableMode(enabled: boolean): void {
    if (this.datasetType() !== 'text' || this.textTableMode() === enabled) return
    this.textTableMode.set(enabled)
    this.resetSources('text')
    this.clearInspection()
  }

  setComposedMode(mode: 'images' | 'text'): void {
    if (this.datasetType() !== 'composed' || this.composedMode() === mode) return
    this.composedMode.set(mode)
    this.resetSources('composed')
    this.clearInspection()
  }

  setName(value: string): void { this.name.set(value); this.clearValidation() }

  async inspectAndContinue(): Promise<void> {
    if (this.busy() || !this.sourcesReady()) return
    this.busy.set(true)
    this.error.set(null)
    try {
      const result = await this.api.inspectDataset({ dataset_type: this.datasetType(), sources: this.sources() })
      this.inspection.set(result)
      this.sources.update((items) => items.map((source) => {
        const inspected = result.sources.find((item) => item.source_id === source.id)
        return inspected ? { ...source, path: inspected.path } : source
      }))
      this.initializeFields()
      this.goTo(1)
    } catch (reason) {
      this.error.set(reason instanceof Error ? reason.message : String(reason))
    } finally { this.busy.set(false) }
  }

  async confirmDefinition(): Promise<void> {
    if (this.busy() || !this.canConfirm()) return
    this.busy.set(true)
    this.error.set(null)
    try {
      const result = await this.api.validateDataset(this.buildDefinition())
      this.validation.set(result)
      if (result.valid) this.goTo(2)
    } catch (reason) {
      this.error.set(reason instanceof Error ? reason.message : String(reason))
    } finally { this.busy.set(false) }
  }

  async register(): Promise<void> {
    if (this.busy() || !this.validation()?.valid) return
    this.busy.set(true)
    this.error.set(null)
    try {
      const document = await this.api.createDataset(this.buildDefinition())
      this.saved.emit(document)
      this.close()
    } catch (reason) {
      const validation = extractDatasetValidation(reason)
      if (validation) { this.validation.set(validation); this.error.set(null); this.goTo(1) }
      else this.error.set(reason instanceof Error ? reason.message : String(reason))
    } finally { this.busy.set(false) }
  }

  goTo(value: number): void { this.step.set(Math.max(0, Math.min(this.steps.length - 1, value))) }
  canConfirm(): boolean { return Boolean(this.inspection()) && !this.roleError() }
  canRegister(): boolean { return Boolean(this.validation()?.valid) }
  sourcesReady(): boolean { return Boolean(this.name().trim()) && this.sources().length > 0 && this.sources().every((source) => Boolean(source.path.trim()) && this.sourceFormatReady(source)) }

  openBrowser(index: number): void { this.browserTarget.set(index); this.browserOpen.set(true) }
  browserTitle(): string { return `Select ${this.sourceTitle(this.sources()[this.browserTarget()] ?? this.sources()[0])}` }
  browserMode(): 'file' | 'directory' {
    const source = this.sources()[this.browserTarget()]
    if (!source) return 'file'
    return sourceMode(this.datasetType(), this.browserTarget(), { textTable: this.textTableMode(), textPathKind: source.type === 'text' ? source.path_kind : undefined, composedMode: this.composedMode() })
  }
  browserExtensions(): string[] {
    return sourceExtensions(this.datasetType(), this.browserTarget(), { textTable: this.textTableMode(), composedMode: this.composedMode() })
  }

  acceptPath(path: string): void {
    const index = this.browserTarget()
    this.sources.update((items) => items.map((source, itemIndex) => {
      if (itemIndex !== index) return source
      if (source.type === 'table') {
        const format = tableFormatFromPath(path)
        return format ? { ...source, path, format, delimiter: format === 'tsv' ? '\t' : format === 'parquet' ? null : ',' } : { ...source, path }
      }
      if (source.type === 'array') return { ...source, path, format: arrayFormatFromPath(path) ?? source.format }
      return { ...source, path }
    }))
    if (!this.name().trim() && index === 0) this.name.set(path.split(/[\\/]/).filter(Boolean).at(-1) ?? path)
    this.browserOpen.set(false)
    this.clearInspection()
  }

  sourceTitle(source: DatasetSource | undefined): string {
    if (!source) return 'source'
    if (source.type === 'table') return `${source.format.toUpperCase()} table`
    if (source.type === 'image_folder') return 'Image folder'
    if (source.type === 'text') return source.path_kind === 'folder' ? 'Text folder' : 'Text file'
    return `${source.format.toUpperCase()} array`
  }
  sourcePathLabel(source: DatasetSource): string { return source.type === 'image_folder' || (source.type === 'text' && source.path_kind === 'folder') ? 'Source folder' : 'Source file' }
  defaultDelimiter(format: TableSource['format']): TableSource['delimiter'] { return format === 'tsv' ? '\t' : ',' }
  updateTable(index: number, change: Partial<TableSource>): void { this.updateSource(index, change); this.clearInspection() }
  updateImage(index: number, change: Partial<ImageFolderSource>): void { this.updateSource(index, change); this.clearInspection() }
  updateText(index: number, change: Partial<TextSource>): void { this.updateSource(index, change); this.clearInspection() }
  updateArray(index: number, change: Partial<ArraySource>): void { this.updateSource(index, change); this.clearInspection() }
  addArraySource(): void { this.sources.update((items) => [...items, this.arraySource(`array-${items.length + 1}`)]); this.clearInspection() }
  removeSource(index: number): void { this.sources.update((items) => items.filter((_, itemIndex) => itemIndex !== index)); this.clearInspection() }
  canRemoveSource(): boolean { return this.datasetType() === 'array' && this.sources().length > 1 }
  hasLabelTable(): boolean { return this.datasetType() === 'images' && this.sources().some((source) => source.type === 'table') }
  toggleImageTable(): void { if (this.hasLabelTable()) this.sources.update((items) => items.filter((source) => source.type !== 'table')); else this.sources.update((items) => [...items, this.tableSource('labels')]); this.clearInspection() }

  fieldCount(result: DatasetInspection): number { return result.sources.reduce((total, source) => total + ('columns' in source ? source.columns.length : 'arrays' in source ? source.arrays.length : source.sample_count > 0 ? 1 : 0), 0) }
  datasetTypeLabel(): string { return datasetCapability(this.datasetType()).label }
  inspectedSourceLabel(source: DatasetSourceInspection): string { return source.type === 'table' ? `${source.format.toUpperCase()} table · ${source.columns.length} columns` : source.type === 'array' ? `${source.format.toUpperCase()} · ${source.arrays.length} keys` : source.type === 'image_folder' ? `Image folder · ${source.sample_count} samples` : `Text source · ${source.sample_count} samples` }
  roleSummary(role: FieldRole): string { return bindingSummary(bindingsFor(this.fields(), this.sources(), this.roleAssignments(), role)) || (role === 'output' ? 'None (input only)' : 'None') }
  role(key: string): FieldRole { return this.roleAssignments()[key] ?? 'unused' }
  setRelationshipKey(value: string): void { this.relationshipKey.set(value as FileTableRelationship['file_key']); this.clearValidation() }
  setRelationshipColumn(value: string): void { this.relationshipColumn.set(value); this.reserveStructuralColumn(value); this.clearValidation() }
  setTimestampColumn(value: string): void { this.timestampColumn.set(value); this.reserveStructuralColumn(value); this.clearValidation() }
  setGroupColumn(value: string): void { this.groupColumn.set(value); this.reserveStructuralColumn(value); this.clearValidation() }
  tableColumns(): string[] { const table = this.inspection()?.sources.find((source): source is Extract<DatasetSourceInspection, { type: 'table' }> => source.type === 'table'); return table?.columns.map((column) => column.name) ?? [] }
  hasRelationship(): boolean { return this.datasetType() === 'composed' || (this.datasetType() === 'images' && this.hasLabelTable()) }
  roleError(): string | null { if (!this.inspection()) return null; if (!this.fields().some((field) => this.role(field.key) === 'input')) return 'The initial definition must contain at least one input.'; if (this.hasRelationship() && !this.relationshipColumn()) return 'Choose the table identifier column for the source relationship.'; return null }
  clearValidation(): void { this.validation.set(null) }
  close(): void { if (!this.busy()) this.closed.emit() }

  private reset(): void { this.step.set(0); this.datasetType.set('tabular'); this.composedMode.set('images'); this.textTableMode.set(false); this.name.set(''); this.resetSources('tabular'); this.clearInspection(); this.relationshipKey.set('filename'); this.relationshipColumn.set(''); this.timestampColumn.set(''); this.groupColumn.set('') }
  private clearInspection(): void { this.inspection.set(null); this.validation.set(null); this.roleAssignments.set({}); this.error.set(null) }
  private resetSources(type: DatasetType): void { if (type === 'tabular' || type === 'time_series') this.sources.set([this.tableSource('table')]); else if (type === 'images') this.sources.set([this.imageSource('images')]); else if (type === 'text') this.sources.set([this.textTableMode() ? this.tableSource('text-table') : this.textSource('text')]); else if (type === 'array') this.sources.set([this.arraySource('array-1')]); else this.sources.set([this.composedMode() === 'text' ? this.textSource('modality') : this.imageSource('modality'), this.tableSource('table')]) }
  private tableSource(id: string): TableSource { return { type: 'table', id, path: '', format: 'csv', has_header: true, delimiter: ',' } }
  private imageSource(id: string): ImageFolderSource { return { type: 'image_folder', id, path: '', recursive: true, label_mode: 'none' } }
  private textSource(id: string): TextSource { return { type: 'text', id, path: '', path_kind: 'folder', recursive: true } }
  private arraySource(id: string): ArraySource { return { type: 'array', id, path: '', format: 'npy' } }
  private updateSource<T extends DatasetSource>(index: number, change: Partial<T>): void { this.sources.update((items) => items.map((source, itemIndex) => itemIndex === index ? { ...source, ...change } as DatasetSource : source)) }
  private sourceFormatReady(source: DatasetSource): boolean { if (source.type === 'table') return tableFormatFromPath(source.path) !== null; if (source.type === 'array') return arrayFormatFromPath(source.path) !== null; return true }
  private initializeFields(): void { if (this.hasRelationship() && !this.relationshipColumn()) this.relationshipColumn.set(this.tableColumns()[0] ?? ''); this.roleAssignments.set(Object.fromEntries(this.fields().map((field) => [field.key, defaultFieldRole(field, this.datasetType())]))) }
  private structuralKeys(): Set<string> { const keys = new Set<string>(); const tableId = this.sources().find((source) => source.type === 'table')?.id ?? ''; for (const column of [this.relationshipColumn(), this.timestampColumn(), this.groupColumn()]) if (column) keys.add(`${tableId}::${column}`); return keys }
  private reserveStructuralColumn(column: string): void { if (!column) return; this.roleAssignments.update((current) => ({ ...current, [`${this.sources().find((source) => source.type === 'table')?.id ?? ''}::${column}`]: 'unused' })) }
  private buildDefinition(): DatasetDefinitionMutation { const fields = this.fields(); const assignments = this.roleAssignments(); const fileSource = this.sources().find((source) => source.type === 'image_folder' || source.type === 'text'); const tableSource = this.sources().find((source) => source.type === 'table'); const relationships = this.hasRelationship() && fileSource && tableSource && this.relationshipColumn() ? [{ type: 'file_table' as const, file_source_id: fileSource.id, file_key: this.relationshipKey(), table_source_id: tableSource.id, table_key_column: this.relationshipColumn() }] : []; return { name: this.name().trim(), dataset_type: this.datasetType(), sources: this.sources(), input_definition: { bindings: bindingsFor(fields, this.sources(), assignments, 'input') }, output_definition: { bindings: bindingsFor(fields, this.sources(), assignments, 'output') }, metadata_definition: { bindings: bindingsFor(fields, this.sources(), assignments, 'metadata') }, relationships, sequence_definition: this.datasetType() === 'time_series' ? { timestamp_column: this.timestampColumn() || null, group_column: this.groupColumn() || null } : null, subset_definition: { type: 'all' }, split_definition: null } }
}
