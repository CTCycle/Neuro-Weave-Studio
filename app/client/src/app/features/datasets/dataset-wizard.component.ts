import { CommonModule } from '@angular/common'
import { Component, EventEmitter, Input, Output, computed, inject, signal } from '@angular/core'
import type { OnChanges, SimpleChanges } from '@angular/core'
import { LucideAlertTriangle, LucideCheck, LucideFilePlus2, LucideInfo, LucidePlus, LucideX } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import { FilesystemBrowserComponent } from '../../shared/components/filesystem-browser.component'
import type {
  ArrayBinding,
  ArraySource,
  ColumnBinding,
  DatasetBinding,
  DatasetDefinitionMutation,
  DatasetDocument,
  DatasetInspection,
  DatasetSource,
  DatasetSourceInspection,
  DatasetSubsetDefinition,
  DatasetType,
  FileTableRelationship,
  FolderLabelBinding,
  ImageBinding,
  ImageFolderSource,
  TableSource,
  TextBinding,
  TextSource,
} from '../../domain/dataset.types'
import type { DatasetValidationResult } from '../../domain/dataset.types'
import { DATASET_CAPABILITIES, datasetCapability } from '../../domain/dataset-capabilities'
import { ModalGuardDirective } from '../../shared/components/modal-guard.directive'

type FieldRole = 'input' | 'output' | 'metadata' | 'unused'

interface DatasetField {
  key: string
  sourceId: string
  label: string
  kind: 'column' | 'array' | 'image' | 'text' | 'folder_label'
  dtype?: string
  structural: boolean
}

@Component({
  selector: 'nw-dataset-wizard',
  standalone: true,
  imports: [CommonModule, LucideAlertTriangle, LucideCheck, LucideFilePlus2, LucideInfo, LucidePlus, LucideX, FilesystemBrowserComponent, ModalGuardDirective],
  template: `
    @if (open) {
      <div class="dialog-backdrop" nwModalGuard (click)="close()">
        <section class="dataset-wizard panel" role="dialog" aria-modal="true" aria-labelledby="dataset-wizard-title" (click)="$event.stopPropagation()">
          <header class="wizard-header">
            <div><span class="eyebrow">{{ editing ? 'EDIT DATASET' : 'NEW DATASET' }} · STEP {{ step() + 1 }} OF {{ steps.length }}</span><h2 id="dataset-wizard-title">{{ steps[step()] }}</h2></div>
            <button class="icon-button" type="button" aria-label="Close dataset wizard" (click)="close()"><svg lucideX [size]="16"></svg></button>
          </header>
          <div class="wizard-progress" aria-hidden="true">@for (label of steps; track label; let index = $index) { <span [class.active]="index <= step()"></span> }</div>

          <div class="wizard-body">
            @if (step() === 0) {
              <div class="wizard-intro"><span class="eyebrow">DATA ORGANIZATION</span><p>Choose how the external source represents samples. Automatic detection is planned but does not run yet.</p></div>
              <div class="dataset-type-grid">
                @for (type of datasetTypes; track type.id) {
                  <button type="button" class="dataset-type-option" [class.selected]="type.id === datasetType()" [class.disabled]="!type.enabled" [disabled]="!type.enabled" (click)="selectType(type.id)">
                    <span class="dataset-type-icon">{{ type.enabled ? '◇' : '…' }}</span><strong>{{ type.label }}</strong><span>{{ type.description }}</span><small>{{ type.sourceLabel }}</small>
                  </button>
                }
              </div>
            }

            @if (step() === 1) {
              <div class="wizard-fields">
                <label><span>Dataset name</span><input [value]="name()" placeholder="e.g. Patient outcomes" (input)="setName($any($event.target).value)" /></label>
                @if (datasetType() === 'composed') { <div class="inline-choice"><span>Primary file modality</span><button type="button" class="choice-button" [class.active]="composedMode() === 'images'" (click)="setComposedMode('images')">Images + table</button><button type="button" class="choice-button" [class.active]="composedMode() === 'text'" (click)="setComposedMode('text')">Text + table</button></div> }
                <div class="source-stack">
                  @for (source of sources(); track source.id; let index = $index) {
                    <div class="source-card">
                      <div class="source-card-header"><div><span class="eyebrow">SOURCE {{ index + 1 }}</span><strong>{{ sourceTitle(source) }}</strong></div>@if (canRemoveSource()) { <button class="icon-button" type="button" [attr.aria-label]="'Remove ' + sourceTitle(source)" (click)="removeSource(index)"><svg lucideX [size]="14"></svg></button> }</div>
                      <label class="path-field"><span>{{ sourcePathLabel(source) }}</span><div><input readonly [value]="source.path" placeholder="No path selected" /><button class="secondary" type="button" (click)="openBrowser(index)">Browse…</button></div></label>
                      @if (source.type === 'table') {
                        <div class="option-grid"><label><span>Format</span><select [value]="source.format" (change)="updateTableFormat(index, $any($event.target).value)"><option value="csv">CSV</option><option value="tsv">TSV</option><option value="parquet">Parquet</option></select></label><label><span>Header row</span><select [value]="source.has_header ? 'yes' : 'no'" (change)="updateTable(index, { has_header: $any($event.target).value === 'yes' })"><option value="yes">First row contains column names</option><option value="no">No header row</option></select></label><label><span>Delimiter</span><select [value]="source.delimiter ?? defaultDelimiter(source.format)" [disabled]="source.format === 'parquet'" (change)="updateTable(index, { delimiter: $any($event.target).value })"><option value=",">Comma</option><option value="\t">Tab</option><option value=";">Semicolon</option><option value="|">Pipe</option></select></label></div>
                      }
                      @if (source.type === 'image_folder') {
                        <div class="option-grid"><label><span>Folder labels</span><select [value]="source.label_mode" (change)="updateImage(index, { label_mode: $any($event.target).value })"><option value="none">No folder labels</option><option value="parent_folder">Parent folder names</option></select></label><label class="check-field"><input type="checkbox" [checked]="source.recursive" (change)="updateImage(index, { recursive: $any($event.target).checked })" /><span>Scan subfolders recursively</span></label></div>
                      }
                      @if (source.type === 'text') {
                        <div class="option-grid"><label><span>Text source</span><select [value]="source.path_kind" (change)="updateText(index, { path_kind: $any($event.target).value })"><option value="file">One text file</option><option value="folder">Text folder</option></select></label><label class="check-field"><input type="checkbox" [checked]="source.recursive" (change)="updateText(index, { recursive: $any($event.target).checked })" /><span>Scan subfolders recursively</span></label></div>
                      }
                      @if (source.type === 'array') { <div class="option-grid"><label><span>Format</span><select [value]="source.format" (change)="updateArrayFormat(index, $any($event.target).value)"><option value="npy">NPY</option><option value="npz">NPZ</option></select></label><p class="field-hint">Headers are inspected without loading array contents.</p></div> }
                    </div>
                  }
                </div>
                @if (datasetType() === 'text') { <div class="inline-choice"><span>Interpret this text dataset as</span><button type="button" class="choice-button" [class.active]="textTableMode()" (click)="setTextTableMode(true)">A table</button><button type="button" class="choice-button" [class.active]="!textTableMode()" (click)="setTextTableMode(false)">Text files</button></div> }
                @if (datasetType() === 'images') { <button class="secondary add-source" type="button" (click)="toggleImageTable()"><svg lucidePlus [size]="14"></svg>{{ hasLabelTable() ? 'Remove label table' : 'Add label / metadata table' }}</button> }
                @if (datasetType() === 'array' && sources().length < 3) { <button class="secondary add-source" type="button" (click)="addArraySource()"><svg lucidePlus [size]="14"></svg>Add array source</button> }
                @if (error()) { <p class="wizard-error" role="alert">{{ error() }}</p> }
                @if (inspection(); as result) { <div class="inspection-summary"><div><span>AVAILABLE SAMPLES</span><strong>{{ result.sample_count | number }}</strong></div><div><span>SOURCES INSPECTED</span><strong>{{ result.sources.length }}</strong></div><div><span>SCHEMA</span><strong>{{ fieldCount(result) }} fields</strong></div></div> }
                <button class="secondary inspect-button" type="button" [disabled]="busy() || !sourcesReady()" (click)="inspectSources()"><svg lucideFilePlus2 [size]="14"></svg>{{ busy() ? 'Inspecting source…' : 'Inspect sources' }}</button>
              </div>
            }

            @if (step() === 2) {
              @if (!inspection()) { <section class="wizard-empty"><span class="eyebrow">INSPECTION REQUIRED</span><h3>Inspect the sources first</h3><p>NeuroWeave needs the lightweight schema before input and output roles can be assigned.</p><button class="secondary" type="button" (click)="goTo(1)">Back to sources</button></section> }
              @else {
                <div class="wizard-intro"><span class="eyebrow">MODEL BOUNDARY</span><p>Assign every available field explicitly. Input and output roles are never inferred from names or position; unused fields remain available for later editing.</p></div>
                <div class="role-list">
                  @for (field of fields(); track field.key) {
                    <div class="role-row" [class.structural]="field.structural"><div class="role-field"><strong>{{ field.label }}</strong><span>{{ sourceLabel(field.sourceId) }}@if (field.dtype) { · {{ field.dtype }} }@if (field.structural) { · structural key }</span></div><select [value]="role(field.key)" [disabled]="field.structural" (change)="setRole(field.key, $any($event.target).value)"><option value="input">Input</option><option value="output">Output / target</option><option value="metadata">Metadata</option><option value="unused">Unused</option></select></div>
                  }
                </div>
                @if (hasRelationship()) {
                  <div class="relationship-card"><span class="eyebrow">SOURCE RELATIONSHIP</span><p>Choose the table identifier that resolves each file to exactly one row.</p><div class="option-grid"><label><span>File identifier</span><select [value]="relationshipKey()" (change)="setRelationshipKey($any($event.target).value)"><option value="filename">Full filename</option><option value="stem">Filename stem</option><option value="relative_path">Relative path</option></select></label><label><span>Table identifier column</span><select [value]="relationshipColumn()" (change)="setRelationshipColumn($any($event.target).value)">@for (column of tableColumns(); track column) { <option [value]="column">{{ column }}</option> }</select></label></div></div>
                }
                @if (datasetType() === 'time_series') { <div class="relationship-card"><span class="eyebrow">SEQUENCE STRUCTURE</span><div class="option-grid"><label><span>Timestamp column (optional)</span><select [value]="timestampColumn()" (change)="setTimestampColumn($any($event.target).value)"><option value="">None</option>@for (column of tableColumns(); track column) { <option [value]="column">{{ column }}</option> }</select></label><label><span>Group / sequence column (optional)</span><select [value]="groupColumn()" (change)="setGroupColumn($any($event.target).value)"><option value="">None</option>@for (column of tableColumns(); track column) { <option [value]="column">{{ column }}</option> }</select></label></div></div> }
                @if (roleError()) { <p class="wizard-error" role="alert">{{ roleError() }}</p> }
              }
            }

            @if (step() === 3) {
              <div class="wizard-intro"><span class="eyebrow">SAMPLE SELECTION</span><p>Store only the selection rule. NeuroWeave never copies selected rows, files, images, or arrays into the application database.</p></div>
              <div class="sample-count-banner"><div><span>AVAILABLE SAMPLES</span><strong>{{ availableSamples() | number }}</strong></div><div><span>SELECTED SAMPLES</span><strong>{{ selectedSamples() | number }}</strong></div></div>
              <div class="subset-options"><label class="subset-option" [class.active]="subsetMode() === 'all'"><input type="radio" name="subset" value="all" [checked]="subsetMode() === 'all'" (change)="subsetMode.set('all'); clearValidation()" /><span><strong>Use all samples</strong><small>Keep the complete inspected source.</small></span></label><label class="subset-option" [class.active]="subsetMode() === 'first_n'"><input type="radio" name="subset" value="first_n" [checked]="subsetMode() === 'first_n'" (change)="subsetMode.set('first_n'); clearValidation()" /><span><strong>First N samples</strong><small>Keep the first records in source order.</small></span></label>@if (subsetMode() === 'first_n') { <label class="inline-number"><span>Count</span><input type="number" min="1" [max]="availableSamples()" [value]="firstN()" (input)="firstN.set(numberValue($any($event.target).value)); clearValidation()" /></label> }<label class="subset-option" [class.active]="subsetMode() === 'range'"><input type="radio" name="subset" value="range" [checked]="subsetMode() === 'range'" (change)="subsetMode.set('range'); clearValidation()" /><span><strong>Inclusive range</strong><small>Use zero-based start and end positions.</small></span></label>@if (subsetMode() === 'range') { <div class="range-fields"><label><span>Start</span><input type="number" min="0" [value]="rangeStart()" (input)="rangeStart.set(numberValue($any($event.target).value)); clearValidation()" /></label><label><span>End</span><input type="number" min="0" [max]="availableSamples() - 1" [value]="rangeEnd()" (input)="rangeEnd.set(numberValue($any($event.target).value)); clearValidation()" /></label></div> }</div>
              @if (!subsetValid()) { <p class="wizard-error" role="alert">{{ subsetError() }}</p> }
            }

            @if (step() === 4) {
              <div class="wizard-intro"><span class="eyebrow">REVIEW & VALIDATION</span><p>Review the lightweight definition before registration. The server will re-inspect the external sources before saving.</p></div>
              <div class="review-grid"><div><span>Name</span><strong>{{ name() }}</strong></div><div><span>Type</span><strong>{{ datasetTypeLabel() }}</strong></div><div><span>Sources</span><strong>{{ sources().length }} · {{ primaryPath() }}</strong></div><div><span>Samples</span><strong>{{ selectedSamples() | number }} selected of {{ availableSamples() | number }}</strong></div><div><span>Inputs</span><strong>{{ roleSummary('input') }}</strong></div><div><span>Outputs</span><strong>{{ roleSummary('output') }}</strong></div></div>
              <div class="copy-boundary"><svg lucideInfo [size]="15"></svg><span>Only dataset metadata, schema information, paths, role definitions, and the subset rule will be stored. Underlying source content stays in place.</span></div>
              @if (validation(); as result) { <div class="validation-result" [class.valid]="result.valid" [class.invalid]="!result.valid"><div class="validation-result-title">@if (result.valid) { <svg lucideCheck class="success-icon" [size]="16"></svg> } @else { <svg lucideAlertTriangle class="error-icon" [size]="16"></svg> }<strong>{{ result.valid ? 'Definition is valid' : 'Definition needs attention' }}</strong></div>@for (finding of result.validation.errors; track finding.code + finding.message) { <p><strong>{{ finding.code }}</strong> · {{ finding.message }}</p> }@for (finding of result.validation.warnings; track finding.code + finding.message) { <p class="warning-line"><strong>Warning</strong> · {{ finding.message }}</p> }</div> }
              @if (error()) { <p class="wizard-error" role="alert">{{ error() }}</p> }
            }

            @if (step() === 5) {
              <section class="wizard-success"><span class="eyebrow">READY TO REGISTER</span><div class="success-mark"><svg lucideCheck [size]="22"></svg></div><h3>{{ editing ? 'Update this dataset definition' : 'Register this dataset definition' }}</h3><p>The external data remains where it is. Registration creates a lightweight metadata record that can be reloaded after restarting NeuroWeave.</p><div class="final-summary"><span>{{ name() }}</span><strong>{{ selectedSamples() | number }} selected samples · {{ datasetTypeLabel() }}</strong></div></section>
            }
          </div>

          <footer class="dialog-actions wizard-actions"><button class="secondary" type="button" (click)="step() === 0 ? close() : goTo(step() - 1)" [disabled]="busy()">{{ step() === 0 ? 'Cancel' : 'Back' }}</button>@if (step() < 4) { <button class="primary" type="button" [disabled]="busy() || !canContinue()" (click)="next()">{{ busy() ? 'Working…' : step() === 1 ? 'Inspect & continue' : 'Continue' }}</button> } @else if (step() === 4) { <button class="primary" type="button" [disabled]="busy() || !canValidate()" (click)="validateDefinition()">{{ busy() ? 'Validating…' : validation()?.valid ? 'Continue to registration' : 'Validate definition' }}</button> } @else { <button class="primary" type="button" [disabled]="busy() || !validation()?.valid" (click)="register()"><svg lucideFilePlus2 [size]="15"></svg>{{ busy() ? 'Registering…' : editing ? 'Save dataset changes' : 'Register dataset' }}</button> }</footer>
        </section>
      </div>
      <nw-filesystem-browser [open]="browserOpen()" [title]="browserTitle()" [selectionMode]="browserMode()" [extensions]="browserExtensions()" (closed)="browserOpen.set(false)" (selected)="acceptPath($event)" />
    }
  `,
  styles: [`
    :host{display:block}.dataset-wizard{width:min(900px,calc(100vw - 32px));max-height:calc(100vh - 32px);display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;overflow:hidden;box-shadow:0 24px 70px rgba(0,0,0,.55)}.wizard-header{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;padding:18px 20px;border-bottom:1px solid var(--border)}.wizard-header h2{margin:5px 0 0}.wizard-progress{display:grid;grid-template-columns:repeat(6,1fr);gap:4px;padding:8px 20px}.wizard-progress span{height:2px;background:var(--border)}.wizard-progress span.active{background:var(--cyan)}.wizard-body{min-height:0;overflow:auto;padding:18px 20px 22px;scrollbar-gutter:stable}.wizard-intro{display:grid;gap:6px;margin-bottom:16px}.wizard-intro p,.field-hint,.wizard-empty p,.wizard-success p{margin:0;color:var(--muted);font-size:10px;line-height:1.5}.dataset-type-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.dataset-type-option{min-width:0;display:grid;grid-template-columns:28px minmax(0,1fr);gap:4px 9px;padding:13px;border:1px solid var(--border);border-radius:6px;background:#0c141b;color:var(--text);text-align:left;cursor:pointer}.dataset-type-option:hover,.dataset-type-option.selected{border-color:var(--cyan);background:rgba(101,201,212,.06)}.dataset-type-option.disabled{opacity:.55;cursor:not-allowed}.dataset-type-option strong{align-self:end;font-size:11px}.dataset-type-option span:not(.dataset-type-icon){grid-column:2;color:var(--muted);font-size:9px;line-height:1.4}.dataset-type-option small{grid-column:2;color:var(--accent-bright);font:500 8px 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase}.dataset-type-icon{grid-row:span 3;display:grid;place-items:center;color:var(--cyan);font-size:22px}.wizard-fields{display:grid;gap:13px}.wizard-fields label,.role-row label,.relationship-card label,.subset-options label,.inline-number,.range-fields label{display:grid;gap:5px;color:var(--muted);font-size:9px}.wizard-fields input,.wizard-fields select,.role-row select,.relationship-card select,.subset-options input[type=number],.inline-number input,.range-fields input{width:100%;min-height:36px;border:1px solid var(--border);border-radius:6px;background:#0c141b;color:var(--text);padding:8px 10px}.path-field>div{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px}.path-field input{font:500 9px 'IBM Plex Mono',Consolas,monospace}.source-stack{display:grid;gap:10px}.source-card,.relationship-card{display:grid;gap:11px;padding:13px;border:1px solid var(--border);border-radius:6px;background:rgba(12,20,27,.7)}.source-card-header{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}.source-card-header strong{display:block;margin-top:4px;font-size:11px}.option-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.check-field{display:flex!important;align-items:center;gap:8px;padding-top:22px}.check-field input{width:16px!important;min-height:16px!important}.add-source{justify-self:start}.inline-choice{display:flex;align-items:center;gap:7px;flex-wrap:wrap;color:var(--muted);font-size:9px}.choice-button{min-height:30px;padding:0 10px;border:1px solid var(--border);border-radius:5px;background:transparent;color:var(--muted);font-size:9px}.choice-button.active{border-color:var(--cyan);color:var(--text);background:rgba(101,201,212,.08)}.inspection-summary,.sample-count-banner{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.inspection-summary>div,.sample-count-banner>div{display:grid;gap:4px;padding:10px;border:1px solid var(--border);background:#0c141b;border-radius:5px}.inspection-summary span,.sample-count-banner span,.review-grid span,.final-summary span{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace;letter-spacing:.05em}.inspection-summary strong,.sample-count-banner strong{font-size:16px}.inspect-button{justify-self:start}.role-list{display:grid;gap:4px}.role-row{display:grid;grid-template-columns:minmax(0,1fr) 180px;align-items:center;gap:10px;padding:9px 10px;border:1px solid var(--border);border-radius:5px;background:#0c141b}.role-row.structural{opacity:.7}.role-field{min-width:0;display:grid;gap:3px}.role-field strong{overflow-wrap:anywhere;font-size:10px}.role-field span{color:var(--muted);font-size:8px}.relationship-card p{margin:0;color:var(--muted);font-size:9px;line-height:1.4}.sample-count-banner{grid-template-columns:repeat(2,minmax(0,1fr));margin-bottom:13px}.subset-options{display:grid;gap:8px}.subset-option{display:flex!important;grid-template-columns:none!important;align-items:flex-start;gap:9px;padding:10px;border:1px solid var(--border);border-radius:5px}.subset-option.active{border-color:var(--cyan);background:rgba(101,201,212,.06)}.subset-option input[type=radio]{width:15px;min-height:15px;margin-top:1px}.subset-option span{display:grid;gap:3px}.subset-option strong{color:var(--text);font-size:10px}.subset-option small{font-size:9px}.inline-number{max-width:180px;margin-left:24px}.range-fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-left:24px}.review-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.review-grid>div{display:grid;gap:4px;padding:10px;border-bottom:1px solid var(--border)}.review-grid strong{font-size:10px;overflow-wrap:anywhere}.copy-boundary{display:flex;gap:8px;margin-top:14px;padding:11px;border:1px solid rgba(101,201,212,.3);border-radius:5px;color:var(--muted);font-size:9px;line-height:1.45}.copy-boundary svg{flex:0 0 auto;color:var(--cyan)}.validation-result{display:grid;gap:6px;margin-top:12px;padding:12px;border:1px solid rgba(239,106,114,.4);border-radius:5px;background:rgba(239,106,114,.06)}.validation-result.valid{border-color:rgba(99,202,143,.45);background:rgba(99,202,143,.06)}.validation-result p{margin:0;color:var(--muted);font-size:9px;line-height:1.4}.validation-result-title{display:flex;align-items:center;gap:7px}.success-icon{color:var(--success,#63ca8f)}.error-icon{color:var(--error)}.warning-line{color:var(--warning)!important}.wizard-empty,.wizard-success{display:grid;justify-items:center;gap:9px;padding:42px 24px;text-align:center}.wizard-empty h3,.wizard-success h3{margin:0}.success-mark{display:grid;place-items:center;width:48px;height:48px;border:1px solid rgba(99,202,143,.5);border-radius:50%;color:var(--success,#63ca8f)}.final-summary{display:grid;gap:5px;width:min(420px,100%);padding:12px;border:1px solid var(--border);border-radius:5px;background:#0c141b}.final-summary strong{font-size:10px}.wizard-error{margin:0;color:var(--error);font-size:9px;line-height:1.4}.wizard-actions{padding:12px 20px;border-top:1px solid var(--border);justify-content:flex-end}.wizard-actions button{display:inline-flex;align-items:center;gap:6px}
    @media(max-width:760px){.dataset-type-grid,.option-grid,.review-grid{grid-template-columns:1fr}.role-row{grid-template-columns:1fr}.check-field{padding-top:0}.dataset-wizard{width:calc(100vw - 16px);max-height:calc(100vh - 16px)}.wizard-body{padding-inline:14px}.wizard-header,.wizard-actions{padding-inline:14px}.wizard-progress{padding-inline:14px}.inline-number,.range-fields{margin-left:0}}
  `],
})
export class DatasetWizardComponent implements OnChanges {
  @Input() open = false
  @Input() editing: DatasetDocument | null = null
  @Output() readonly closed = new EventEmitter<void>()
  @Output() readonly saved = new EventEmitter<DatasetDocument>()

  private readonly api = inject(ApiService)
  readonly steps = ['Choose type', 'Data sources', 'Input / output', 'Samples', 'Review', 'Register']
  readonly datasetTypes = DATASET_CAPABILITIES
  readonly step = signal(0)
  readonly datasetType = signal<DatasetType>('tabular')
  readonly composedMode = signal<'images' | 'text'>('images')
  readonly name = signal('')
  readonly sources = signal<DatasetSource[]>([])
  readonly inspection = signal<DatasetInspection | null>(null)
  readonly validation = signal<DatasetValidationResult | null>(null)
  readonly roleAssignments = signal<Record<string, FieldRole>>({})
  readonly relationshipKey = signal<FileTableRelationship['file_key']>('filename')
  readonly relationshipColumn = signal('')
  readonly timestampColumn = signal('')
  readonly groupColumn = signal('')
  readonly subsetMode = signal<DatasetSubsetDefinition['type']>('all')
  readonly firstN = signal(1)
  readonly rangeStart = signal(0)
  readonly rangeEnd = signal(0)
  readonly textTableMode = signal(false)
  readonly browserOpen = signal(false)
  readonly browserTarget = signal(0)
  readonly busy = signal(false)
  readonly error = signal<string | null>(null)

  readonly fields = computed(() => this.buildFields(this.inspection()))
  readonly availableSamples = computed(() => this.inspection()?.sample_count ?? 0)
  readonly selectedSamples = computed(() => this.computeSelectedSamples())

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['open']?.currentValue === true && changes['open']?.previousValue !== true) {
      if (this.editing) this.hydrate(this.editing)
      else this.reset()
    }
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
    this.textTableMode.set(enabled)
    this.resetSources('text')
    if (enabled) this.sources.set([this.tableSource('text-table')])
    this.clearInspection()
  }

  setComposedMode(mode: 'images' | 'text'): void {
    if (this.datasetType() !== 'composed' || this.composedMode() === mode) return
    this.composedMode.set(mode)
    this.resetSources('composed')
    this.clearInspection()
  }

  setName(value: string): void { this.name.set(value); this.clearValidation() }

  next(): void {
    if (this.step() === 1) {
      void this.inspectSources().then((success) => { if (success) this.goTo(2) })
      return
    }
    if (this.canContinue()) this.goTo(this.step() + 1)
  }

  goTo(value: number): void { this.step.set(Math.max(0, Math.min(this.steps.length - 1, value))) }

  canContinue(): boolean {
    if (this.step() === 0) return true
    if (this.step() === 1) return this.sourcesReady()
    if (this.step() === 2) return Boolean(this.inspection()) && Boolean(this.inputCount()) && (!this.hasRelationship() || Boolean(this.relationshipColumn()))
    if (this.step() === 3) return this.subsetValid()
    return true
  }

  canValidate(): boolean { return this.canContinue() && Boolean(this.inspection()) }

  async inspectSources(): Promise<boolean> {
    if (this.busy() || !this.sourcesReady()) return false
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
      return true
    } catch (reason) {
      this.error.set(reason instanceof Error ? reason.message : String(reason))
      return false
    } finally { this.busy.set(false) }
  }

  async validateDefinition(): Promise<void> {
    if (this.busy() || !this.canValidate()) return
    this.busy.set(true)
    this.error.set(null)
    try {
      const result = await this.api.validateDataset(this.buildDefinition())
      this.validation.set(result)
      if (result.valid) this.goTo(5)
    } catch (reason) {
      this.error.set(reason instanceof Error ? reason.message : String(reason))
    } finally { this.busy.set(false) }
  }

  async register(): Promise<void> {
    const result = this.validation()
    if (this.busy() || !result?.valid) return
    this.busy.set(true)
    this.error.set(null)
    try {
      const document = this.editing
        ? await this.api.updateDataset(this.editing.id, this.buildDefinition())
        : await this.api.createDataset(this.buildDefinition())
      this.saved.emit(document)
      this.close()
    } catch (reason) {
      this.error.set(reason instanceof Error ? reason.message : String(reason))
      this.goTo(4)
    } finally { this.busy.set(false) }
  }

  openBrowser(index: number): void { this.browserTarget.set(index); this.browserOpen.set(true) }

  browserTitle(): string { return `Select ${this.sourceTitle(this.sources()[this.browserTarget()] ?? this.sources()[0])}` }

  browserMode(): 'file' | 'directory' {
    const source = this.sources()[this.browserTarget()]
    if (!source) return 'file'
    return source.type === 'image_folder' || (source.type === 'text' && source.path_kind === 'folder') ? 'directory' : 'file'
  }

  browserExtensions(): string[] {
    const source = this.sources()[this.browserTarget()]
    if (!source || source.type === 'image_folder' || (source.type === 'text' && source.path_kind === 'folder')) return []
    if (source.type === 'table') return ['csv', 'tsv', 'parquet']
    if (source.type === 'array') return ['npy', 'npz']
    return ['txt', 'text', 'md', 'jsonl']
  }

  acceptPath(path: string): void {
    const index = this.browserTarget()
    this.sources.update((items) => items.map((source, itemIndex) => itemIndex === index ? { ...source, path } : source))
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

  sourceLabel(sourceId: string): string { return this.sourceTitle(this.sources().find((source) => source.id === sourceId)) }

  defaultDelimiter(format: TableSource['format']): TableSource['delimiter'] { return format === 'tsv' ? '\t' : ',' }

  updateTable(index: number, change: Partial<TableSource>): void { this.updateSource(index, change); this.clearInspection() }
  updateTableFormat(index: number, format: TableSource['format']): void { this.updateTable(index, { format, delimiter: format === 'tsv' ? '\t' : format === 'parquet' ? null : ',' }) }
  updateImage(index: number, change: Partial<ImageFolderSource>): void { this.updateSource(index, change); this.clearInspection() }
  updateText(index: number, change: Partial<TextSource>): void { this.updateSource(index, change); this.clearInspection() }
  updateArrayFormat(index: number, format: ArraySource['format']): void { this.updateSource(index, { format }); this.clearInspection() }

  addArraySource(): void { this.sources.update((items) => [...items, this.arraySource(`array-${items.length + 1}`)]); this.clearInspection() }

  removeSource(index: number): void { this.sources.update((items) => items.filter((_, itemIndex) => itemIndex !== index)); this.clearInspection() }

  canRemoveSource(): boolean { return this.datasetType() === 'array' && this.sources().length > 1 }

  hasLabelTable(): boolean { return this.sources().some((source) => source.type === 'table') && this.datasetType() === 'images' }

  toggleImageTable(): void {
    if (this.hasLabelTable()) this.sources.update((items) => items.filter((source) => source.type !== 'table'))
    else this.sources.update((items) => [...items, this.tableSource('labels')])
    this.clearInspection()
  }

  fieldCount(result: DatasetInspection): number { return result.sources.reduce((total, source) => total + ('columns' in source ? source.columns.length : 'arrays' in source ? source.arrays.length : source.sample_count > 0 ? 1 : 0), 0) }

  role(key: string): FieldRole { return this.roleAssignments()[key] ?? 'unused' }

  setRole(key: string, role: FieldRole): void {
    const field = this.fields().find((item) => item.key === key)
    if (field?.structural) return
    this.roleAssignments.update((current) => ({ ...current, [key]: role }))
    this.clearValidation()
  }

  inputCount(): number { return Object.values(this.roleAssignments()).filter((role) => role === 'input').length }

  roleError(): string | null {
    if (!this.inspection()) return null
    if (!this.inputCount()) return 'Define at least one model input before continuing.'
    if (this.hasRelationship() && !this.relationshipColumn()) return 'Choose the table identifier column for the source relationship.'
    return null
  }

  tableColumns(): string[] {
    const table = this.inspection()?.sources.find((source): source is Extract<DatasetSourceInspection, { type: 'table' }> => source.type === 'table')
    return table?.columns.map((column) => column.name) ?? []
  }

  hasRelationship(): boolean { return (this.datasetType() === 'composed' || this.datasetType() === 'images' && this.hasLabelTable()) }

  setRelationshipKey(value: string): void { this.relationshipKey.set(value as FileTableRelationship['file_key']); this.clearValidation() }
  setRelationshipColumn(value: string): void { this.relationshipColumn.set(value); this.roleAssignments.update((current) => ({ ...current, [`${this.tableSourceId()}::${value}`]: 'unused' })); this.clearValidation() }
  setTimestampColumn(value: string): void { this.timestampColumn.set(value); this.reserveStructuralColumn(value); this.clearValidation() }
  setGroupColumn(value: string): void { this.groupColumn.set(value); this.reserveStructuralColumn(value); this.clearValidation() }

  subsetValid(): boolean {
    const count = this.availableSamples()
    if (this.subsetMode() === 'all') return count > 0
    if (this.subsetMode() === 'first_n') return this.firstN() >= 1 && this.firstN() <= count
    return this.rangeStart() >= 0 && this.rangeEnd() >= this.rangeStart() && this.rangeEnd() < count
  }

  subsetError(): string { return this.availableSamples() === 0 ? 'The inspected source contains no samples.' : this.subsetMode() === 'first_n' ? `First N must be between 1 and ${this.availableSamples()}.` : `Range must stay between 0 and ${Math.max(this.availableSamples() - 1, 0)} with end >= start.` }

  numberValue(value: string): number { const parsed = Number(value); return Number.isFinite(parsed) ? Math.max(0, Math.floor(parsed)) : 0 }

  roleSummary(role: FieldRole): string {
    const fields = this.fields().filter((field) => this.role(field.key) === role)
    return fields.length ? fields.map((field) => field.label).join(', ') : role === 'output' ? 'None (input only)' : 'None'
  }

  datasetTypeLabel(): string { return datasetCapability(this.datasetType()).label }
  primaryPath(): string { return this.sources()[0]?.path || 'No source selected' }

  close(): void { if (!this.busy()) this.closed.emit() }
  clearValidation(): void { this.validation.set(null) }

  private clearInspection(): void { this.inspection.set(null); this.validation.set(null); this.roleAssignments.set({}); this.error.set(null) }

  private reset(): void {
    this.step.set(0); this.datasetType.set('tabular'); this.name.set(''); this.textTableMode.set(false); this.resetSources('tabular'); this.inspection.set(null); this.validation.set(null); this.error.set(null); this.subsetMode.set('all'); this.firstN.set(1); this.rangeStart.set(0); this.rangeEnd.set(0); this.relationshipColumn.set(''); this.timestampColumn.set(''); this.groupColumn.set(''); this.roleAssignments.set({})
  }

  private hydrate(document: DatasetDocument): void {
    this.step.set(0); this.datasetType.set(document.dataset_type); this.composedMode.set(document.sources.some((source) => source.type === 'text') ? 'text' : 'images'); this.name.set(document.name); this.sources.set(JSON.parse(JSON.stringify(document.sources)) as DatasetSource[]); this.inspection.set(null); this.validation.set(null); this.error.set(null); this.relationshipKey.set(document.relationships[0]?.file_key ?? 'filename'); this.relationshipColumn.set(document.relationships[0]?.table_key_column ?? ''); this.timestampColumn.set(document.sequence_definition?.timestamp_column ?? ''); this.groupColumn.set(document.sequence_definition?.group_column ?? ''); this.subsetMode.set(document.subset_definition.type); if (document.subset_definition.type === 'first_n') this.firstN.set(document.subset_definition.count); if (document.subset_definition.type === 'range') { this.rangeStart.set(document.subset_definition.start); this.rangeEnd.set(document.subset_definition.end) } this.textTableMode.set(document.dataset_type === 'text' && document.sources[0]?.type === 'table'); void this.inspectSources()
  }

  private resetSources(type: DatasetType): void {
    if (type === 'tabular' || type === 'time_series') this.sources.set([this.tableSource('table')])
    else if (type === 'images') this.sources.set([this.imageSource('images')])
    else if (type === 'text') this.sources.set([this.textSource('text')])
    else if (type === 'array') this.sources.set([this.arraySource('array-1')])
    else this.sources.set([this.composedMode() === 'text' ? this.textSource('modality') : this.imageSource('modality'), this.tableSource('table')])
  }

  private tableSource(id: string): TableSource { return { type: 'table', id, path: '', format: 'csv', has_header: true, delimiter: ',' } }
  private imageSource(id: string): ImageFolderSource { return { type: 'image_folder', id, path: '', recursive: true, label_mode: 'none' } }
  private textSource(id: string): TextSource { return { type: 'text', id, path: '', path_kind: 'folder', recursive: true } }
  private arraySource(id: string): ArraySource { return { type: 'array', id, path: '', format: 'npy' } }

  private updateSource<T extends DatasetSource>(index: number, change: Partial<T>): void { this.sources.update((items) => items.map((source, itemIndex) => itemIndex === index ? { ...source, ...change } as DatasetSource : source)) }

  sourcesReady(): boolean { return Boolean(this.name().trim()) && this.sources().length > 0 && this.sources().every((source) => Boolean(source.path.trim())) }

  private buildFields(result: DatasetInspection | null): DatasetField[] {
    if (!result) return []
    const fields: DatasetField[] = []
    for (const source of result.sources) {
      if (source.type === 'table') for (const column of source.columns) fields.push({ key: `${source.source_id}::${column.name}`, sourceId: source.source_id, label: column.name, kind: 'column', dtype: column.dtype, structural: this.isStructural(source.source_id, column.name) })
      if (source.type === 'array') for (const array of source.arrays) fields.push({ key: `${source.source_id}::${array.key}`, sourceId: source.source_id, label: `${array.key} · [${array.shape.join(' × ')}]`, kind: 'array', dtype: array.dtype, structural: false })
      if (source.type === 'image_folder') { fields.push({ key: `${source.source_id}::__image`, sourceId: source.source_id, label: 'Image samples', kind: 'image', structural: false }); if (source.class_names.length) fields.push({ key: `${source.source_id}::__label`, sourceId: source.source_id, label: 'Folder labels', kind: 'folder_label', structural: false }) }
      if (source.type === 'text') fields.push({ key: `${source.source_id}::__text`, sourceId: source.source_id, label: 'Text samples', kind: 'text', structural: false })
    }
    return fields
  }

  private isStructural(sourceId: string, column: string): boolean { return (this.hasRelationship() && this.relationshipColumn() === column && sourceId === this.tableSourceId()) || (this.datasetType() === 'time_series' && (this.timestampColumn() === column || this.groupColumn() === column) && sourceId === this.tableSourceId()) }

  private tableSourceId(): string { return this.sources().find((source) => source.type === 'table')?.id ?? '' }

  private reserveStructuralColumn(column: string): void {
    if (!column) return
    this.roleAssignments.update((current) => ({ ...current, [`${this.tableSourceId()}::${column}`]: 'unused' }))
  }

  private initializeFields(): void {
    const next: Record<string, FieldRole> = {}
    for (const field of this.fields()) next[field.key] = this.defaultRole(field)
    if (this.editing) this.applyDocumentRoles(next, this.editing)
    if (this.hasRelationship() && !this.relationshipColumn()) this.relationshipColumn.set(this.tableColumns()[0] ?? '')
    this.roleAssignments.set(next)
    this.rangeEnd.set(Math.max(0, this.availableSamples() - 1))
    this.firstN.set(Math.min(1, this.availableSamples()))
  }

  private defaultRole(field: DatasetField): FieldRole { if (field.structural) return 'unused'; if (field.kind === 'image' || field.kind === 'text') return 'input'; if (field.kind === 'folder_label') return 'output'; return 'unused' }

  private applyDocumentRoles(next: Record<string, FieldRole>, document: DatasetDocument): void {
    for (const [role, definition] of [['input', document.input_definition], ['output', document.output_definition], ['metadata', document.metadata_definition]] as const) for (const binding of definition.bindings) for (const key of this.bindingKeys(binding)) if (next[key] !== undefined) next[key] = role
  }

  private bindingKeys(binding: DatasetBinding): string[] {
    if (binding.type === 'columns') return binding.columns.map((column) => `${binding.source_id}::${column}`)
    if (binding.type === 'array') return binding.keys.map((key) => `${binding.source_id}::${key}`)
    if (binding.type === 'image') return [`${binding.source_id}::__image`]
    if (binding.type === 'text') return [`${binding.source_id}::__text`]
    return [`${binding.source_id}::__label`]
  }

  private buildDefinition(): DatasetDefinitionMutation {
    const roleDefinition = (role: FieldRole) => ({ bindings: this.bindingsFor(role) })
    const sequence_definition = this.datasetType() === 'time_series' ? { timestamp_column: this.timestampColumn() || null, group_column: this.groupColumn() || null } : null
    const relationships = this.hasRelationship() && this.relationshipColumn() ? [{ type: 'file_table' as const, file_source_id: this.sources().find((source) => source.type === 'image_folder' || source.type === 'text')?.id ?? '', file_key: this.relationshipKey(), table_source_id: this.tableSourceId(), table_key_column: this.relationshipColumn() }] : []
    return { name: this.name().trim(), dataset_type: this.datasetType(), sources: this.sources(), input_definition: roleDefinition('input'), output_definition: roleDefinition('output'), metadata_definition: roleDefinition('metadata'), relationships, sequence_definition, subset_definition: this.subsetDefinition() }
  }

  private bindingsFor(role: FieldRole): DatasetBinding[] {
    const fields = this.fields().filter((field) => this.role(field.key) === role)
    const bindings: DatasetBinding[] = []
    for (const source of this.sources()) {
      const selected = fields.filter((field) => field.sourceId === source.id)
      if (!selected.length) continue
      if (source.type === 'table') bindings.push({ type: 'columns', source_id: source.id, columns: selected.filter((field) => field.kind === 'column').map((field) => field.label) } as ColumnBinding)
      else if (source.type === 'array') bindings.push({ type: 'array', source_id: source.id, keys: selected.filter((field) => field.kind === 'array').map((field) => field.label.split(' · [')[0]) } as ArrayBinding)
      else if (source.type === 'image_folder') for (const field of selected) bindings.push(field.kind === 'folder_label' ? { type: 'folder_label', source_id: source.id } as FolderLabelBinding : { type: 'image', source_id: source.id } as ImageBinding)
      else for (const field of selected) if (field.kind === 'text') bindings.push({ type: 'text', source_id: source.id } as TextBinding)
    }
    return bindings.filter((binding) => binding.type !== 'columns' || binding.columns.length).filter((binding) => binding.type !== 'array' || binding.keys.length)
  }

  private subsetDefinition(): DatasetSubsetDefinition { return this.subsetMode() === 'all' ? { type: 'all' } : this.subsetMode() === 'first_n' ? { type: 'first_n', count: this.firstN() } : { type: 'range', start: this.rangeStart(), end: this.rangeEnd() } }

  private computeSelectedSamples(): number { const count = this.availableSamples(); if (this.subsetMode() === 'all') return count; if (this.subsetMode() === 'first_n') return Math.min(this.firstN(), count); return this.rangeEnd() >= this.rangeStart() ? Math.max(0, Math.min(this.rangeEnd(), count - 1) - this.rangeStart() + 1) : 0 }
}
