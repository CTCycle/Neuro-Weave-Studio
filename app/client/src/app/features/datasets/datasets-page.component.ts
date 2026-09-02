import { Component, computed, inject, signal } from '@angular/core'
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms'
import { LucideActivity, LucideBox, LucideFilePlus2, LucideFolderOpen, LucideInfo, LucideTrash2, LucideX } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import { DatasetKind, DatasetReference, ExperimentDraftsService } from '../../core/state/experiment-drafts.service'
import { FilesystemBrowserComponent } from '../../shared/components/filesystem-browser.component'

interface DatasetTypeOption {
  id: DatasetKind
  label: string
  description: string
  source: string
}

const DATASET_TYPES: DatasetTypeOption[] = [
  { id: 'tabular', label: 'Tabular data', description: 'CSV, TSV, spreadsheets, Parquet, Arrow, JSON or JSONL tables.', source: 'File' },
  { id: 'arrays', label: 'Serialized arrays', description: 'NumPy, tensor, SafeTensors, MAT or HDF5-style array containers.', source: 'File' },
  { id: 'images', label: 'Image folder', description: 'A directory tree containing image samples, optionally labeled by folders.', source: 'Folder' },
  { id: 'paired', label: 'Folder + reference data', description: 'A sample folder paired with a table or manifest that provides targets and metadata.', source: 'Folder + file' },
  { id: 'documents', label: 'Documents', description: 'A directory of text, Markdown, JSON, PDF or other document-oriented samples.', source: 'Folder' },
  { id: 'audio', label: 'Audio folder', description: 'A directory containing common audio files for classification or sequence workflows.', source: 'Folder' },
  { id: 'timeseries', label: 'Time series', description: 'Ordered or timestamped records stored in tabular or columnar files.', source: 'File' },
  { id: 'custom', label: 'Custom / other', description: 'Reference any other file or directory without forcing a predefined interpretation.', source: 'File or folder' },
]

@Component({
  selector: 'nw-datasets-page',
  standalone: true,
  imports: [ReactiveFormsModule, LucideActivity, LucideBox, LucideFilePlus2, LucideFolderOpen, LucideInfo, LucideTrash2, LucideX, FilesystemBrowserComponent],
  template: `
    <div class="datasets-page">
      <header class="catalog-page-header">
        <div><span class="eyebrow">DATA WORKSPACE</span><h1>Datasets</h1><p>Reference local data in place. NeuroWeave stores paths for this session and does not copy or persist dataset contents.</p></div>
        <button class="primary" type="button" (click)="openWizard()"><svg lucideFilePlus2 [size]="16"></svg>Add dataset</button>
      </header>

      <section class="dataset-policy panel"><svg lucideInfo [size]="16"></svg><div><strong>Path-backed by design</strong><span>Dataset definitions live in memory only. Future operations will read from the selected source path and may write to a designated output path. In-place modification can be supported, but should remain an explicit opt-in.</span></div></section>

      @if (!drafts.datasets().length) {
        <section class="dataset-empty panel"><svg lucideFolderOpen [size]="30"></svg><h2>No datasets referenced</h2><p>Add a local file or folder through the dataset wizard. Nothing is uploaded into NeuroWeave.</p><button class="primary" type="button" (click)="openWizard()">Add first dataset</button></section>
      } @else {
        <div class="dataset-grid">
          @for (dataset of drafts.datasets(); track dataset.id) {
            <article class="dataset-card panel">
              <div class="dataset-card-top"><span class="eyebrow">{{ kindLabel(dataset.kind) }}</span><button class="icon-button danger-button" type="button" [attr.aria-label]="'Remove dataset ' + dataset.name" (click)="drafts.removeDataset(dataset.id)"><svg lucideTrash2 [size]="15"></svg></button></div>
              <div class="dataset-card-title"><div class="dataset-icon"><svg lucideBox [size]="18"></svg></div><div><h2>{{ dataset.name }}</h2><code>{{ dataset.primaryPath }}</code></div></div>
              @if (dataset.referencePath) { <div class="dataset-reference"><span>Reference</span><code>{{ dataset.referencePath }}</code></div> }
              <div class="dataset-preview"><span>PREVIEW</span>@if (dataset.previewEntries.length) { @for (item of dataset.previewEntries; track item) { <small>{{ item }}</small> } } @else { <small>Path reference ready. Preview will be resolved when an operation inspects the source.</small> }</div>
              <div class="dataset-meta"><span>{{ splitLabel(dataset) }}</span><span>{{ dataset.preprocessing.length }} preprocessing choices</span></div>
              <footer class="dataset-actions"><button class="secondary" type="button" (click)="openExplore(dataset)">Explore</button><button class="secondary" type="button" (click)="openPreprocessing(dataset)">Preprocessing</button><button class="secondary" type="button" (click)="applySmartSplit(dataset)">Smart split</button><button class="secondary" type="button" (click)="openCustomSplit(dataset)">Custom split</button></footer>
            </article>
          }
        </div>
      }

      @if (notice()) { <button class="notice dataset-notice" type="button" (click)="notice.set(null)">{{ notice() }}</button> }

      @if (wizardOpen()) {
        <div class="dialog-backdrop" (click)="closeWizard()"><section class="dataset-wizard panel" role="dialog" aria-modal="true" aria-labelledby="dataset-wizard-title" (click)="$event.stopPropagation()">
          <header class="wizard-header"><div><span class="eyebrow">NEW DATASET · STEP {{ wizardStep() + 1 }} OF 3</span><h2 id="dataset-wizard-title">{{ wizardTitle() }}</h2></div><button class="icon-button" type="button" aria-label="Close dataset wizard" (click)="closeWizard()"><svg lucideX [size]="16"></svg></button></header>
          <div class="wizard-progress"><span [class.active]="wizardStep() >= 0"></span><span [class.active]="wizardStep() >= 1"></span><span [class.active]="wizardStep() >= 2"></span></div>
          <form [formGroup]="datasetForm" class="wizard-body">
            @if (wizardStep() === 0) {
              <div class="dataset-type-grid">@for (type of datasetTypes; track type.id) { <button type="button" class="dataset-type-option" [class.selected]="datasetForm.controls.kind.value === type.id" (click)="selectKind(type.id)"><strong>{{ type.label }}</strong><span>{{ type.description }}</span><small>{{ type.source }}</small></button> }</div>
            }
            @if (wizardStep() === 1) {
              <div class="wizard-fields">
                <label><span>Name</span><input formControlName="name" placeholder="Dataset name" /></label>
                @if (datasetForm.controls.kind.value === 'custom') { <label><span>Custom source type</span><select formControlName="customSourceMode"><option value="directory">Folder</option><option value="file">File</option></select></label> }
                <label class="path-field"><span>{{ primaryPathLabel() }}</span><div><input formControlName="primaryPath" readonly placeholder="No path selected" /><button class="secondary" type="button" (click)="openBrowser('primary')">Browse…</button></div></label>
                @if (datasetForm.controls.kind.value === 'paired') { <label class="path-field"><span>Reference table / manifest</span><div><input formControlName="referencePath" readonly placeholder="No reference file selected" /><button class="secondary" type="button" (click)="openBrowser('reference')">Browse…</button></div></label> }
                @switch (datasetForm.controls.kind.value) {
                  @case ('tabular') { <div class="option-grid"><label><span>Header</span><select formControlName="header"><option value="auto">Auto detect</option><option value="first">First row</option><option value="none">No header</option></select></label><label><span>Delimiter</span><select formControlName="delimiter"><option value="auto">Auto detect</option><option value=",">Comma</option><option value="tab">Tab</option><option value=";">Semicolon</option></select></label></div> }
                  @case ('arrays') { <div class="option-grid"><label><span>Array interpretation</span><select formControlName="arrayMode"><option value="auto">Auto</option><option value="features">Features</option><option value="features-targets">Features + targets</option></select></label><label class="check-field"><input type="checkbox" formControlName="memoryMap" /><span>Prefer memory mapping when supported</span></label></div> }
                  @case ('images') { <div class="option-grid"><label><span>Labels</span><select formControlName="labelStrategy"><option value="folders">Parent folder names</option><option value="none">Unlabeled</option></select></label><label class="check-field"><input type="checkbox" formControlName="recursive" /><span>Scan subfolders recursively</span></label></div> }
                  @case ('paired') { <div class="option-grid"><label><span>Match samples by</span><select formControlName="referenceMatch"><option value="stem">Filename stem</option><option value="filename">Full filename</option><option value="relative-path">Relative path</option></select></label><label class="check-field"><input type="checkbox" formControlName="recursive" /><span>Scan sample folder recursively</span></label></div> }
                  @case ('documents') { <div class="option-grid"><label><span>Document mode</span><select formControlName="documentMode"><option value="text">Extract text</option><option value="raw">Keep raw documents</option><option value="metadata">Metadata first</option></select></label><label class="check-field"><input type="checkbox" formControlName="recursive" /><span>Scan subfolders recursively</span></label></div> }
                  @case ('audio') { <div class="option-grid"><label><span>Labels</span><select formControlName="labelStrategy"><option value="folders">Parent folder names</option><option value="none">Unlabeled</option></select></label><label class="check-field"><input type="checkbox" formControlName="recursive" /><span>Scan subfolders recursively</span></label></div> }
                  @case ('timeseries') { <div class="option-grid"><label><span>Ordering</span><select formControlName="sequenceMode"><option value="timestamp">Timestamp column</option><option value="row-order">Row order</option><option value="grouped">Grouped sequences</option></select></label><label><span>Timestamp / group column</span><input formControlName="sequenceKey" placeholder="Optional until inspection" /></label></div> }
                  @case ('custom') { <label><span>Interpretation notes</span><textarea formControlName="customNotes" rows="4" placeholder="Describe how this source should be interpreted by future operations"></textarea></label> }
                }
              </div>
            }
            @if (wizardStep() === 2) {
              <div class="wizard-review"><div><span>Type</span><strong>{{ kindLabel(datasetForm.controls.kind.value) }}</strong></div><div><span>Primary path</span><code>{{ datasetForm.controls.primaryPath.value }}</code></div>@if (datasetForm.controls.referencePath.value) { <div><span>Reference path</span><code>{{ datasetForm.controls.referencePath.value }}</code></div> }<div><span>Storage</span><strong>Session-only path reference</strong></div><p>No dataset bytes will be imported into the application database. The source remains where it is on disk.</p></div>
            }
          </form>
          <footer class="dialog-actions wizard-actions"><button class="secondary" type="button" (click)="wizardStep() === 0 ? closeWizard() : previousWizard()">{{ wizardStep() === 0 ? 'Cancel' : 'Back' }}</button>@if (wizardStep() < 2) { <button class="primary" type="button" [disabled]="!canContinueWizard()" (click)="nextWizard()">Continue</button> } @else { <button class="primary" type="button" [disabled]="creatingDataset()" (click)="createDataset()">{{ creatingDataset() ? 'Reading preview…' : 'Add dataset reference' }}</button> }</footer>
        </section></div>
      }

      @if (activeDataset(); as dataset) {
        @if (detailMode() === 'explore') { <div class="dialog-backdrop" (click)="closeDetail()"><section class="dataset-detail panel" role="dialog" aria-modal="true" (click)="$event.stopPropagation()"><header><div><span class="eyebrow">DATASET EXPLORER</span><h2>{{ dataset.name }}</h2></div><button class="icon-button" type="button" aria-label="Close" (click)="closeDetail()"><svg lucideX [size]="16"></svg></button></header><div class="detail-body"><code class="detail-path">{{ dataset.primaryPath }}</code><h3>Source preview</h3>@if (dataset.previewEntries.length) { <div class="preview-list">@for (entry of dataset.previewEntries; track entry) { <span>{{ entry }}</span> }</div> } @else { <p>Preview metadata has not been resolved for this source yet.</p> }<h3>Interpretation</h3><div class="option-summary">@for (option of optionEntries(dataset); track option[0]) { <span>{{ option[0] }}</span><strong>{{ option[1] }}</strong> }</div></div></section></div> }
        @if (detailMode() === 'preprocess') { <div class="dialog-backdrop" (click)="closeDetail()"><section class="dataset-detail panel" role="dialog" aria-modal="true" (click)="$event.stopPropagation()"><header><div><span class="eyebrow">PREPROCESSING DRAFT</span><h2>{{ dataset.name }}</h2></div><button class="icon-button" type="button" aria-label="Close" (click)="closeDetail()"><svg lucideX [size]="16"></svg></button></header><div class="detail-body"><p>These choices are session-only placeholders for the future preprocessing pipeline. No files are modified now.</p><div class="preprocess-options">@for (option of preprocessingChoices; track option) { <label><input type="checkbox" [checked]="preprocessSelection().includes(option)" (change)="togglePreprocess(option)" /><span>{{ option }}</span></label> }</div><div class="in-place-warning"><svg lucideInfo [size]="15"></svg><span>Future preprocessing should write to a separate output path by default. Direct source modification should remain an explicit advanced choice.</span></div></div><footer class="dialog-actions"><button class="primary" type="button" (click)="savePreprocessing(dataset)">Save draft</button></footer></section></div> }
        @if (detailMode() === 'split') { <div class="dialog-backdrop" (click)="closeDetail()"><section class="dataset-detail panel" role="dialog" aria-modal="true" (click)="$event.stopPropagation()"><header><div><span class="eyebrow">CUSTOM SPLIT</span><h2>{{ dataset.name }}</h2></div><button class="icon-button" type="button" aria-label="Close" (click)="closeDetail()"><svg lucideX [size]="16"></svg></button></header><form class="detail-body split-form" [formGroup]="splitForm"><label><span>Train %</span><input type="number" min="0" max="100" formControlName="train" /></label><label><span>Validation %</span><input type="number" min="0" max="100" formControlName="validation" /></label><label><span>Test %</span><input type="number" min="0" max="100" formControlName="test" /></label><strong [class.invalid]="splitTotal() !== 100">Total: {{ splitTotal() }}%</strong></form><footer class="dialog-actions"><button class="primary" type="button" [disabled]="splitTotal() !== 100" (click)="saveCustomSplit(dataset)">Save split</button></footer></section></div> }
      }

      <nw-filesystem-browser [open]="browserOpen()" [title]="browserTitle()" [selectionMode]="browserSelectionMode()" [extensions]="browserExtensions()" (closed)="browserOpen.set(false)" (selected)="acceptBrowserSelection($event)"></nw-filesystem-browser>
    </div>
  `,
  styles: [`
    .datasets-page{width:100%;min-width:0}.dataset-policy{display:flex;gap:10px;align-items:flex-start;margin-bottom:16px;padding:12px 14px}.dataset-policy svg{flex:0 0 auto;color:var(--cyan)}.dataset-policy div{display:grid;gap:4px}.dataset-policy strong{font-size:11px}.dataset-policy span{color:var(--muted);font-size:10px;line-height:1.45}
    .dataset-empty{min-height:330px;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;padding:40px;color:var(--muted)}.dataset-empty svg{color:var(--cyan)}.dataset-empty h2{color:var(--text);margin:12px 0 4px}.dataset-empty p{max-width:520px;font-size:11px;line-height:1.5}.dataset-empty button{margin-top:10px}
    .dataset-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.dataset-card{min-width:0;padding:14px;display:flex;flex-direction:column;gap:12px}.dataset-card-top{display:flex;align-items:center;justify-content:space-between}.dataset-card-title{display:grid;grid-template-columns:38px minmax(0,1fr);gap:10px;align-items:center}.dataset-icon{width:38px;height:38px;display:grid;place-items:center;border:1px solid var(--border-strong);border-radius:6px;color:var(--cyan);background:#0c141b}.dataset-card h2{margin:0 0 3px;font-size:15px}.dataset-card code,.dataset-reference code,.detail-path,.wizard-review code{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font:500 9px 'IBM Plex Mono',Consolas,monospace}.dataset-reference{display:grid;gap:3px;padding-top:8px;border-top:1px solid var(--border)}.dataset-reference span,.dataset-preview>span{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace;letter-spacing:.1em}.dataset-preview{display:grid;gap:4px;min-height:72px;padding:10px;border:1px solid var(--border);border-radius:6px;background:#0c141b}.dataset-preview small{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text);font:500 9px 'IBM Plex Mono',Consolas,monospace}.dataset-meta{display:flex;gap:8px;flex-wrap:wrap}.dataset-meta span{padding:4px 6px;border:1px solid var(--border);border-radius:4px;color:var(--muted);font-size:8px}.dataset-actions{display:flex;gap:6px;flex-wrap:wrap;margin-top:auto;padding-top:2px}.dataset-actions button{min-height:32px;font-size:9px}.dataset-notice{position:fixed;right:18px;bottom:18px;z-index:70}
    .dataset-wizard{width:min(860px,calc(100vw - 32px));max-height:calc(100vh - 32px);display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;overflow:hidden;box-shadow:0 24px 70px rgba(0,0,0,.55)}.wizard-header,.dataset-detail header{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;padding:18px 20px;border-bottom:1px solid var(--border)}.wizard-header h2,.dataset-detail h2{margin:5px 0 0}.wizard-progress{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;padding:8px 20px}.wizard-progress span{height:2px;background:var(--border)}.wizard-progress span.active{background:var(--cyan)}.wizard-body{min-height:0;overflow:auto;padding:14px 20px 20px}.dataset-type-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.dataset-type-option{min-width:0;display:grid;gap:5px;padding:13px;border:1px solid var(--border);border-radius:6px;background:#0c141b;color:var(--text);text-align:left;cursor:pointer}.dataset-type-option:hover,.dataset-type-option.selected{border-color:var(--cyan);background:rgba(101,201,212,.06)}.dataset-type-option strong{font-size:11px}.dataset-type-option span{color:var(--muted);font-size:9px;line-height:1.4}.dataset-type-option small{color:var(--accent-bright);font:500 8px 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase}.wizard-fields{display:grid;gap:12px}.wizard-fields label,.split-form label{display:grid;gap:5px;color:var(--muted);font-size:9px}.wizard-fields input,.wizard-fields select,.wizard-fields textarea,.split-form input{width:100%;min-height:36px;border:1px solid var(--border);border-radius:6px;background:#0c141b;color:var(--text);padding:8px 10px}.wizard-fields textarea{resize:vertical}.path-field>div{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px}.path-field input{font:500 9px 'IBM Plex Mono',Consolas,monospace}.option-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.check-field{display:flex!important;align-items:center!important;gap:8px!important;padding-top:22px}.check-field input{width:16px!important;min-height:16px!important}.wizard-actions{padding:12px 20px;border-top:1px solid var(--border);justify-content:flex-end}.wizard-review{display:grid;gap:10px}.wizard-review>div{display:grid;gap:4px;padding-bottom:10px;border-bottom:1px solid var(--border)}.wizard-review span{color:var(--muted);font-size:9px}.wizard-review strong{font-size:11px}.wizard-review p{margin:6px 0 0;color:var(--muted);font-size:10px;line-height:1.5}
    .dataset-detail{width:min(680px,calc(100vw - 32px));max-height:calc(100vh - 32px);display:grid;grid-template-rows:auto minmax(0,1fr) auto;overflow:hidden;box-shadow:0 24px 70px rgba(0,0,0,.55)}.detail-body{min-height:0;overflow:auto;padding:18px 20px}.detail-body p{color:var(--muted);font-size:10px;line-height:1.5}.detail-body h3{margin:18px 0 8px;font-size:11px}.preview-list{display:grid;gap:4px}.preview-list span{padding:7px 8px;border:1px solid var(--border);border-radius:4px;background:#0c141b;font:500 9px 'IBM Plex Mono',Consolas,monospace}.option-summary{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,2fr);gap:5px 10px;font-size:9px}.option-summary span{color:var(--muted)}.preprocess-options{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.preprocess-options label{display:flex;align-items:center;gap:8px;padding:9px;border:1px solid var(--border);border-radius:5px;font-size:9px}.in-place-warning{display:flex;gap:8px;margin-top:14px;padding:10px;border:1px solid rgba(229,183,95,.35);border-radius:5px;color:var(--warning);font-size:9px;line-height:1.4}.split-form{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.split-form>strong{grid-column:1/-1}.split-form>strong.invalid{color:var(--error)}
    @media(max-width:900px){.dataset-grid{grid-template-columns:1fr}.dataset-type-grid,.option-grid{grid-template-columns:1fr}.check-field{padding-top:0}.split-form{grid-template-columns:1fr}.preprocess-options{grid-template-columns:1fr}}
  `],
})
export class DatasetsPageComponent {
  readonly drafts = inject(ExperimentDraftsService)
  private readonly api = inject(ApiService)
  private readonly fb = inject(FormBuilder)
  readonly datasetTypes = DATASET_TYPES
  readonly wizardOpen = signal(false)
  readonly wizardStep = signal(0)
  readonly browserOpen = signal(false)
  readonly browserTarget = signal<'primary' | 'reference'>('primary')
  readonly creatingDataset = signal(false)
  readonly notice = signal<string | null>(null)
  readonly activeDataset = signal<DatasetReference | null>(null)
  readonly detailMode = signal<'explore' | 'preprocess' | 'split' | null>(null)
  readonly preprocessSelection = signal<string[]>([])
  readonly preprocessingChoices = ['Normalize / standardize', 'Missing-value handling', 'Shuffle / reorder', 'Deduplicate', 'Resize / resample', 'Text cleanup']

  readonly datasetForm = this.fb.nonNullable.group({
    kind: this.fb.nonNullable.control<DatasetKind>('tabular'),
    name: ['', Validators.required],
    primaryPath: ['', Validators.required],
    referencePath: [''],
    customSourceMode: this.fb.nonNullable.control<'file' | 'directory'>('directory'),
    header: 'auto',
    delimiter: 'auto',
    arrayMode: 'auto',
    memoryMap: true,
    labelStrategy: 'folders',
    recursive: true,
    referenceMatch: 'stem',
    documentMode: 'text',
    sequenceMode: 'timestamp',
    sequenceKey: '',
    customNotes: '',
  })
  readonly splitForm = this.fb.nonNullable.group({ train: 80, validation: 10, test: 10 })
  readonly splitTotal = computed(() => Number(this.splitForm.controls.train.value) + Number(this.splitForm.controls.validation.value) + Number(this.splitForm.controls.test.value))

  openWizard(): void {
    this.datasetForm.reset({ kind: 'tabular', name: '', primaryPath: '', referencePath: '', customSourceMode: 'directory', header: 'auto', delimiter: 'auto', arrayMode: 'auto', memoryMap: true, labelStrategy: 'folders', recursive: true, referenceMatch: 'stem', documentMode: 'text', sequenceMode: 'timestamp', sequenceKey: '', customNotes: '' })
    this.wizardStep.set(0)
    this.wizardOpen.set(true)
  }
  closeWizard(): void { this.wizardOpen.set(false) }
  selectKind(kind: DatasetKind): void { this.datasetForm.controls.kind.setValue(kind); this.datasetForm.controls.primaryPath.setValue(''); this.datasetForm.controls.referencePath.setValue('') }
  nextWizard(): void { if (this.canContinueWizard()) this.wizardStep.update((step) => Math.min(2, step + 1)) }
  previousWizard(): void { this.wizardStep.update((step) => Math.max(0, step - 1)) }
  canContinueWizard(): boolean {
    if (this.wizardStep() === 0) return true
    if (this.wizardStep() === 1) return !!this.datasetForm.controls.name.value.trim() && !!this.datasetForm.controls.primaryPath.value && (this.datasetForm.controls.kind.value !== 'paired' || !!this.datasetForm.controls.referencePath.value)
    return true
  }
  wizardTitle(): string { return ['Choose a dataset type', 'Reference the source', 'Review dataset reference'][this.wizardStep()] }
  primaryPathLabel(): string { return this.browserSelectionModeFor('primary') === 'directory' ? 'Source folder' : 'Source file' }
  kindLabel(kind: DatasetKind): string { return DATASET_TYPES.find((item) => item.id === kind)?.label ?? kind }

  openBrowser(target: 'primary' | 'reference'): void { this.browserTarget.set(target); this.browserOpen.set(true) }
  browserTitle(): string { return this.browserTarget() === 'reference' ? 'Select reference data file' : `Select ${this.primaryPathLabel().toLowerCase()}` }
  browserSelectionMode(): 'file' | 'directory' { return this.browserSelectionModeFor(this.browserTarget()) }
  browserExtensions(): string[] { return this.extensionsFor(this.browserTarget()) }
  acceptBrowserSelection(path: string): void {
    const control = this.browserTarget() === 'reference' ? this.datasetForm.controls.referencePath : this.datasetForm.controls.primaryPath
    control.setValue(path)
    if (!this.datasetForm.controls.name.value.trim() && this.browserTarget() === 'primary') this.datasetForm.controls.name.setValue(this.pathName(path))
    this.browserOpen.set(false)
  }

  async createDataset(): Promise<void> {
    if (!this.canContinueWizard() || this.creatingDataset()) return
    this.creatingDataset.set(true)
    const raw = this.datasetForm.getRawValue()
    try {
      const previewEntries = await this.buildPreview(raw.kind, raw.primaryPath, raw.customSourceMode)
      this.drafts.addDataset({
        id: this.newId('dataset'), name: raw.name.trim(), kind: raw.kind, primaryPath: raw.primaryPath,
        referencePath: raw.referencePath || null,
        options: { customSourceMode: raw.customSourceMode, header: raw.header, delimiter: raw.delimiter, arrayMode: raw.arrayMode, memoryMap: raw.memoryMap, labelStrategy: raw.labelStrategy, recursive: raw.recursive, referenceMatch: raw.referenceMatch, documentMode: raw.documentMode, sequenceMode: raw.sequenceMode, sequenceKey: raw.sequenceKey, customNotes: raw.customNotes },
        previewEntries, preprocessing: [], split: { mode: 'none', train: 0, validation: 0, test: 0 }, createdAt: new Date().toISOString(),
      })
      this.wizardOpen.set(false)
      this.notice.set(`Referenced “${raw.name.trim()}” for this session.`)
    } catch (reason) {
      this.notice.set(`Dataset preview failed: ${reason instanceof Error ? reason.message : String(reason)}`)
    } finally { this.creatingDataset.set(false) }
  }

  openExplore(dataset: DatasetReference): void { this.activeDataset.set(dataset); this.detailMode.set('explore') }
  openPreprocessing(dataset: DatasetReference): void { this.activeDataset.set(dataset); this.preprocessSelection.set([...dataset.preprocessing]); this.detailMode.set('preprocess') }
  togglePreprocess(option: string): void { this.preprocessSelection.update((items) => items.includes(option) ? items.filter((item) => item !== option) : [...items, option]) }
  savePreprocessing(dataset: DatasetReference): void { const choices = [...this.preprocessSelection()]; this.drafts.updateDataset(dataset.id, (item) => ({ ...item, preprocessing: choices })); this.closeDetail(); this.notice.set('Preprocessing draft updated. No source files were modified.') }
  applySmartSplit(dataset: DatasetReference): void { this.drafts.updateDataset(dataset.id, (item) => ({ ...item, split: { mode: 'smart', train: 80, validation: 10, test: 10 } })); this.notice.set(`Smart split drafted for “${dataset.name}”: 80% train, 10% validation, 10% test.`) }
  openCustomSplit(dataset: DatasetReference): void { this.activeDataset.set(dataset); this.splitForm.setValue({ train: dataset.split.mode === 'none' ? 80 : dataset.split.train, validation: dataset.split.mode === 'none' ? 10 : dataset.split.validation, test: dataset.split.mode === 'none' ? 10 : dataset.split.test }); this.detailMode.set('split') }
  saveCustomSplit(dataset: DatasetReference): void { if (this.splitTotal() !== 100) return; const values = this.splitForm.getRawValue(); this.drafts.updateDataset(dataset.id, (item) => ({ ...item, split: { mode: 'custom', ...values } })); this.closeDetail(); this.notice.set(`Custom split updated for “${dataset.name}”.`) }
  closeDetail(): void { this.detailMode.set(null); this.activeDataset.set(null) }
  splitLabel(dataset: DatasetReference): string { return dataset.split.mode === 'none' ? 'Split not configured' : `${dataset.split.mode === 'smart' ? 'Smart' : 'Custom'} ${dataset.split.train}/${dataset.split.validation}/${dataset.split.test}` }
  optionEntries(dataset: DatasetReference): Array<[string, string]> { return Object.entries(dataset.options).filter(([, value]) => value !== '' && value !== false).map(([key, value]) => [key, String(value)]) }

  private browserSelectionModeFor(target: 'primary' | 'reference'): 'file' | 'directory' {
    if (target === 'reference') return 'file'
    const kind = this.datasetForm.controls.kind.value
    if (kind === 'images' || kind === 'paired' || kind === 'documents' || kind === 'audio') return 'directory'
    if (kind === 'custom') return this.datasetForm.controls.customSourceMode.value
    return 'file'
  }
  private extensionsFor(target: 'primary' | 'reference'): string[] {
    if (target === 'reference') return ['csv', 'tsv', 'xlsx', 'xls', 'json', 'jsonl', 'parquet']
    switch (this.datasetForm.controls.kind.value) {
      case 'tabular': return ['csv', 'tsv', 'xlsx', 'xls', 'parquet', 'feather', 'arrow', 'json', 'jsonl']
      case 'arrays': return ['npy', 'npz', 'pt', 'pth', 'safetensors', 'mat', 'h5', 'hdf5']
      case 'timeseries': return ['csv', 'tsv', 'parquet', 'feather', 'arrow', 'jsonl']
      default: return []
    }
  }
  private async buildPreview(kind: DatasetKind, path: string, customMode: 'file' | 'directory'): Promise<string[]> {
    const directorySource = kind === 'images' || kind === 'paired' || kind === 'documents' || kind === 'audio' || (kind === 'custom' && customMode === 'directory')
    if (!directorySource) return [this.pathName(path)]
    const response = await this.api.browseFilesystem(path, true, [])
    return response.items.slice(0, 8).map((item) => `${item.is_dir ? 'DIR' : 'FILE'} · ${item.name}`)
  }
  private pathName(path: string): string { return path.split(/[\\/]/).filter(Boolean).at(-1) ?? path }
  private newId(prefix: string): string { return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}` }
}
