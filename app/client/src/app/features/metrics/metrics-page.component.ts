import { Component, EventEmitter, Input, Output, inject, signal, viewChild } from '@angular/core'
import type { ElementRef } from '@angular/core'
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms'
import { LucideActivity, LucideEye, LucideFilePlus2, LucideInfo, LucideLockKeyhole, LucidePencil, LucideTrash2, LucideX } from '@lucide/angular'
import type { CompilationOption, MetricDefinition } from '../../domain/types'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { MathFormulaComponent } from '../../shared/components/math-formula.component'
import { ModalGuardDirective } from '../../shared/components/modal-guard.directive'

@Component({
  selector: 'nw-metrics-section',
  standalone: true,
  imports: [LucideEye, LucideLockKeyhole, LucidePencil, LucideTrash2, MathFormulaComponent],
  template: `<section class="metrics-section panel"><div class="section-heading"><div><span class="eyebrow">{{ eyebrow }}</span><h2>{{ title }}</h2></div><span class="catalog-count">{{ items.length }}</span></div><div class="metrics-list">@for (item of items; track item.id) { <article class="metrics-card"><div class="metrics-card-heading"><div class="metrics-card-title"><strong>{{ item.label }}</strong><code>{{ item.id }}</code></div><div class="metrics-card-actions"><button class="icon-button" type="button" title="View definition" [attr.aria-label]="'View ' + item.label" (click)="view.emit(item)"><svg lucideEye [size]="14"></svg></button>@if (item.can_edit) { <button class="icon-button" type="button" title="Edit definition" [attr.aria-label]="'Edit ' + item.label" (click)="edit.emit(item)"><svg lucidePencil [size]="14"></svg></button> } @if (item.can_delete) { <button class="icon-button danger-button" type="button" title="Delete definition" [attr.aria-label]="'Delete ' + item.label" (click)="remove.emit(item)"><svg lucideTrash2 [size]="14"></svg></button> } @if (item.locked) { <span class="locked-badge" title="Built-in definitions are managed by NeuroWeave"><svg lucideLockKeyhole [size]="12"></svg>Locked</span> }</div></div><div class="metrics-card-body" [class.has-formula]="!!item.formula"><p>{{ item.description }}</p>@if (item.formula) { <div class="metrics-formula"><span>Formula</span><nw-math-formula [formula]="item.formula" variant="prominent"></nw-math-formula></div> }</div></article> }</div></section>`,
})
export class MetricsSectionComponent {
  @Input() title = ''
  @Input() eyebrow = ''
  @Input() items: CompilationOption[] = []
  @Output() readonly view = new EventEmitter<CompilationOption>()
  @Output() readonly edit = new EventEmitter<CompilationOption>()
  @Output() readonly remove = new EventEmitter<CompilationOption>()
}

@Component({
  selector: 'nw-metrics-page',
  standalone: true,
  imports: [ReactiveFormsModule, LucideActivity, LucideFilePlus2, LucideInfo, LucideX, MetricsSectionComponent, MathFormulaComponent, ModalGuardDirective],
  template: `
    @if (!resources.runtimeReady() || !resources.metricsReady()) {
      <section class="page-state panel"><span class="eyebrow">METRICS CATALOG</span><h1>Loading catalog…</h1></section>
    } @else if (resources.hasError('metrics')) {
      <section class="page-state panel"><span class="eyebrow">METRICS CATALOG</span><h1>Catalog unavailable</h1><p>{{ resources.errors()['metrics'] }}</p><button class="secondary" type="button" (click)="resources.refreshMetrics()">Retry catalog</button></section>
    } @else if (resources.compilationCapabilities(); as capabilities) {
      @let catalog = capabilities.keras;
      <div class="metrics-page">
        <header class="catalog-page-header"><div><span class="eyebrow page-eyebrow">COMPILATION CATALOG</span><h1>Metrics</h1><p>Compare framework definitions and manage reusable custom loss and metric equations.</p></div><div class="catalog-header-actions"><div class="environment-summary"><svg lucideActivity [size]="16"></svg><strong>{{ catalog.metrics.length + catalog.losses.length }} definitions</strong></div><button class="primary" type="button" (click)="openWizard()"><svg lucideFilePlus2 [size]="16"></svg>Add loss / metric</button></div></header>
        <div class="metrics-grid"><nw-metrics-section title="Evaluation metrics" eyebrow="EVALUATION" [items]="catalog.metrics" (view)="viewDefinition($event)" (edit)="editDefinition($event)" (remove)="requestDelete($event)"></nw-metrics-section><nw-metrics-section title="Loss functions" eyebrow="OPTIMIZATION" [items]="catalog.losses" (view)="viewDefinition($event)" (edit)="editDefinition($event)" (remove)="requestDelete($event)"></nw-metrics-section></div>
      </div>
    } @else {
      <section class="page-state panel"><span class="eyebrow">METRICS CATALOG</span><h1>Catalog unavailable</h1><p>The runtime metadata did not include the Keras metrics and loss catalog.</p></section>
    }

    @if (wizardOpen()) {
      <div class="dialog-backdrop" nwModalGuard (click)="closeWizard()"><section class="formula-wizard panel" role="dialog" aria-modal="true" aria-labelledby="formula-wizard-title" (click)="$event.stopPropagation()"><header><div><span class="eyebrow">{{ editingId() ? 'EDIT DEFINITION' : 'CUSTOM DEFINITION' }}</span><h2 id="formula-wizard-title">{{ editingId() ? 'Edit loss or metric' : 'Add loss or metric' }}</h2></div><button class="icon-button" type="button" aria-label="Close" (click)="closeWizard()"><svg lucideX [size]="16"></svg></button></header><form class="formula-form" [formGroup]="form" (submit)="$event.preventDefault(); saveDefinition()"><div class="formula-grid"><label><span>Kind</span><select formControlName="kind"><option value="metric">Metric</option><option value="loss">Loss function</option></select></label><label><span>Name</span><input formControlName="label" placeholder="e.g. Balanced absolute error" /></label></div><label><span>Description</span><textarea rows="2" formControlName="description" placeholder="What the equation measures and when it is useful"></textarea></label><div class="formula-mode" role="tablist" aria-label="Equation entry mode"><button type="button" role="tab" [attr.aria-selected]="form.controls.mode.value === 'latex'" [class.active]="form.controls.mode.value === 'latex'" (click)="form.controls.mode.setValue('latex')">LaTeX</button><button type="button" role="tab" [attr.aria-selected]="form.controls.mode.value === 'builder'" [class.active]="form.controls.mode.value === 'builder'" (click)="form.controls.mode.setValue('builder')">Equation builder</button></div>@if (form.controls.mode.value === 'builder') { <div class="equation-palette"><span>Insert</span>@for (token of equationTokens; track token.label) { <button type="button" (click)="insertToken(token.value)">{{ token.label }}</button> }</div> }<label><span>Formula</span><textarea #formulaInput class="formula-input" rows="4" formControlName="formula" spellcheck="false" placeholder="\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2"></textarea></label><div class="formula-preview"><span>LIVE PREVIEW</span>@if (form.controls.formula.value.trim()) { <nw-math-formula [formula]="form.controls.formula.value" variant="prominent"></nw-math-formula> } @else { <small>Enter LaTeX directly or assemble the equation with the palette.</small> }</div><div class="formula-note"><svg lucideInfo [size]="15"></svg><span>The builder edits the same formula source shown above, so switching modes keeps the current work intact.</span></div>@if (formError()) { <p class="form-error" role="alert">{{ formError() }}</p> }</form><footer class="dialog-actions"><button class="secondary" type="button" (click)="closeWizard()">Cancel</button><button class="primary" type="button" [disabled]="form.invalid || saving()" (click)="saveDefinition()">{{ saving() ? 'Saving…' : editingId() ? 'Save changes' : 'Add definition' }}</button></footer></section></div>
    }

    @if (viewing(); as item) { <div class="dialog-backdrop" nwModalGuard (click)="viewing.set(null)"><section class="metric-view-dialog panel" role="dialog" aria-modal="true" aria-labelledby="metric-view-title" (click)="$event.stopPropagation()"><header><div><span class="eyebrow">{{ item.kind === 'loss' ? 'LOSS FUNCTION' : 'METRIC' }} · {{ item.locked ? 'BUILT IN' : 'CUSTOM' }}</span><h2 id="metric-view-title">{{ item.label }}</h2></div><button class="icon-button" type="button" aria-label="Close definition" (click)="viewing.set(null)"><svg lucideX [size]="16"></svg></button></header><div class="metric-view-body"><p>{{ item.description }}</p>@if (item.formula) { <div class="metrics-formula"><span>Formula</span><nw-math-formula [formula]="item.formula" variant="prominent"></nw-math-formula></div> }</div><footer class="dialog-actions"><button class="secondary" type="button" (click)="viewing.set(null)">Close</button>@if (item.can_edit) { <button class="primary" type="button" (click)="editDefinition(item)">Edit definition</button> }</footer></section></div> }

    @if (deleteTarget(); as item) { <div class="dialog-backdrop" nwModalGuard (click)="deleteTarget.set(null)"><section class="confirm-dialog panel" role="alertdialog" aria-modal="true" aria-labelledby="metric-delete-title" (click)="$event.stopPropagation()"><span class="eyebrow">DELETE DEFINITION</span><h2 id="metric-delete-title">Delete “{{ item.label }}”?</h2><p>This custom definition will be removed from the installation-wide compilation catalog. Models currently using it will block deletion.</p>@if (formError()) { <p class="form-error" role="alert">{{ formError() }}</p> }<div class="dialog-actions"><button class="secondary" type="button" (click)="deleteTarget.set(null)">Cancel</button><button class="primary danger-action" type="button" [disabled]="saving()" (click)="deleteDefinition()">{{ saving() ? 'Deleting…' : 'Delete definition' }}</button></div></section></div> }
  `,
  styles: [`
    .formula-wizard{width:min(760px,calc(100vw - 32px));max-height:calc(100vh - 32px);display:grid;grid-template-rows:auto minmax(0,1fr) auto;overflow:hidden;box-shadow:0 24px 70px rgba(0,0,0,.55)}.formula-wizard header,.metric-view-dialog header{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;padding:18px 20px;border-bottom:1px solid var(--border)}.formula-wizard h2,.metric-view-dialog h2{margin:5px 0 0}.formula-form{min-height:0;overflow:auto;display:grid;gap:12px;padding:18px 20px}.formula-form label{display:grid;gap:5px;color:var(--muted);font-size:9px}.formula-form input,.formula-form select,.formula-form textarea{width:100%;border:1px solid var(--border);border-radius:6px;background:#0c141b;color:var(--text);padding:8px 10px}.formula-form input,.formula-form select{min-height:36px}.formula-form textarea{resize:vertical}.formula-grid{display:grid;grid-template-columns:180px minmax(0,1fr);gap:10px}.formula-mode{display:inline-flex;width:max-content;border:1px solid var(--border);border-radius:6px;overflow:hidden}.formula-mode button{min-height:32px;border:0;border-right:1px solid var(--border);background:#0c141b;color:var(--muted);padding:0 11px;cursor:pointer}.formula-mode button:last-child{border-right:0}.formula-mode button.active{background:var(--surface-3);color:var(--cyan)}.equation-palette{display:flex;align-items:center;gap:5px;flex-wrap:wrap;padding:9px;border:1px solid var(--border);border-radius:6px;background:#0b1218}.equation-palette>span{margin-right:4px;color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace}.equation-palette button{min-height:28px;border:1px solid var(--border);border-radius:4px;background:var(--surface-2);color:var(--text);padding:0 8px;font:500 10px 'IBM Plex Mono',Consolas,monospace;cursor:pointer}.equation-palette button:hover{border-color:var(--cyan);color:var(--cyan)}.formula-input{font:500 11px 'IBM Plex Mono',Consolas,monospace}.formula-preview{min-height:84px;display:grid;align-content:center;gap:6px;padding:12px;border:1px solid var(--border);border-radius:6px;background:#081016}.formula-preview>span{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace;letter-spacing:.1em}.formula-preview small{color:var(--muted);font-size:9px}.formula-note{display:flex;align-items:flex-start;gap:8px;color:var(--muted);font-size:9px;line-height:1.45}.formula-note svg{flex:0 0 auto;color:var(--cyan)}.form-error{margin:0;padding:8px 10px;border:1px solid color-mix(in srgb,var(--error) 55%,var(--border));background:rgba(239,106,114,.08);color:var(--error);font-size:10px;line-height:1.45}.metric-view-dialog{width:min(560px,calc(100vw - 32px));display:grid;grid-template-rows:auto minmax(0,1fr) auto;overflow:hidden;box-shadow:0 24px 70px rgba(0,0,0,.55)}.metric-view-body{min-height:0;overflow:auto;padding:18px 20px}.metric-view-body>p{margin:0;color:var(--muted);font-size:11px;line-height:1.5}.metric-view-body .metrics-formula{margin-top:16px}.metric-view-dialog>.dialog-actions{padding:12px 20px;border-top:1px solid var(--border)}
    @media(max-width:800px){.formula-grid{grid-template-columns:1fr}}
  `],
})
export class MetricsPageComponent {
  readonly resources = inject(AppResourcesService)
  private readonly api = inject(ApiService)
  private readonly fb = inject(FormBuilder)
  readonly wizardOpen = signal(false)
  readonly editingId = signal<string | null>(null)
  readonly viewing = signal<MetricDefinition | null>(null)
  readonly deleteTarget = signal<MetricDefinition | null>(null)
  readonly saving = signal(false)
  readonly formError = signal<string | null>(null)
  readonly formulaInput = viewChild<ElementRef<HTMLTextAreaElement>>('formulaInput')
  readonly equationTokens = [
    { label: 'y', value: 'y' }, { label: 'ŷ', value: '\\hat{y}' }, { label: 'mean(', value: 'mean(' },
    { label: 'sum(', value: 'sum(' }, { label: 'abs(', value: 'abs(' }, { label: 'x²', value: '^2' },
    { label: 'sqrt(', value: 'sqrt(' }, { label: 'log(', value: 'log(' }, { label: 'exp(', value: 'exp(' },
    { label: '+', value: ' + ' }, { label: '−', value: ' - ' }, { label: '×', value: ' * ' }, { label: '/', value: ' / ' },
  ]
  readonly form = this.fb.nonNullable.group({
    kind: this.fb.nonNullable.control<'metric' | 'loss'>('metric'),
    label: ['', [Validators.required, Validators.pattern(/\S/)]],
    description: ['', [Validators.required, Validators.pattern(/\S/)]],
    mode: this.fb.nonNullable.control<'latex' | 'builder'>('latex'),
    formula: ['', [Validators.required, Validators.pattern(/\S/)]],
  })

  constructor() { void this.resources.loadRuntimeMetadata(); void this.resources.loadMetrics() }

  openWizard(): void {
    this.form.controls.kind.enable({ emitEvent: false })
    this.editingId.set(null)
    this.formError.set(null)
    this.form.reset({ kind: 'metric', label: '', description: '', mode: 'latex', formula: '' })
    this.wizardOpen.set(true)
  }

  viewDefinition(item: CompilationOption): void {
    const definition = this.resources.metricDefinitions().find((candidate) => candidate.id === item.id)
    if (definition) this.viewing.set(definition)
  }

  editDefinition(item: CompilationOption | MetricDefinition): void {
    const definition = 'kind' in item ? item : this.resources.metricDefinitions().find((candidate) => candidate.id === item.id)
    if (!definition || !definition.can_edit) return
    this.viewing.set(null)
    this.deleteTarget.set(null)
    this.editingId.set(definition.id)
    this.formError.set(null)
    this.form.reset({ kind: definition.kind, label: definition.label, description: definition.description, mode: 'latex', formula: definition.formula ?? '' })
    this.form.controls.kind.disable({ emitEvent: false })
    this.wizardOpen.set(true)
  }

  requestDelete(item: CompilationOption): void {
    const definition = this.resources.metricDefinitions().find((candidate) => candidate.id === item.id)
    if (definition?.can_delete) { this.formError.set(null); this.deleteTarget.set(definition) }
  }

  closeWizard(): void { if (!this.saving()) this.wizardOpen.set(false) }

  insertToken(value: string): void {
    const control = this.form.controls.formula
    const element = this.formulaInput()?.nativeElement
    const start = element?.selectionStart ?? control.value.length
    const end = element?.selectionEnd ?? start
    const next = `${control.value.slice(0, start)}${value}${control.value.slice(end)}`
    control.setValue(next)
    control.markAsDirty()
    queueMicrotask(() => {
      element?.focus()
      const cursor = start + value.length
      element?.setSelectionRange(cursor, cursor)
    })
  }

  async saveDefinition(): Promise<void> {
    if (this.form.invalid || this.saving()) return
    const value = this.form.getRawValue()
    this.saving.set(true)
    this.formError.set(null)
    try {
      if (this.editingId()) {
        await this.api.updateMetric(this.editingId()!, { label: value.label.trim(), description: value.description.trim(), formula: value.formula.trim() })
      } else {
        await this.api.createMetric({ kind: value.kind, label: value.label.trim(), description: value.description.trim(), formula: value.formula.trim() })
      }
      await Promise.all([this.resources.refreshMetrics(), this.resources.retryRuntimeMetadata()])
      this.wizardOpen.set(false)
    } catch (reason) {
      this.formError.set(reason instanceof Error ? reason.message : String(reason))
    } finally { this.saving.set(false) }
  }

  async deleteDefinition(): Promise<void> {
    const target = this.deleteTarget()
    if (!target || this.saving()) return
    this.saving.set(true)
    this.formError.set(null)
    try {
      await this.api.deleteMetric(target.id)
      await Promise.all([this.resources.refreshMetrics(), this.resources.retryRuntimeMetadata()])
      this.deleteTarget.set(null)
    } catch (reason) {
      this.formError.set(reason instanceof Error ? reason.message : String(reason))
    } finally { this.saving.set(false) }
  }
}
