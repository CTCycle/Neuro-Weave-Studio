import { Component, computed, inject, Input, signal } from '@angular/core'
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms'
import { LucideActivity, LucideFilePlus2, LucideInfo, LucideX } from '@lucide/angular'
import type { CompilationOption } from '../../domain/types'
import { ExperimentDraftsService } from '../../core/state/experiment-drafts.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { MathFormulaComponent } from '../../shared/components/math-formula.component'

@Component({
  selector: 'nw-metrics-section',
  standalone: true,
  imports: [MathFormulaComponent],
  template: `<section class="metrics-section panel"><div class="section-heading"><div><span class="eyebrow">{{ eyebrow }}</span><h2>{{ title }}</h2></div><span class="catalog-count">{{ items.length }}</span></div><div class="metrics-list">@for (item of items; track item.id) { <article class="metrics-card"><div class="metrics-card-heading"><div><strong>{{ item.label }}</strong><code>{{ item.id }}</code></div><span class="read-only-badge">{{ badge }}</span></div><div class="metrics-card-body" [class.has-formula]="!!item.formula"><p>{{ item.description }}</p>@if (item.formula) { <div class="metrics-formula"><span>Formula</span><nw-math-formula [formula]="item.formula"></nw-math-formula></div> }</div></article> }</div></section>`,
})
export class MetricsSectionComponent {
  @Input() title = ''
  @Input() eyebrow = ''
  @Input() badge = 'READ ONLY'
  @Input() items: CompilationOption[] = []
}

@Component({
  selector: 'nw-metrics-page',
  standalone: true,
  imports: [ReactiveFormsModule, LucideActivity, LucideFilePlus2, LucideInfo, LucideX, MetricsSectionComponent, MathFormulaComponent],
  template: `
    @if (!resources.runtimeReady()) {
      <section class="page-state panel"><span class="eyebrow">METRICS CATALOG</span><h1>Loading catalog…</h1></section>
    } @else if (resources.compilationCapabilities(); as capabilities) {
      @let catalog = capabilities.keras;
      <div class="metrics-page">
        <header class="catalog-page-header"><div><span class="eyebrow">COMPILATION CATALOG</span><h1>Metrics</h1><p>Review built-in definitions and draft custom loss or metric equations for future compilation support.</p></div><div class="catalog-header-actions"><div class="environment-summary"><svg lucideActivity [size]="16"></svg><strong>{{ catalog.metrics.length + catalog.losses.length + drafts.customDefinitions().length }} definitions</strong></div><button class="primary" type="button" (click)="openWizard()"><svg lucideFilePlus2 [size]="16"></svg>Add loss / metric</button></div></header>
        <div class="metrics-note panel"><svg lucideInfo [size]="16"></svg><span>Built-in definitions remain server-owned and read-only. Custom definitions created here are session-only drafts and are not yet selectable by Compilation or Training.</span></div>
        @if (customMetrics().length || customLosses().length) { <div class="custom-metrics-grid">@if (customMetrics().length) { <nw-metrics-section title="Custom metrics" eyebrow="SESSION DRAFTS" badge="SESSION DRAFT" [items]="customMetrics()"></nw-metrics-section> } @if (customLosses().length) { <nw-metrics-section title="Custom loss functions" eyebrow="SESSION DRAFTS" badge="SESSION DRAFT" [items]="customLosses()"></nw-metrics-section> }</div> }
        <div class="metrics-grid"><nw-metrics-section title="Metrics" eyebrow="EVALUATION" [items]="catalog.metrics"></nw-metrics-section><nw-metrics-section title="Loss functions" eyebrow="OPTIMIZATION" [items]="catalog.losses"></nw-metrics-section></div>
      </div>
    } @else {
      <section class="page-state panel"><span class="eyebrow">METRICS CATALOG</span><h1>Catalog unavailable</h1><p>The runtime metadata did not include the Keras metrics and loss catalog.</p></section>
    }

    @if (wizardOpen()) {
      <div class="dialog-backdrop" (click)="wizardOpen.set(false)"><section class="formula-wizard panel" role="dialog" aria-modal="true" aria-labelledby="formula-wizard-title" (click)="$event.stopPropagation()"><header><div><span class="eyebrow">CUSTOM DEFINITION</span><h2 id="formula-wizard-title">Add loss or metric</h2></div><button class="icon-button" type="button" aria-label="Close" (click)="wizardOpen.set(false)"><svg lucideX [size]="16"></svg></button></header><form class="formula-form" [formGroup]="form"><div class="formula-grid"><label><span>Kind</span><select formControlName="kind"><option value="metric">Metric</option><option value="loss">Loss function</option></select></label><label><span>Name</span><input formControlName="label" placeholder="e.g. Balanced absolute error" /></label></div><label><span>Description</span><textarea rows="2" formControlName="description" placeholder="What the equation measures and when it is useful"></textarea></label><div class="formula-mode" role="tablist" aria-label="Equation entry mode"><button type="button" role="tab" [attr.aria-selected]="form.controls.mode.value === 'latex'" [class.active]="form.controls.mode.value === 'latex'" (click)="form.controls.mode.setValue('latex')">LaTeX</button><button type="button" role="tab" [attr.aria-selected]="form.controls.mode.value === 'builder'" [class.active]="form.controls.mode.value === 'builder'" (click)="form.controls.mode.setValue('builder')">Equation builder</button></div>@if (form.controls.mode.value === 'builder') { <div class="equation-palette"><span>Insert</span>@for (token of equationTokens; track token.label) { <button type="button" (click)="insertToken(token.value)">{{ token.label }}</button> }</div> }<label><span>Formula</span><textarea class="formula-input" rows="4" formControlName="formula" spellcheck="false" placeholder="\\frac{1}{n}\\sum_i |y_i - \\hat{y}_i|"></textarea></label><div class="formula-preview"><span>LIVE PREVIEW</span>@if (form.controls.formula.value.trim()) { <nw-math-formula [formula]="form.controls.formula.value"></nw-math-formula> } @else { <small>Enter LaTeX directly or assemble the equation with the palette.</small> }</div><div class="formula-note"><svg lucideInfo [size]="15"></svg><span>The builder edits the same LaTeX source shown above. This keeps the stored draft deterministic while providing a faster visual construction path.</span></div></form><footer class="dialog-actions"><button class="secondary" type="button" (click)="wizardOpen.set(false)">Cancel</button><button class="primary" type="button" [disabled]="form.invalid" (click)="saveDefinition()">Add session draft</button></footer></section></div>
    }
  `,
  styles: [`
    .custom-metrics-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:14px}.formula-wizard{width:min(760px,calc(100vw - 32px));max-height:calc(100vh - 32px);display:grid;grid-template-rows:auto minmax(0,1fr) auto;overflow:hidden;box-shadow:0 24px 70px rgba(0,0,0,.55)}.formula-wizard header{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;padding:18px 20px;border-bottom:1px solid var(--border)}.formula-wizard h2{margin:5px 0 0}.formula-form{min-height:0;overflow:auto;display:grid;gap:12px;padding:18px 20px}.formula-form label{display:grid;gap:5px;color:var(--muted);font-size:9px}.formula-form input,.formula-form select,.formula-form textarea{width:100%;border:1px solid var(--border);border-radius:6px;background:#0c141b;color:var(--text);padding:8px 10px}.formula-form input,.formula-form select{min-height:36px}.formula-form textarea{resize:vertical}.formula-grid{display:grid;grid-template-columns:180px minmax(0,1fr);gap:10px}.formula-mode{display:inline-flex;width:max-content;border:1px solid var(--border);border-radius:6px;overflow:hidden}.formula-mode button{min-height:32px;border:0;border-right:1px solid var(--border);background:#0c141b;color:var(--muted);padding:0 11px;cursor:pointer}.formula-mode button:last-child{border-right:0}.formula-mode button.active{background:var(--surface-3);color:var(--cyan)}.equation-palette{display:flex;align-items:center;gap:5px;flex-wrap:wrap;padding:9px;border:1px solid var(--border);border-radius:6px;background:#0b1218}.equation-palette>span{margin-right:4px;color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace}.equation-palette button{min-height:28px;border:1px solid var(--border);border-radius:4px;background:var(--surface-2);color:var(--text);padding:0 8px;font:500 10px 'IBM Plex Mono',Consolas,monospace;cursor:pointer}.equation-palette button:hover{border-color:var(--cyan);color:var(--cyan)}.formula-input{font:500 11px 'IBM Plex Mono',Consolas,monospace}.formula-preview{min-height:84px;display:grid;align-content:center;gap:6px;padding:12px;border:1px solid var(--border);border-radius:6px;background:#081016}.formula-preview>span{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace;letter-spacing:.1em}.formula-preview small{color:var(--muted);font-size:9px}.formula-preview nw-math-formula{font-size:16px}.formula-note{display:flex;align-items:flex-start;gap:8px;color:var(--muted);font-size:9px;line-height:1.45}.formula-note svg{flex:0 0 auto;color:var(--cyan)}.formula-wizard>.dialog-actions{padding:12px 20px;border-top:1px solid var(--border);justify-content:flex-end}@media(max-width:800px){.custom-metrics-grid{grid-template-columns:1fr}.formula-grid{grid-template-columns:1fr}}
  `],
})
export class MetricsPageComponent {
  readonly resources = inject(AppResourcesService)
  readonly drafts = inject(ExperimentDraftsService)
  private readonly fb = inject(FormBuilder)
  readonly wizardOpen = signal(false)
  readonly equationTokens = [
    { label: 'y', value: 'y' }, { label: 'ŷ', value: '\\hat{y}' }, { label: 'Σ', value: '\\sum_{i=1}^{n}' },
    { label: 'mean', value: '\\frac{1}{n}' }, { label: '|x|', value: '\\left|x\\right|' }, { label: 'x²', value: 'x^2' },
    { label: '√', value: '\\sqrt{x}' }, { label: 'log', value: '\\log(x)' }, { label: 'exp', value: '\\exp(x)' },
    { label: '+', value: ' + ' }, { label: '−', value: ' - ' }, { label: '×', value: ' \\times ' }, { label: '/', value: '\\frac{x}{y}' },
  ]
  readonly form = this.fb.nonNullable.group({
    kind: this.fb.nonNullable.control<'metric' | 'loss'>('metric'),
    label: ['', Validators.required],
    description: ['', Validators.required],
    mode: this.fb.nonNullable.control<'latex' | 'builder'>('latex'),
    formula: ['', Validators.required],
  })
  readonly customMetrics = computed<CompilationOption[]>(() => this.drafts.customDefinitions().filter((item) => item.kind === 'metric').map((item) => ({ id: item.id, label: item.label, description: item.description, parameters: [], formula: item.formula })))
  readonly customLosses = computed<CompilationOption[]>(() => this.drafts.customDefinitions().filter((item) => item.kind === 'loss').map((item) => ({ id: item.id, label: item.label, description: item.description, parameters: [], formula: item.formula })))

  constructor() { void this.resources.loadRuntimeMetadata() }

  openWizard(): void {
    this.form.reset({ kind: 'metric', label: '', description: '', mode: 'latex', formula: '' })
    this.wizardOpen.set(true)
  }

  insertToken(value: string): void {
    const control = this.form.controls.formula
    control.setValue(`${control.value}${value}`)
    control.markAsDirty()
  }

  saveDefinition(): void {
    if (this.form.invalid) return
    const value = this.form.getRawValue()
    this.drafts.addCustomDefinition({
      id: `custom-${value.kind}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
      kind: value.kind,
      label: value.label.trim(),
      description: value.description.trim(),
      formula: value.formula.trim(),
      createdAt: new Date().toISOString(),
    })
    this.wizardOpen.set(false)
  }
}
