import { Component, inject, signal } from '@angular/core'
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms'
import { RouterLink } from '@angular/router'
import { LucideActivity, LucideFilePlus2, LucideInfo, LucideTrash2, LucideX } from '@lucide/angular'
import { ExperimentDraftsService } from '../../core/state/experiment-drafts.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { ModalGuardDirective } from '../../shared/components/modal-guard.directive'

@Component({
  selector: 'nw-training-page',
  standalone: true,
  imports: [ReactiveFormsModule, RouterLink, LucideActivity, LucideFilePlus2, LucideInfo, LucideTrash2, LucideX, ModalGuardDirective],
  template: `
    <div class="training-page">
      <header class="catalog-page-header">
        <div><span class="eyebrow page-eyebrow">EXPERIMENT WORKSPACE</span><h1>Training</h1><p>Draft a session-only training configuration by pairing a saved model with a dataset reference. Training does not start workers or persist records yet.</p></div>
        <button class="primary" type="button" [disabled]="!drafts.hasDatasets() || !resources.projects().length" (click)="openWizard()"><svg lucideFilePlus2 [size]="16"></svg>New training</button>
      </header>

      @if (!drafts.hasDatasets()) { <section class="training-prerequisite panel"><strong>A dataset reference is required.</strong><span>Add a dataset path before creating a training session.</span><a class="secondary inline-action" routerLink="/datasets">Open Datasets</a></section> }
      @if (!resources.projects().length) { <section class="training-prerequisite panel"><strong>A saved model is required.</strong><span>Create and save an architecture before drafting a training session.</span><a class="secondary inline-action" routerLink="/models">Open Models</a></section> }

      @if (drafts.trainings().length) {
        <div class="training-grid">
          @for (training of drafts.trainings(); track training.id) {
            <article class="training-card panel">
              <div class="training-card-top"><span class="eyebrow">TRAINING DRAFT</span><button class="icon-button danger-button" type="button" [attr.aria-label]="'Remove training ' + training.name" (click)="drafts.removeTraining(training.id)"><svg lucideTrash2 [size]="15"></svg></button></div>
              <div class="training-title"><div class="training-icon"><svg lucideActivity [size]="18"></svg></div><div><h2>{{ training.name }}</h2><span>DRAFT · not started</span></div></div>
              <div class="training-links"><div><span>Model</span><strong>{{ training.modelName }}</strong></div><div><span>Dataset</span><strong>{{ training.datasetName }}</strong></div></div>
              <div class="training-stats"><div><span>Epochs</span><strong>{{ training.epochs }}</strong></div><div><span>Batch size</span><strong>{{ training.batchSize }}</strong></div><div><span>Status</span><strong>Draft</strong></div></div>
              <footer><span>Created {{ formatDate(training.createdAt) }}</span><button class="secondary" type="button" disabled title="Training execution will be added in a later implementation">Start training</button></footer>
            </article>
          }
        </div>
      } @else {
        <section class="training-empty panel"><svg lucideActivity [size]="30"></svg><h2>No training sessions</h2><p>Create a lightweight session draft once a model and dataset path are available.</p><button class="primary" type="button" [disabled]="!drafts.hasDatasets() || !resources.projects().length" (click)="openWizard()">Create training draft</button></section>
      }

      @if (wizardOpen()) {
        <div class="dialog-backdrop" nwModalGuard (click)="wizardOpen.set(false)"><section class="training-wizard panel" role="dialog" aria-modal="true" aria-labelledby="training-wizard-title" (click)="$event.stopPropagation()"><header><div><span class="eyebrow">NEW TRAINING</span><h2 id="training-wizard-title">Training setup</h2></div><button class="icon-button" type="button" aria-label="Close" (click)="wizardOpen.set(false)"><svg lucideX [size]="16"></svg></button></header><form class="training-form" [formGroup]="form"><label><span>Name</span><input formControlName="name" placeholder="Training session name" /></label><label><span>Model</span><select formControlName="modelId">@for (project of resources.projects(); track project.id) { <option [value]="project.id">{{ project.name }} · {{ project.execution_backend }}</option> }</select></label><label><span>Dataset</span><select formControlName="datasetId">@for (dataset of drafts.datasets(); track dataset.id) { <option [value]="dataset.id">{{ dataset.name }} · {{ dataset.kind }}</option> }</select></label><div class="training-form-grid"><label><span>Epochs</span><input type="number" min="1" max="100000" formControlName="epochs" /></label><label><span>Batch size</span><input type="number" min="1" max="1000000" formControlName="batchSize" /></label></div><div class="training-readiness"><svg lucideInfo [size]="15"></svg><span>This wizard creates configuration only. Optimizer, loss, metrics, callbacks, checkpoints and actual execution will be integrated with the existing model configuration and runtime boundaries later.</span></div></form><footer class="dialog-actions"><button class="secondary" type="button" (click)="wizardOpen.set(false)">Cancel</button><button class="primary" type="button" [disabled]="form.invalid" (click)="createTraining()">Create draft</button></footer></section></div>
      }
    </div>
  `,
  styles: [`
    .training-page{width:100%;min-width:0}.training-prerequisite{display:flex;align-items:flex-start;gap:10px;margin-bottom:14px;padding:12px 14px;font-size:10px}.training-prerequisite span{color:var(--muted);line-height:1.45}.training-prerequisite{align-items:center}.training-prerequisite strong{font-size:10px}.training-prerequisite a{margin-left:auto;flex:0 0 auto}.training-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.training-card{min-width:0;padding:14px;display:grid;gap:13px}.training-card-top{display:flex;align-items:center;justify-content:space-between}.training-title{display:grid;grid-template-columns:40px minmax(0,1fr);gap:10px;align-items:center}.training-icon{width:40px;height:40px;display:grid;place-items:center;border:1px solid var(--border-strong);border-radius:6px;background:#0c141b;color:var(--accent-bright)}.training-title h2{margin:0 0 3px}.training-title span{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace}.training-links{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.training-links>div,.training-stats>div{display:grid;gap:3px;padding:9px;border:1px solid var(--border);border-radius:5px;background:#0c141b}.training-links span,.training-stats span{color:var(--muted);font-size:8px}.training-links strong,.training-stats strong{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:10px}.training-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.training-card footer{display:flex;align-items:center;justify-content:space-between;gap:10px;padding-top:4px;border-top:1px solid var(--border);color:var(--muted);font-size:8px}.training-empty{min-height:320px;display:flex;align-items:center;justify-content:center;flex-direction:column;padding:40px;text-align:center;color:var(--muted)}.training-empty svg{color:var(--accent-bright)}.training-empty h2{margin:12px 0 4px;color:var(--text)}.training-empty p{font-size:10px}.training-empty button{margin-top:9px}
    .training-wizard{width:min(650px,calc(100vw - 32px));max-height:calc(100vh - 32px);display:grid;grid-template-rows:auto minmax(0,1fr) auto;overflow:hidden;box-shadow:0 24px 70px rgba(0,0,0,.55)}.training-wizard header{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;padding:18px 20px;border-bottom:1px solid var(--border)}.training-wizard h2{margin:5px 0 0}.training-form{min-height:0;overflow:auto;display:grid;gap:12px;padding:18px 20px}.training-form label{display:grid;gap:5px;color:var(--muted);font-size:9px}.training-form input,.training-form select{min-height:36px;border:1px solid var(--border);border-radius:6px;background:#0c141b;color:var(--text);padding:0 10px}.training-form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.training-readiness{display:flex;gap:8px;padding:10px;border:1px solid var(--border);border-radius:5px;color:var(--muted);font-size:9px;line-height:1.45}.training-readiness svg{flex:0 0 auto;color:var(--cyan)}.training-wizard>.dialog-actions{padding:12px 20px;border-top:1px solid var(--border);justify-content:flex-end}
    @media(max-width:850px){.training-grid{grid-template-columns:1fr}.training-links,.training-stats,.training-form-grid{grid-template-columns:1fr}.training-prerequisite{align-items:flex-start;flex-direction:column}.training-prerequisite a{margin-left:0}}
  `],
})
export class TrainingPageComponent {
  readonly drafts = inject(ExperimentDraftsService)
  readonly resources = inject(AppResourcesService)
  private readonly fb = inject(FormBuilder)
  readonly wizardOpen = signal(false)
  readonly form = this.fb.nonNullable.group({
    name: ['', [Validators.required, Validators.pattern(/\S/)]],
    modelId: ['', Validators.required],
    datasetId: ['', Validators.required],
    epochs: [10, [Validators.required, Validators.min(1), Validators.max(100000)]],
    batchSize: [32, [Validators.required, Validators.min(1), Validators.max(1000000)]],
  })

  constructor() { void this.resources.loadProjects() }

  openWizard(): void {
    const model = this.resources.projects()[0]
    const dataset = this.drafts.datasets()[0]
    if (!model || !dataset) return
    this.form.reset({ name: `${model.name} training`, modelId: model.id, datasetId: dataset.id, epochs: 10, batchSize: 32 })
    this.wizardOpen.set(true)
  }

  createTraining(): void {
    if (this.form.invalid) return
    const value = this.form.getRawValue()
    if (!value.name.trim()) return
    const model = this.resources.projects().find((item) => item.id === value.modelId)
    const dataset = this.drafts.datasets().find((item) => item.id === value.datasetId)
    if (!model || !dataset) return
    this.drafts.addTraining({
      id: this.newId(), name: value.name.trim(), modelId: model.id, modelName: model.name,
      datasetId: dataset.id, datasetName: dataset.name, epochs: Number(value.epochs), batchSize: Number(value.batchSize),
      status: 'draft', createdAt: new Date().toISOString(),
    })
    this.wizardOpen.set(false)
  }

  formatDate(value: string): string {
    const date = new Date(value)
    return Number.isNaN(date.valueOf()) ? 'recently' : new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(date)
  }
  private newId(): string { return `training-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}` }
}
