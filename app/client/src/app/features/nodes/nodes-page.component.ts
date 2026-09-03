import { Component, computed, inject, signal } from '@angular/core'
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms'
import {
  LucideBox,
  LucideFilePlus2,
  LucideInfo,
  LucideLockKeyhole,
  LucidePencil,
  LucideTrash2,
  LucideX,
} from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import { CustomNodeApiService } from '../../core/api/custom-node-api.service'
import type {
  CompositeNodeSpec,
  CustomNodeDefinition,
  CustomNodeDefinitionMutation,
  CustomNodeKind,
  CustomNodeSpec,
  MathematicalNodeSpec,
  RepositoryNodeSpec,
} from '../../domain/custom-node.types'
import type { NodeManifest } from '../../domain/types'
import { MathFormulaComponent } from '../../shared/components/math-formula.component'
import { ModalGuardDirective } from '../../shared/components/modal-guard.directive'

type OriginFilter = 'all' | 'builtin' | 'custom'

interface LibraryItem {
  key: string
  id: string
  displayName: string
  category: string
  description: string
  origin: 'builtin' | 'custom'
  kind: string
  builtin?: NodeManifest
  custom?: CustomNodeDefinition
}

interface CompositeDraft {
  definitionId: string
  definitionVersion: number
  label: string
  parametersText: string
}

@Component({
  selector: 'nw-nodes-page',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    LucideBox,
    LucideFilePlus2,
    LucideInfo,
    LucideLockKeyhole,
    LucidePencil,
    LucideTrash2,
    LucideX,
    MathFormulaComponent,
    ModalGuardDirective,
  ],
  template: `
    <div class="node-library-page">
      <header class="catalog-page-header">
        <div>
          <span class="eyebrow page-eyebrow">NODE LIBRARY</span>
          <h1>Nodes</h1>
          <p>Inspect the full built-in catalog and manage installation-wide custom node definitions.</p>
        </div>
        <div class="catalog-header-actions">
          <div class="library-summary"><svg lucideBox [size]="16"></svg><strong>{{ allItems().length }} definitions</strong></div>
          <button class="primary" type="button" (click)="openCreateWizard()"><svg lucideFilePlus2 [size]="16"></svg>Add custom node</button>
        </div>
      </header>

      @if (loading()) {
        <section class="page-state panel"><span class="eyebrow">NODE LIBRARY</span><h2>Loading node definitions…</h2></section>
      } @else if (loadError()) {
        <section class="page-state panel"><span class="eyebrow">NODE LIBRARY</span><h2>Library unavailable</h2><p>{{ loadError() }}</p><button class="secondary" type="button" (click)="load()">Retry</button></section>
      } @else {
        <section class="library-workspace panel">
          <aside class="library-index">
            <div class="library-toolbar">
              <input type="search" aria-label="Search node library" placeholder="Search nodes, categories, ids…" [value]="query()" (input)="query.set($any($event.target).value)" />
              <div class="origin-filter" role="group" aria-label="Filter node origin">
                <button type="button" [class.active]="originFilter() === 'all'" (click)="originFilter.set('all')">All</button>
                <button type="button" [class.active]="originFilter() === 'builtin'" (click)="originFilter.set('builtin')">Built-in</button>
                <button type="button" [class.active]="originFilter() === 'custom'" (click)="originFilter.set('custom')">Custom</button>
              </div>
            </div>
            <div class="library-list" role="listbox" aria-label="Node definitions">
              @for (item of filteredItems(); track item.key) {
                <button class="library-row" type="button" role="option" [attr.aria-selected]="selectedKey() === item.key" [class.selected]="selectedKey() === item.key" (click)="selectedKey.set(item.key)">
                  <span class="row-main"><strong>{{ item.displayName }}</strong><code>{{ item.id }}</code></span>
                  <span class="row-meta"><span>{{ item.category }}</span><span class="origin-badge" [class.custom]="item.origin === 'custom'">{{ item.origin === 'builtin' ? 'Built-in' : 'Custom' }}</span></span>
                </button>
              } @empty {
                <div class="empty-list"><strong>No matching nodes</strong><span>Adjust the search or origin filter.</span></div>
              }
            </div>
          </aside>

          <article class="library-detail">
            @if (selectedItem(); as item) {
              <header class="detail-header">
                <div><span class="eyebrow">{{ item.origin === 'builtin' ? 'BUILT-IN DEFINITION' : 'CUSTOM DEFINITION' }}</span><h2>{{ item.displayName }}</h2><code>{{ item.id }}</code></div>
                <div class="detail-actions">
                  @if (item.origin === 'builtin') {
                    <span class="locked-badge"><svg lucideLockKeyhole [size]="13"></svg>Locked</span>
                  } @else if (item.custom; as custom) {
                    <button class="icon-button" type="button" title="Edit custom node" [attr.aria-label]="'Edit ' + item.displayName" (click)="openEditWizard(custom)"><svg lucidePencil [size]="15"></svg></button>
                    <button class="icon-button danger-button" type="button" title="Delete custom node" [attr.aria-label]="'Delete ' + item.displayName" (click)="deleteTarget.set(custom)"><svg lucideTrash2 [size]="15"></svg></button>
                  }
                </div>
              </header>
              <div class="detail-body">
                <p class="detail-description">{{ item.description }}</p>
                <dl class="definition-facts"><div><dt>Category</dt><dd>{{ item.category }}</dd></div><div><dt>Type</dt><dd>{{ item.kind }}</dd></div><div><dt>Origin</dt><dd>{{ item.origin }}</dd></div></dl>

                @if (item.builtin; as manifest) {
                  <section class="detail-section"><div class="section-heading"><span>Ports</span><small>{{ manifest.inputs.length }} inputs · {{ manifest.outputs.length }} outputs</small></div><div class="port-columns"><div><h3>Inputs</h3>@for (port of manifest.inputs; track port.id) { <div class="definition-line"><strong>{{ port.display_name }}</strong><code>{{ port.id }} · {{ port.kind }}</code><span>{{ port.description || 'No additional description.' }}</span></div> } @empty { <span class="muted-copy">No input ports.</span> }</div><div><h3>Outputs</h3>@for (port of manifest.outputs; track port.id) { <div class="definition-line"><strong>{{ port.display_name }}</strong><code>{{ port.id }} · {{ port.kind }}</code><span>{{ port.description || 'No additional description.' }}</span></div> } @empty { <span class="muted-copy">No output ports.</span> }</div></div></section>
                  <section class="detail-section"><div class="section-heading"><span>Parameters</span><small>{{ manifest.parameters.length }} controls</small></div><div class="parameter-list">@for (parameter of manifest.parameters; track parameter.id) { <div class="definition-line parameter-line"><div><strong>{{ parameter.display_name }}</strong><code>{{ parameter.id }} · {{ parameter.type }}</code></div><span>{{ parameter.description }}</span><small>Default: {{ valueLabel(parameter.editor_default ?? parameter.default) }}</small></div> } @empty { <span class="muted-copy">This node has no authoring parameters.</span> }</div></section>
                  <section class="detail-section"><div class="section-heading"><span>Implementations</span><small>{{ implementationNames(manifest).length }} declared</small></div><div class="implementation-list">@for (implementation of implementationNames(manifest); track implementation) { <span>{{ implementation }}</span> }</div><div class="documentation-ref">Documentation: <code>{{ manifest.documentation_ref }}</code></div></section>
                } @else if (item.custom; as custom) {
                  @switch (custom.spec.type) {
                    @case ('mathematical') {
                      <section class="detail-section"><div class="section-heading"><span>Mathematical operation</span><small>{{ custom.spec.input_symbols.length }} input symbols</small></div><div class="formula-detail"><nw-math-formula [formula]="custom.spec.formula" variant="prominent"></nw-math-formula><code>{{ custom.spec.formula }}</code></div><div class="symbol-list">@for (symbol of custom.spec.input_symbols; track symbol) { <span>{{ symbol }}</span> }</div></section>
                    }
                    @case ('composite') {
                      <section class="detail-section"><div class="section-heading"><span>Composite sequence</span><small>{{ custom.spec.components.length }} components</small></div><div class="component-detail-list">@for (component of custom.spec.components; track $index) { <div class="definition-line"><strong>{{ component.label || displayNameFor(component.definition_id) }}</strong><code>{{ component.definition_id }}&#64;{{ component.definition_version }}</code><span>{{ objectEntryCount(component.parameters) }} parameter values</span></div> }</div></section>
                    }
                    @case ('repository') {
                      <section class="detail-section"><div class="section-heading"><span>Public repository reference</span><small>Hugging Face</small></div><dl class="repository-facts"><div><dt>Repository</dt><dd><code>{{ custom.spec.repository_id }}</code></dd></div><div><dt>Revision</dt><dd>{{ custom.spec.revision }}</dd></div><div><dt>Artifact</dt><dd>{{ custom.spec.artifact_type }}</dd></div>@if (custom.spec.entrypoint) { <div><dt>Entrypoint</dt><dd><code>{{ custom.spec.entrypoint }}</code></dd></div> }</dl><div class="safety-note"><svg lucideInfo [size]="14"></svg><span>This definition stores a public repository reference only. It does not enable remote-code execution.</span></div></section>
                    }
                  }
                  <section class="detail-section timestamps"><span>Created {{ dateLabel(custom.created_at) }}</span><span>Updated {{ dateLabel(custom.updated_at) }}</span></section>
                }
              </div>
            } @else {
              <div class="empty-detail"><strong>Select a node</strong><span>Choose a definition to inspect its complete contract.</span></div>
            }
          </article>
        </section>
      }
    </div>

    @if (wizardOpen()) {
      <div class="dialog-backdrop" nwModalGuard (click)="closeWizard()">
        <section class="node-wizard panel" role="dialog" aria-modal="true" aria-labelledby="node-wizard-title" (click)="$event.stopPropagation()">
          <header><div><span class="eyebrow">{{ editingId() ? 'EDIT CUSTOM NODE' : 'CUSTOM NODE WIZARD' }}</span><h2 id="node-wizard-title">{{ wizardTitle() }}</h2></div><button class="icon-button" type="button" aria-label="Close custom node wizard" (click)="closeWizard()"><svg lucideX [size]="16"></svg></button></header>
          <div class="wizard-progress"><span [class.active]="wizardStep() === 1" [class.complete]="wizardStep() > 1">1 · Type</span><span [class.active]="wizardStep() === 2" [class.complete]="wizardStep() > 2">2 · Identity</span><span [class.active]="wizardStep() === 3">3 · Definition</span></div>

          <div class="wizard-body">
            @if (wizardStep() === 1) {
              <div class="type-picker">
                <button type="button" [class.selected]="selectedKind() === 'mathematical'" (click)="selectedKind.set('mathematical')"><strong>Mathematical operation</strong><span>Define a tensor-style operation using LaTeX or the guided equation builder.</span></button>
                <button type="button" [class.selected]="selectedKind() === 'composite'" (click)="selectedKind.set('composite')"><strong>Composite node</strong><span>Assemble a reusable sequential block from existing built-in nodes and their parameters.</span></button>
                <button type="button" [class.selected]="selectedKind() === 'repository'" (click)="selectedKind.set('repository')"><strong>Public repository</strong><span>Reference a public model or layer from Hugging Face without enabling arbitrary remote code.</span></button>
              </div>
            } @else if (wizardStep() === 2) {
              <form class="identity-form" [formGroup]="identityForm"><label><span>Name</span><input formControlName="displayName" placeholder="e.g. Gated projection" /></label><label><span>Category</span><input formControlName="category" placeholder="e.g. custom blocks" /></label><label class="wide-field"><span>Description</span><textarea rows="4" formControlName="description" placeholder="Explain what the node represents and when it should be used."></textarea></label></form>
            } @else {
              @switch (selectedKind()) {
                @case ('mathematical') {
                  <form class="definition-form" [formGroup]="mathForm"><div class="formula-mode" role="tablist" aria-label="Equation entry mode"><button type="button" [class.active]="mathForm.controls.mode.value === 'latex'" (click)="mathForm.controls.mode.setValue('latex')">LaTeX</button><button type="button" [class.active]="mathForm.controls.mode.value === 'builder'" (click)="mathForm.controls.mode.setValue('builder')">Equation builder</button></div>@if (mathForm.controls.mode.value === 'builder') { <div class="equation-palette"><span>Insert</span>@for (token of equationTokens; track token.label) { <button type="button" (click)="insertEquationToken(token.value)">{{ token.label }}</button> }</div> }<label><span>Formula</span><textarea rows="5" formControlName="formula" spellcheck="false" placeholder="\\sigma(Wx + b)"></textarea></label><label><span>Input symbols</span><input formControlName="inputSymbols" placeholder="x, y" /><small>Comma-separated identifiers used by the formula.</small></label><div class="formula-preview"><span>LIVE PREVIEW</span>@if (mathForm.controls.formula.value.trim()) { <nw-math-formula [formula]="mathForm.controls.formula.value" variant="prominent"></nw-math-formula> } @else { <small>Enter a formula to preview it.</small> }</div></form>
                }
                @case ('composite') {
                  <div class="composite-builder"><div class="component-adder"><label><span>Built-in component</span><select [value]="componentCandidate()" (change)="componentCandidate.set($any($event.target).value)"><option value="">Select a node…</option>@for (manifest of composableManifests(); track manifest.id) { <option [value]="manifest.id">{{ manifest.display_name }} · {{ manifest.category }}</option> }</select></label><button class="secondary" type="button" [disabled]="!componentCandidate()" (click)="addComponent()">Add component</button></div><p class="builder-hint">Components are stored as an ordered sequential block. Parameter JSON is prefilled from the selected built-in node so the complete parameter contract remains visible.</p><div class="component-editor-list">@for (component of compositeComponents(); track $index; let index = $index) { <section class="component-editor"><header><div><strong>{{ component.label }}</strong><code>{{ component.definitionId }}&#64;{{ component.definitionVersion }}</code></div><button class="icon-button danger-button" type="button" [attr.aria-label]="'Remove ' + component.label" (click)="removeComponent(index)"><svg lucideTrash2 [size]="14"></svg></button></header><label><span>Parameter values</span><textarea rows="7" spellcheck="false" [value]="component.parametersText" (input)="updateComponentParameters(index, $any($event.target).value)"></textarea></label></section> } @empty { <div class="empty-composite"><strong>No components yet</strong><span>Add at least one built-in node to define the composite.</span></div> }</div></div>
                }
                @case ('repository') {
                  <form class="definition-form" [formGroup]="repositoryForm"><div class="repository-provider"><strong>Hugging Face</strong><span>Public repository reference</span></div><label><span>Repository id</span><input formControlName="repositoryId" placeholder="namespace/repository" /></label><div class="form-columns"><label><span>Revision</span><input formControlName="revision" placeholder="main" /></label><label><span>Artifact type</span><select formControlName="artifactType"><option value="model">Model</option><option value="layer">Layer</option></select></label></div><label><span>Entrypoint or class, optional</span><input formControlName="entrypoint" placeholder="e.g. AutoModel" /></label><div class="safety-note"><svg lucideInfo [size]="14"></svg><span>NeuroWeave stores the repository coordinates only. This wizard never enables trust_remote_code or executes repository code in the server process.</span></div></form>
                }
              }
            }
            @if (formError()) { <p class="form-error" role="alert">{{ formError() }}</p> }
          </div>

          <footer class="dialog-actions"><button class="secondary" type="button" [disabled]="saving()" (click)="wizardStep() === 1 ? closeWizard() : previousStep()">{{ wizardStep() === 1 ? 'Cancel' : 'Back' }}</button>@if (wizardStep() < 3) { <button class="primary" type="button" [disabled]="!canContinue()" (click)="nextStep()">Continue</button> } @else { <button class="primary" type="button" [disabled]="!canSave() || saving()" (click)="saveDefinition()">{{ saving() ? 'Saving…' : editingId() ? 'Save changes' : 'Add custom node' }}</button> }</footer>
        </section>
      </div>
    }

    @if (deleteTarget(); as target) {
      <div class="dialog-backdrop" nwModalGuard (click)="deleteTarget.set(null)"><section class="confirm-dialog panel" role="alertdialog" aria-modal="true" aria-labelledby="custom-node-delete-title" (click)="$event.stopPropagation()"><span class="eyebrow">DELETE CUSTOM NODE</span><h2 id="custom-node-delete-title">Delete “{{ target.display_name }}”?</h2><p>The installation-wide custom definition will be removed. Built-in definitions are never affected.</p>@if (formError()) { <p class="form-error" role="alert">{{ formError() }}</p> }<div class="dialog-actions"><button class="secondary" type="button" (click)="deleteTarget.set(null)">Cancel</button><button class="primary danger-action" type="button" [disabled]="saving()" (click)="deleteDefinition()">{{ saving() ? 'Deleting…' : 'Delete custom node' }}</button></div></section></div>
    }
  `,
  styles: [`
    .node-library-page{display:grid;gap:18px;min-width:0}.library-summary{display:inline-flex;align-items:center;gap:8px;color:var(--muted);font-size:10px}.library-summary svg{color:var(--cyan)}.library-workspace{display:grid;grid-template-columns:minmax(300px,38%) minmax(0,1fr);min-height:620px;overflow:hidden}.library-index{min-width:0;border-right:1px solid var(--border);display:grid;grid-template-rows:auto minmax(0,1fr)}.library-toolbar{display:grid;gap:10px;padding:14px;border-bottom:1px solid var(--border)}.library-toolbar input,.identity-form input,.identity-form textarea,.definition-form input,.definition-form textarea,.definition-form select,.component-adder select,.component-editor textarea{width:100%;border:1px solid var(--border);border-radius:6px;background:#0c141b;color:var(--text);padding:8px 10px}.library-toolbar input{min-height:36px}.origin-filter,.formula-mode{display:inline-flex;width:max-content;border:1px solid var(--border);border-radius:6px;overflow:hidden}.origin-filter button,.formula-mode button{border:0;border-right:1px solid var(--border);background:#0c141b;color:var(--muted);min-height:30px;padding:0 10px;cursor:pointer}.origin-filter button:last-child,.formula-mode button:last-child{border-right:0}.origin-filter button.active,.formula-mode button.active{background:var(--surface-3);color:var(--cyan)}.library-list{min-height:0;overflow:auto}.library-row{width:100%;border:0;border-bottom:1px solid var(--border);background:transparent;color:var(--text);display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;text-align:left;padding:11px 14px;cursor:pointer}.library-row:hover,.library-row.selected{background:var(--surface-2)}.library-row.selected{box-shadow:inset 2px 0 0 var(--cyan)}.row-main{min-width:0;display:grid;gap:3px}.row-main strong{font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.row-main code{color:var(--muted);font-size:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.row-meta{display:grid;justify-items:end;gap:4px;color:var(--muted);font-size:8px}.origin-badge{border:1px solid var(--border);border-radius:999px;padding:2px 6px;font:500 7px 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase}.origin-badge.custom{color:var(--cyan);border-color:color-mix(in srgb,var(--cyan) 40%,var(--border))}.empty-list,.empty-detail,.empty-composite{display:grid;align-content:center;justify-items:center;gap:5px;color:var(--muted);padding:32px;text-align:center}.empty-list strong,.empty-detail strong,.empty-composite strong{color:var(--text);font-size:11px}.empty-list span,.empty-detail span,.empty-composite span{font-size:9px}.library-detail{min-width:0;display:grid;grid-template-rows:auto minmax(0,1fr)}.detail-header{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;padding:18px 20px;border-bottom:1px solid var(--border)}.detail-header h2{margin:5px 0 4px}.detail-header code{color:var(--muted);font-size:9px}.detail-actions{display:flex;align-items:center;gap:6px}.locked-badge{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--border);border-radius:999px;color:var(--muted);padding:5px 8px;font-size:8px}.detail-body{min-height:0;overflow:auto;padding:20px;display:grid;align-content:start;gap:18px}.detail-description{margin:0;color:var(--muted);font-size:11px;line-height:1.6}.definition-facts,.repository-facts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;margin:0;background:var(--border);border:1px solid var(--border);border-radius:6px;overflow:hidden}.definition-facts>div,.repository-facts>div{background:var(--surface);padding:10px}.definition-facts dt,.repository-facts dt{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase}.definition-facts dd,.repository-facts dd{margin:4px 0 0;color:var(--text);font-size:10px}.detail-section{display:grid;gap:10px;border-top:1px solid var(--border);padding-top:15px}.section-heading{display:flex;align-items:center;justify-content:space-between;gap:12px}.section-heading>span{font-weight:600;font-size:11px}.section-heading small{color:var(--muted);font-size:8px}.port-columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.port-columns h3{margin:0 0 8px;color:var(--muted);font-size:9px;text-transform:uppercase}.definition-line{display:grid;gap:3px;padding:9px 0;border-bottom:1px solid var(--border)}.definition-line:last-child{border-bottom:0}.definition-line strong{font-size:10px}.definition-line code{color:var(--cyan);font-size:8px}.definition-line span,.definition-line small,.muted-copy{color:var(--muted);font-size:9px;line-height:1.45}.parameter-line{grid-template-columns:minmax(180px,.7fr) minmax(0,1.3fr) auto;align-items:start;gap:14px}.parameter-line>div{display:grid;gap:3px}.implementation-list,.symbol-list{display:flex;gap:6px;flex-wrap:wrap}.implementation-list span,.symbol-list span{border:1px solid var(--border);border-radius:4px;background:var(--surface-2);padding:5px 7px;color:var(--text);font:500 8px 'IBM Plex Mono',Consolas,monospace}.documentation-ref{color:var(--muted);font-size:9px}.documentation-ref code{color:var(--text)}.formula-detail{display:grid;gap:8px;padding:14px;border:1px solid var(--border);border-radius:6px;background:#081016}.formula-detail code{color:var(--muted);font-size:9px;white-space:pre-wrap;overflow-wrap:anywhere}.repository-facts{grid-template-columns:repeat(2,minmax(0,1fr))}.safety-note{display:flex;gap:8px;align-items:flex-start;color:var(--muted);font-size:9px;line-height:1.5}.safety-note svg{flex:0 0 auto;color:var(--cyan)}.timestamps{display:flex;justify-content:space-between;color:var(--muted);font-size:8px}.node-wizard{width:min(820px,calc(100vw - 32px));max-height:calc(100vh - 32px);display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;overflow:hidden;box-shadow:0 24px 70px rgba(0,0,0,.55)}.node-wizard>header{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;padding:18px 20px;border-bottom:1px solid var(--border)}.node-wizard h2{margin:5px 0 0}.wizard-progress{display:flex;gap:16px;padding:9px 20px;border-bottom:1px solid var(--border);color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace}.wizard-progress span.active{color:var(--cyan)}.wizard-progress span.complete{color:var(--text)}.wizard-body{min-height:0;overflow:auto;padding:20px}.type-picker{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.type-picker button{min-height:150px;border:1px solid var(--border);border-radius:8px;background:var(--surface-2);color:var(--text);display:grid;align-content:start;gap:10px;padding:16px;text-align:left;cursor:pointer}.type-picker button:hover,.type-picker button.selected{border-color:var(--cyan);background:color-mix(in srgb,var(--surface-2) 88%,var(--cyan))}.type-picker strong{font-size:11px}.type-picker span{color:var(--muted);font-size:9px;line-height:1.5}.identity-form,.definition-form{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.identity-form label,.definition-form label,.component-adder label,.component-editor label{display:grid;gap:5px;color:var(--muted);font-size:9px}.identity-form input,.definition-form input,.definition-form select,.component-adder select{min-height:36px}.identity-form textarea,.definition-form textarea,.component-editor textarea{resize:vertical}.wide-field,.definition-form>label,.formula-mode,.equation-palette,.formula-preview,.repository-provider,.safety-note{grid-column:1/-1}.formula-mode{margin-bottom:2px}.equation-palette{display:flex;align-items:center;gap:5px;flex-wrap:wrap;padding:9px;border:1px solid var(--border);border-radius:6px;background:#0b1218}.equation-palette>span{margin-right:4px;color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace}.equation-palette button{min-height:28px;border:1px solid var(--border);border-radius:4px;background:var(--surface-2);color:var(--text);padding:0 8px;font:500 10px 'IBM Plex Mono',Consolas,monospace;cursor:pointer}.formula-preview{min-height:88px;display:grid;align-content:center;gap:6px;padding:12px;border:1px solid var(--border);border-radius:6px;background:#081016}.formula-preview>span{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace}.formula-preview small{color:var(--muted);font-size:9px}.composite-builder{display:grid;gap:12px}.component-adder{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:end;gap:10px}.builder-hint{margin:0;color:var(--muted);font-size:9px;line-height:1.5}.component-editor-list{display:grid;gap:10px}.component-editor{border:1px solid var(--border);border-radius:6px;background:var(--surface-2);overflow:hidden}.component-editor>header{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;padding:10px 12px;border-bottom:1px solid var(--border)}.component-editor>header>div{display:grid;gap:3px}.component-editor strong{font-size:10px}.component-editor code{color:var(--cyan);font-size:8px}.component-editor label{padding:10px 12px}.component-editor textarea{font:500 9px 'IBM Plex Mono',Consolas,monospace}.repository-provider{display:flex;align-items:center;justify-content:space-between;gap:12px;border:1px solid var(--border);border-radius:6px;background:var(--surface-2);padding:10px 12px}.repository-provider strong{font-size:10px}.repository-provider span{color:var(--muted);font-size:9px}.form-columns{grid-column:1/-1;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.form-columns label{display:grid;gap:5px;color:var(--muted);font-size:9px}.form-error{margin:14px 0 0;padding:8px 10px;border:1px solid color-mix(in srgb,var(--error) 55%,var(--border));background:rgba(239,106,114,.08);color:var(--error);font-size:10px;line-height:1.45}
    @media(max-width:900px){.library-workspace{grid-template-columns:1fr}.library-index{border-right:0;border-bottom:1px solid var(--border);max-height:380px}.type-picker{grid-template-columns:1fr}.type-picker button{min-height:110px}.parameter-line{grid-template-columns:1fr}.port-columns{grid-template-columns:1fr}}
    @media(max-width:650px){.definition-facts,.repository-facts,.identity-form,.definition-form,.form-columns{grid-template-columns:1fr}.component-adder{grid-template-columns:1fr}.timestamps{display:grid;gap:4px}}
  `],
})
export class NodesPageComponent {
  private readonly api = inject(ApiService)
  private readonly customApi = inject(CustomNodeApiService)
  private readonly fb = inject(FormBuilder)

  readonly builtins = signal<NodeManifest[]>([])
  readonly customs = signal<CustomNodeDefinition[]>([])
  readonly loading = signal(true)
  readonly loadError = signal<string | null>(null)
  readonly query = signal('')
  readonly originFilter = signal<OriginFilter>('all')
  readonly selectedKey = signal<string | null>(null)
  readonly wizardOpen = signal(false)
  readonly wizardStep = signal<1 | 2 | 3>(1)
  readonly selectedKind = signal<CustomNodeKind | null>(null)
  readonly editingId = signal<string | null>(null)
  readonly saving = signal(false)
  readonly formError = signal<string | null>(null)
  readonly deleteTarget = signal<CustomNodeDefinition | null>(null)
  readonly componentCandidate = signal('')
  readonly compositeComponents = signal<CompositeDraft[]>([])

  readonly allItems = computed<LibraryItem[]>(() => [
    ...this.builtins().map((manifest): LibraryItem => ({
      key: `builtin:${manifest.id}`,
      id: manifest.id,
      displayName: manifest.display_name,
      category: manifest.category,
      description: manifest.description,
      origin: 'builtin',
      kind: manifest.operation ? `${manifest.operation.family} operation` : manifest.semantic_role ?? 'operation',
      builtin: manifest,
    })),
    ...this.customs().map((definition): LibraryItem => ({
      key: `custom:${definition.id}`,
      id: definition.id,
      displayName: definition.display_name,
      category: definition.category,
      description: definition.description,
      origin: 'custom',
      kind: this.kindLabel(definition.kind),
      custom: definition,
    })),
  ].sort((left, right) => left.category.localeCompare(right.category) || left.displayName.localeCompare(right.displayName)))

  readonly filteredItems = computed(() => {
    const query = this.query().trim().toLocaleLowerCase()
    const origin = this.originFilter()
    return this.allItems().filter((item) => {
      if (origin !== 'all' && item.origin !== origin) return false
      if (!query) return true
      return `${item.displayName} ${item.id} ${item.category} ${item.description}`.toLocaleLowerCase().includes(query)
    })
  })

  readonly selectedItem = computed(() => this.allItems().find((item) => item.key === this.selectedKey()) ?? null)
  readonly composableManifests = computed(() => this.builtins().filter((manifest) => !manifest.deprecated && manifest.semantic_role === 'operation'))

  readonly identityForm = this.fb.nonNullable.group({
    displayName: ['', [Validators.required, Validators.pattern(/\S/)]],
    category: ['', [Validators.required, Validators.pattern(/\S/)]],
    description: ['', [Validators.required, Validators.pattern(/\S/)]],
  })
  readonly mathForm = this.fb.nonNullable.group({
    mode: this.fb.nonNullable.control<'latex' | 'builder'>('latex'),
    formula: ['', [Validators.required, Validators.pattern(/\S/)]],
    inputSymbols: ['x', [Validators.required, Validators.pattern(/\S/)]],
  })
  readonly repositoryForm = this.fb.nonNullable.group({
    repositoryId: ['', [Validators.required, Validators.pattern(/^[^/\s]+\/[^/\s]+$/)]],
    revision: ['main', [Validators.required, Validators.pattern(/\S/)]],
    artifactType: this.fb.nonNullable.control<'model' | 'layer'>('model'),
    entrypoint: [''],
  })
  readonly equationTokens = [
    { label: 'x', value: 'x' }, { label: 'y', value: 'y' }, { label: 'W', value: 'W' },
    { label: 'b', value: 'b' }, { label: 'σ(', value: '\\sigma(' }, { label: 'ReLU(', value: '\\operatorname{ReLU}(' },
    { label: '+', value: ' + ' }, { label: '−', value: ' - ' }, { label: '×', value: ' \\cdot ' },
    { label: '(', value: '(' }, { label: ')', value: ')' }, { label: 'frac', value: '\\frac{}{}' },
  ]

  constructor() { void this.load() }

  async load(): Promise<void> {
    this.loading.set(true)
    this.loadError.set(null)
    try {
      const [builtins, customs] = await Promise.all([this.api.catalog(), this.customApi.list()])
      this.builtins.set(builtins)
      this.customs.set(customs)
      const items = [
        ...builtins.map((item) => `builtin:${item.id}`),
        ...customs.map((item) => `custom:${item.id}`),
      ]
      if (!this.selectedKey() || !items.includes(this.selectedKey()!)) this.selectedKey.set(items[0] ?? null)
    } catch (reason) {
      this.loadError.set(reason instanceof Error ? reason.message : String(reason))
    } finally {
      this.loading.set(false)
    }
  }

  openCreateWizard(): void {
    this.editingId.set(null)
    this.selectedKind.set(null)
    this.wizardStep.set(1)
    this.formError.set(null)
    this.identityForm.reset({ displayName: '', category: 'custom', description: '' })
    this.mathForm.reset({ mode: 'latex', formula: '', inputSymbols: 'x' })
    this.repositoryForm.reset({ repositoryId: '', revision: 'main', artifactType: 'model', entrypoint: '' })
    this.compositeComponents.set([])
    this.componentCandidate.set('')
    this.wizardOpen.set(true)
  }

  openEditWizard(definition: CustomNodeDefinition): void {
    this.editingId.set(definition.id)
    this.selectedKind.set(definition.kind)
    this.wizardStep.set(2)
    this.formError.set(null)
    this.identityForm.reset({
      displayName: definition.display_name,
      category: definition.category,
      description: definition.description,
    })
    this.compositeComponents.set([])
    if (definition.spec.type === 'mathematical') {
      this.mathForm.reset({ mode: 'latex', formula: definition.spec.formula, inputSymbols: definition.spec.input_symbols.join(', ') })
    } else if (definition.spec.type === 'composite') {
      this.compositeComponents.set(definition.spec.components.map((component) => ({
        definitionId: component.definition_id,
        definitionVersion: component.definition_version,
        label: component.label || this.displayNameFor(component.definition_id),
        parametersText: JSON.stringify(component.parameters, null, 2),
      })))
    } else {
      this.repositoryForm.reset({
        repositoryId: definition.spec.repository_id,
        revision: definition.spec.revision,
        artifactType: definition.spec.artifact_type,
        entrypoint: definition.spec.entrypoint ?? '',
      })
    }
    this.wizardOpen.set(true)
  }

  closeWizard(): void { if (!this.saving()) this.wizardOpen.set(false) }

  previousStep(): void {
    if (this.wizardStep() === 3) this.wizardStep.set(2)
    else if (this.wizardStep() === 2 && !this.editingId()) this.wizardStep.set(1)
    else this.closeWizard()
  }

  nextStep(): void {
    if (!this.canContinue()) return
    if (this.wizardStep() === 1) this.wizardStep.set(2)
    else if (this.wizardStep() === 2) this.wizardStep.set(3)
  }

  canContinue(): boolean {
    if (this.wizardStep() === 1) return this.selectedKind() !== null
    if (this.wizardStep() === 2) return this.identityForm.valid
    return false
  }

  canSave(): boolean {
    if (this.wizardStep() !== 3 || !this.selectedKind() || !this.identityForm.valid) return false
    if (this.selectedKind() === 'mathematical') return this.mathForm.valid
    if (this.selectedKind() === 'composite') return this.compositeComponents().length > 0
    return this.repositoryForm.valid
  }

  addComponent(): void {
    const manifest = this.builtins().find((item) => item.id === this.componentCandidate())
    if (!manifest) return
    const defaults = Object.fromEntries(manifest.parameters.map((parameter) => [
      parameter.id,
      parameter.editor_default ?? parameter.default,
    ]))
    this.compositeComponents.update((components) => [...components, {
      definitionId: manifest.id,
      definitionVersion: manifest.version,
      label: manifest.display_name,
      parametersText: JSON.stringify(defaults, null, 2),
    }])
    this.componentCandidate.set('')
  }

  removeComponent(index: number): void {
    this.compositeComponents.update((components) => components.filter((_, itemIndex) => itemIndex !== index))
  }

  updateComponentParameters(index: number, parametersText: string): void {
    this.compositeComponents.update((components) => components.map((component, itemIndex) => itemIndex === index ? { ...component, parametersText } : component))
  }

  insertEquationToken(value: string): void {
    const control = this.mathForm.controls.formula
    control.setValue(`${control.value}${value}`)
    control.markAsDirty()
  }

  async saveDefinition(): Promise<void> {
    if (!this.canSave() || this.saving()) return
    this.formError.set(null)
    let spec: CustomNodeSpec
    try {
      spec = this.buildSpec()
    } catch (reason) {
      this.formError.set(reason instanceof Error ? reason.message : String(reason))
      return
    }
    const identity = this.identityForm.getRawValue()
    const payload: CustomNodeDefinitionMutation = {
      display_name: identity.displayName.trim(),
      category: identity.category.trim(),
      description: identity.description.trim(),
      spec,
    }
    this.saving.set(true)
    try {
      const saved = this.editingId()
        ? await this.customApi.update(this.editingId()!, payload)
        : await this.customApi.create(payload)
      await this.refreshCustoms()
      this.selectedKey.set(`custom:${saved.id}`)
      this.wizardOpen.set(false)
    } catch (reason) {
      this.formError.set(reason instanceof Error ? reason.message : String(reason))
    } finally {
      this.saving.set(false)
    }
  }

  async deleteDefinition(): Promise<void> {
    const target = this.deleteTarget()
    if (!target || this.saving()) return
    this.saving.set(true)
    this.formError.set(null)
    try {
      await this.customApi.delete(target.id)
      this.deleteTarget.set(null)
      if (this.selectedKey() === `custom:${target.id}`) this.selectedKey.set(null)
      await this.refreshCustoms()
    } catch (reason) {
      this.formError.set(reason instanceof Error ? reason.message : String(reason))
    } finally {
      this.saving.set(false)
    }
  }

  private buildSpec(): CustomNodeSpec {
    const kind = this.selectedKind()
    if (kind === 'mathematical') {
      const value = this.mathForm.getRawValue()
      const inputSymbols = value.inputSymbols.split(',').map((item) => item.trim()).filter(Boolean)
      if (!inputSymbols.length) throw new Error('Add at least one input symbol.')
      const spec: MathematicalNodeSpec = { type: 'mathematical', formula: value.formula.trim(), input_symbols: inputSymbols }
      return spec
    }
    if (kind === 'composite') {
      const components = this.compositeComponents().map((component) => {
        let parameters: Record<string, unknown>
        try {
          const parsed: unknown = JSON.parse(component.parametersText)
          if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) throw new Error()
          parameters = parsed as Record<string, unknown>
        } catch {
          throw new Error(`Parameter values for ${component.label} must be a JSON object.`)
        }
        return {
          definition_id: component.definitionId,
          definition_version: component.definitionVersion,
          label: component.label,
          parameters,
        }
      })
      const spec: CompositeNodeSpec = { type: 'composite', connection_mode: 'sequential', components }
      return spec
    }
    if (kind === 'repository') {
      const value = this.repositoryForm.getRawValue()
      const spec: RepositoryNodeSpec = {
        type: 'repository',
        provider: 'huggingface',
        repository_id: value.repositoryId.trim(),
        revision: value.revision.trim(),
        artifact_type: value.artifactType,
        entrypoint: value.entrypoint.trim() || null,
      }
      return spec
    }
    throw new Error('Choose a custom node type.')
  }

  private async refreshCustoms(): Promise<void> {
    this.customs.set(await this.customApi.list())
    if (!this.selectedKey()) this.selectedKey.set(this.allItems()[0]?.key ?? null)
  }

  wizardTitle(): string {
    if (this.wizardStep() === 1) return 'Choose a node type'
    if (this.wizardStep() === 2) return this.editingId() ? 'Edit node identity' : 'Describe the custom node'
    return this.selectedKind() === 'mathematical' ? 'Define the operation' : this.selectedKind() === 'composite' ? 'Build the composite' : 'Reference a public artifact'
  }

  kindLabel(kind: CustomNodeKind): string {
    return kind === 'mathematical' ? 'Mathematical operation' : kind === 'composite' ? 'Composite node' : 'Public repository'
  }

  implementationNames(manifest: NodeManifest): string[] {
    return Object.entries(manifest.implementations).filter(([, mapping]) => mapping.supported).map(([name]) => name)
  }

  displayNameFor(definitionId: string): string {
    return this.builtins().find((item) => item.id === definitionId)?.display_name ?? definitionId
  }

  valueLabel(value: unknown): string {
    if (value === undefined) return 'framework default'
    if (typeof value === 'string') return value
    try { return JSON.stringify(value) }
    catch { return String(value) }
  }

  objectEntryCount(value: Record<string, unknown>): number { return Object.keys(value).length }

  dateLabel(value: string): string { return new Date(value).toLocaleString() }
}
