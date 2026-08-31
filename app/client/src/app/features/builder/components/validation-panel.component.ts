import { Component, inject, input, output } from '@angular/core'
import { LucideCheckCircle2, LucideChevronDown, LucideXCircle } from '@lucide/angular'
import type { ValidationDiagnostic, ValidationResult } from '../../../domain/types'
import { GraphEditorStore } from '../state/graph-editor.store'

@Component({
  selector: 'nw-validation-panel',
  standalone: true,
  imports: [LucideCheckCircle2, LucideChevronDown, LucideXCircle],
  template: `
    <section class="validation-panel panel" [class.has-result]="result()" [class.idle]="!result()">
      <button type="button" class="validation-summary" [attr.aria-expanded]="expanded()" (click)="toggle()"><span><span class="eyebrow">GRAPH HEALTH</span><strong>Validation</strong>@if (result(); as value) { <small>{{ value.diagnostics.length ? value.diagnostics.length + ' finding' + (value.diagnostics.length === 1 ? '' : 's') : 'No findings' }}</small> } @else { <small>Not run</small> }</span><span class="validation-summary-status">@if (result(); as value) { <span class="status-indicator" [class.valid]="value.valid" [class.invalid]="!value.valid">@if (value.valid) { <svg lucideCheckCircle2 [size]="15"></svg>Ready } @else { <svg lucideXCircle [size]="15"></svg>Blocked }</span> }<svg lucideChevronDown [class.expanded]="expanded()" [size]="16"></svg></span></button>
      @if (!result()) { <p class="muted">Run validation to inspect graph, parameters, shapes, source generation, runtime, and backend compatibility.</p> }
      @if (result() && expanded()) { <div class="validation-body"><div class="model-stats"><div><span>Nodes</span><strong>{{ result()?.shape_summaries?.length ?? 0 }}</strong></div><div><span>Parameters</span><strong>{{ result()?.total_parameters?.toLocaleString() ?? 'Unknown' }}</strong></div><div><span>Diagnostics</span><strong>{{ result()?.diagnostics?.length ?? 0 }}</strong></div></div><div class="diagnostic-list">@if (!result()?.diagnostics?.length) { <div class="diagnostic info"><strong>info</strong><span>No validation findings.</span></div> } @for (diagnostic of result()?.diagnostics ?? []; track $index) { <button type="button" class="diagnostic" [class.error]="diagnostic.severity === 'error'" [class.warning]="diagnostic.severity === 'warning'" (click)="selectDiagnostic(diagnostic)"><strong>{{ diagnostic.severity }}</strong><span>{{ diagnostic.message }}<small>{{ diagnosticContext(diagnostic) }}</small></span></button> }</div></div> }
    </section>
  `,
})
export class ValidationPanelComponent {
  private readonly store = inject(GraphEditorStore)
  readonly result = input<ValidationResult | null>(null)
  readonly expanded = input(false)
  readonly expandedChange = output<boolean>()

  toggle(): void { if (this.result()) this.expandedChange.emit(!this.expanded()) }
  selectDiagnostic(diagnostic: ValidationDiagnostic): void { if (diagnostic.node_id) this.store.selectNode(diagnostic.node_id); else if (diagnostic.edge_id) this.store.selectEdge(diagnostic.edge_id) }
  diagnosticContext(diagnostic: ValidationDiagnostic): string { return diagnostic.node_id ? `Node ${diagnostic.node_id}` : diagnostic.edge_id ? `Connection ${diagnostic.edge_id}` : diagnostic.parameter_id ? `Parameter ${diagnostic.parameter_id}` : 'Graph-level finding' }
}
