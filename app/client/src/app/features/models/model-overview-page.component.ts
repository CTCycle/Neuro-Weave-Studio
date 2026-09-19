import { Component, inject } from '@angular/core'
import { LucideInfo } from '@lucide/angular'
import { implementationLabel } from '../../domain/types'
import { CompilationStatusBadgeComponent, ValidationStatusBadgeComponent } from '../../shared/components/model-status-badge.component'
import { ReadOnlyGraphPreviewComponent } from '../../shared/components/read-only-graph-preview.component'
import { ModelWorkspaceState } from './model-workspace.state'

@Component({
  selector: 'nw-model-overview-page',
  standalone: true,
  imports: [LucideInfo, ValidationStatusBadgeComponent, CompilationStatusBadgeComponent, ReadOnlyGraphPreviewComponent],
  template: `
    @if (state.project(); as project) {
      <div class="overview-page">
        <div class="overview-grid"><section class="overview-preview panel"><div class="section-heading"><div><span class="eyebrow">ARCHITECTURE PREVIEW</span><h2>Read-only graph</h2></div><span class="muted">Open Builder to edit</span></div><nw-read-only-graph-preview [project]="project"></nw-read-only-graph-preview></section>
          <aside class="overview-summary"><section class="summary-card workspace-summary-card"><div class="section-heading"><div><span class="eyebrow">MODEL SUMMARY</span><h2>Configuration</h2></div><svg lucideInfo [size]="16"></svg></div><dl class="detail-list"><div><dt>Execution backend</dt><dd>{{ project.execution_backend }}</dd></div><div><dt>Keras API layer</dt><dd>{{ project.keras_enabled ? 'Enabled' : 'Disabled' }}</dd></div>@if (project.keras_enabled) { <div><dt>Optimizer</dt><dd>{{ project.graph.model_configuration.keras.compilation.optimizer.id }}</dd></div><div><dt>Learning rate</dt><dd>{{ project.graph.model_configuration.keras.compilation.optimizer.parameters['learning_rate'] ?? '—' }}</dd></div><div><dt>Default loss</dt><dd>{{ project.graph.model_configuration.keras.compilation.losses['__default__']?.id || 'Not configured' }}</dd></div><div><dt>Metrics</dt><dd>{{ metricList(project) }}</dd></div><div><dt>Mixed precision</dt><dd>{{ project.graph.model_configuration.keras.mixed_precision ? 'Enabled' : 'Disabled' }}</dd></div> } @else { <div><dt>Native compiler</dt><dd>{{ project.execution_backend === 'tensorflow' ? 'tf.function / XLA' : project.execution_backend === 'pytorch' ? 'torch.compile' : 'jax.jit' }}</dd></div><div><dt>Compiler state</dt><dd>Available from builder</dd></div> }</dl></section><section class="summary-card workspace-summary-card"><div class="section-heading"><div><span class="eyebrow">CURRENT REVISION</span><h2>Readiness</h2></div></div><div class="readiness-stack"><div><span>Validation</span><nw-validation-status-badge [status]="project.validation_status"></nw-validation-status-badge></div><div><span>Source compilation</span><nw-compilation-status-badge [compiled]="!!project.generated_source" [target]="implementationLabel(project.execution_backend, project.keras_enabled)"></nw-compilation-status-badge></div><div><span>Generated source</span><strong>{{ project.generated_source ? 'Available' : 'Not generated' }}</strong></div></div></section></aside>
        </div>
      </div>
    }
  `,
})
export class ModelOverviewPageComponent {
  readonly state = inject(ModelWorkspaceState)
  implementationLabel = implementationLabel
  metricList(project: { graph: { model_configuration: { keras: { compilation: { metrics: Record<string, Array<{ id: string }>> } } } } }): string { return project.graph.model_configuration.keras.compilation.metrics['__default__']?.map((item) => item.id).join(', ') || 'None configured' }
}
