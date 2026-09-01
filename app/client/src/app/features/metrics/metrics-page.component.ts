import { Component, inject, Input } from '@angular/core'
import { LucideActivity, LucideInfo } from '@lucide/angular'
import type { CompilationOption } from '../../domain/types'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { MathFormulaComponent } from '../../shared/components/math-formula.component'

@Component({
  selector: 'nw-metrics-section',
  standalone: true,
  imports: [MathFormulaComponent],
  template: `<section class="metrics-section panel"><div class="section-heading"><div><span class="eyebrow">{{ eyebrow }}</span><h2>{{ title }}</h2></div><span class="catalog-count">{{ items.length }}</span></div><div class="metrics-list">@for (item of items; track item.id) { <article class="metrics-card"><div class="metrics-card-heading"><div><strong>{{ item.label }}</strong><code>{{ item.id }}</code></div><span class="read-only-badge">READ ONLY</span></div><div class="metrics-card-body" [class.has-formula]="!!item.formula"><p>{{ item.description }}</p>@if (item.formula) { <div class="metrics-formula"><span>Formula</span><nw-math-formula [formula]="item.formula"></nw-math-formula></div> }</div></article> }</div></section>`,
})
export class MetricsSectionComponent {
  @Input() title = ''
  @Input() eyebrow = ''
  @Input() items: CompilationOption[] = []
}

@Component({
  selector: 'nw-metrics-page',
  standalone: true,
  imports: [LucideActivity, LucideInfo, MetricsSectionComponent],
  template: `
    @if (!resources.runtimeReady()) { <section class="page-state panel"><span class="eyebrow">METRICS CATALOG</span><h1>Loading catalog…</h1></section> } @else if (resources.compilationCapabilities(); as capabilities) { @let catalog = capabilities.keras; <div class="metrics-page"><header class="catalog-page-header"><div><span class="eyebrow">COMPILATION CATALOG</span><h1>Metrics</h1><p>Review the read-only metric and loss definitions available to model compilation.</p></div><div class="environment-summary"><svg lucideActivity [size]="16"></svg><strong>{{ catalog.metrics.length + catalog.losses.length }} definitions</strong></div></header><div class="metrics-note panel"><svg lucideInfo [size]="16"></svg><span>This catalog is intentionally read-only for now. It is the shared metadata boundary for the future editable metrics and loss catalog.</span></div><div class="metrics-grid"><nw-metrics-section title="Metrics" eyebrow="EVALUATION" [items]="catalog.metrics"></nw-metrics-section><nw-metrics-section title="Loss functions" eyebrow="OPTIMIZATION" [items]="catalog.losses"></nw-metrics-section></div></div> } @else { <section class="page-state panel"><span class="eyebrow">METRICS CATALOG</span><h1>Catalog unavailable</h1><p>The runtime metadata did not include the Keras metrics and loss catalog.</p></section> }
  `,
})
export class MetricsPageComponent {
  readonly resources = inject(AppResourcesService)
  constructor() { void this.resources.loadRuntimeMetadata() }
}
