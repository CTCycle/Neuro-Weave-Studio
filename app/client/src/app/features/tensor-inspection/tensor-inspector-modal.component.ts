import { DecimalPipe } from '@angular/common'
import { Component, inject, signal } from '@angular/core'
import { LucideChevronLeft, LucideChevronRight, LucideX } from '@lucide/angular'
import { ModalGuardDirective } from '../../shared/components/modal-guard.directive'
import { TensorInspectionCoordinator } from './tensor-inspection.coordinator'
import type { TensorDescriptor, TensorInspectionNode } from '../../domain/types'

@Component({
  selector: 'nw-tensor-inspector-modal',
  standalone: true,
  imports: [ModalGuardDirective, DecimalPipe, LucideX, LucideChevronLeft, LucideChevronRight],
  template: `
    @if (inspection.open()) {
      <div class="dialog-backdrop tensor-inspection-backdrop" nwModalGuard (click)="$event.stopPropagation()">
        <section class="tensor-inspection-modal panel" role="dialog" aria-modal="true" aria-labelledby="tensor-inspector-title" (click)="$event.stopPropagation()">
          <header class="tensor-inspector-header">
            <div><span class="eyebrow">TENSOR INSPECTION</span><h2 id="tensor-inspector-title">{{ inspection.node()?.node_name || 'Preparing inspection…' }}</h2><p>{{ inspection.node()?.node_type || 'Instrumented forward pass' }}</p></div>
            <button type="button" class="icon-button" aria-label="Close tensor inspection" title="Close" (click)="close()"><svg lucideX [size]="16"></svg></button>
          </header>
          @if (inspection.session(); as session) {
            <div class="tensor-inspector-toolbar">
              <label>Find node<input class="tensor-node-search" type="search" placeholder="Search nodes" aria-label="Search inspectable nodes" [value]="nodeSearch()" (input)="nodeSearch.set($any($event.target).value)"></label>
              <label>Execution order<select [value]="inspection.selectedNodeId()" (change)="selectNode($event)">@for (node of visibleNodes(session.nodes); track node.node_id) { <option [value]="node.node_id" [selected]="node.node_id === inspection.selectedNodeId()" [disabled]="!node.inspectable">{{ node.topological_index + 1 }} · {{ node.node_name }} · depth {{ node.depth }}{{ node.inspectable ? '' : ' · unavailable' }}</option> }</select></label>
              <div class="tensor-inspector-nav"><button type="button" class="secondary" (click)="inspection.previousNode()" [disabled]="!hasPrevious()" aria-label="Previous node"><svg lucideChevronLeft [size]="14"></svg></button><button type="button" class="secondary" (click)="inspection.nextNode()" [disabled]="!hasNext()" aria-label="Next node"><svg lucideChevronRight [size]="14"></svg></button></div>
              <span class="muted">Seed {{ session.seed }} · {{ session.runtime_api }} / {{ session.execution_backend }} · {{ session.compiler_mode }} · {{ branchContext(inspection.node()) }}</span>
            </div>
            <div class="tensor-inspection-warning" role="note">Synthetic input · untrained initialized weights · not trained activations</div>
          }
          @if (inspection.busy()) { <div class="tensor-inspection-loading"><span class="eyebrow">RUNNING</span><strong>Capturing one forward pass…</strong><span class="muted">Intermediate tensors are retained only for this inspection session.</span></div> }
          @if (inspection.error(); as error) { <div class="tensor-inspection-error" role="alert">{{ error }}</div> }
          @if (inspection.node(); as node) {
            <div class="tensor-inspection-body">
              <aside class="tensor-inspection-summary">
                <div class="tensor-inspection-ports"><span class="eyebrow">TENSORS</span><label>Tensor<select [value]="inspection.selectedTensorId()" (change)="selectTensor($event)">@for (tensor of tensors(node); track tensor.id) { <option [value]="tensor.id">{{ tensor.direction }} · {{ tensor.port_id }}{{ tensor.item_index === null || tensor.item_index === undefined ? '' : ' [' + tensor.item_index + ']' }}</option> }</select></label></div>
                @if (selectedTensor(node); as tensor) {
                  <dl class="tensor-detail-list"><div><dt>Shape</dt><dd>{{ shape(tensor) }}</dd></div><div><dt>Datatype</dt><dd>{{ tensor.dtype }}</dd></div><div><dt>Elements</dt><dd>{{ tensor.element_count | number }}</dd></div><div><dt>Stored values</dt><dd>{{ tensor.storage_mode }} · {{ tensor.value_count | number }}</dd></div></dl>
                  @if (tensor.shape.length >= 2 && tensor.storage_mode === 'full') { <div class="tensor-axis-controls"><span class="eyebrow">DISPLAY AXES</span>@for (slot of displayAxisSlots(tensor); track slot) { <label>Axis {{ slot + 1 }}<select [value]="inspection.displayAxes()[slot]" (change)="setDisplayAxis(slot, $event)">@for (axis of axisOptions(tensor); track axis) { <option [value]="axis" [selected]="axis === inspection.displayAxes()[slot]">{{ axis }}</option> }</select></label> } @if (fixedAxes(tensor).length) { <span class="eyebrow">FIXED AXES</span>@for (axis of fixedAxes(tensor); track axis) { <label>Axis {{ axis }}<input type="number" min="0" [max]="(tensor.shape[axis] || 1) - 1" [value]="inspection.fixedIndices()[axis] || 0" (change)="setAxis(axis, $event)"></label> } } </div> } @if (tensor.storage_mode !== 'full') { <p class="tensor-sampling-note">Values are a deterministic bounded sample; exact axis slicing is unavailable.</p> }
                }
                @if (node.inputs.length || node.outputs.length) { <dl class="tensor-port-shapes">@if (node.inputs.length) { <div><dt>Input shape{{ node.inputs.length > 1 ? 's' : '' }}</dt><dd>{{ portShapes(node.inputs) }}</dd></div> } @if (node.outputs.length) { <div><dt>Output shape{{ node.outputs.length > 1 ? 's' : '' }}</dt><dd>{{ portShapes(node.outputs) }}</dd></div> }</dl> }
              </aside>
              <main class="tensor-inspection-main">
                <nav class="tensor-view-tabs" aria-label="Tensor views">@for (view of views(node); track view) { <button type="button" [class.active]="inspection.view() === view" (click)="inspection.setView(view)">{{ label(view) }}</button> }</nav>
                @if (selectedTensor(node); as tensor) {
                  @if (inspection.view() === 'statistics') { <div class="tensor-stat-grid">@if (tensor.statistics; as stats) { <div><span>Minimum</span><strong>{{ format(stats.minimum) }}</strong></div><div><span>Maximum</span><strong>{{ format(stats.maximum) }}</strong></div><div><span>Mean</span><strong>{{ format(stats.mean) }}</strong></div><div><span>Std dev</span><strong>{{ format(stats.standard_deviation) }}</strong></div><div><span>Median</span><strong>{{ format(stats.median) }}</strong></div><div><span>Zeros</span><strong>{{ formatPercent(stats.zero_proportion) }}</strong></div><div><span>NaN</span><strong>{{ stats.nan_count }}</strong></div><div><span>Infinite</span><strong>{{ stats.infinite_count }}</strong></div><small class="muted">Statistics scope: {{ stats.mode }}</small> } @else { <p class="muted">Statistics are unavailable for this tensor.</p> }</div> }
                  @if (inspection.view() === 'histogram') { <div class="tensor-histogram">@if (tensor.histogram; as histogram) { @for (count of histogram.counts; track $index) { <span [style.height.%]="barHeight(count, histogram.counts)"></span> }<small class="muted">{{ histogram.mode }} distribution · {{ histogram.counts.length }} bins</small> } @else { <p class="muted">Histogram unavailable for this datatype.</p> }</div> }
                @if (inspection.view() === 'values' || inspection.view() === 'plot' || inspection.view() === 'heatmap' || inspection.view() === 'scatter') { @if (inspection.values(); as result) { @if (inspection.view() === 'values') { <div class="tensor-values"><div class="tensor-value-meta">{{ result.mode }} view · values {{ inspection.valuesOffset() + 1 | number }}–{{ inspection.valuesOffset() + flatValueCount(result.values) | number }} of {{ result.total_values | number }} in selected slice</div><pre>{{ formatValues(result.values) }}</pre><div class="tensor-value-pagination"><button type="button" class="secondary" (click)="inspection.previousValuesPage()" [disabled]="!inspection.valuesHasPreviousPage()">Previous</button><button type="button" class="secondary" (click)="inspection.nextValuesPage()" [disabled]="!inspection.valuesHasNextPage()">Next</button></div></div> } @if (inspection.view() === 'plot') { <svg class="tensor-line-plot" viewBox="0 0 640 220" role="img" aria-label="Tensor line plot"><polyline [attr.points]="linePoints(result.values)" fill="none" stroke="currentColor" stroke-width="2"></polyline></svg><small class="muted">{{ result.mode }} line view · bounded to 2,048 points</small> } @if (inspection.view() === 'heatmap') { <div class="tensor-heatmap" [style.grid-template-columns]="heatmapColumns(result.view_shape)">@for (value of flatValues(result.values); track $index) { <span [style.background]="heatColor(value, tensor)" [title]="format(value)"></span> }</div><small class="muted">{{ result.mode }} heatmap view · bounded to 64 × 64 cells</small> } @if (inspection.view() === 'scatter') { <div class="tensor-scatter"><svg class="tensor-scatter-plot" viewBox="0 0 640 300" role="img" aria-label="Tensor scatter plot"><line x1="34" y1="12" x2="34" y2="270" stroke="currentColor" opacity=".45"></line><line x1="34" y1="270" x2="628" y2="270" stroke="currentColor" opacity=".45"></line>@for (point of scatterPoints(result.values); track $index) { <circle [attr.cx]="point.x" [attr.cy]="point.y" r="3" fill="currentColor" opacity=".8"></circle> }</svg><small class="muted">{{ result.mode }} [N, 2] view · each point is one row in the displayed subset</small></div> } } @else { <div class="tensor-values-loading">Loading bounded values…</div> } }
                }
              </main>
            </div>
          }
        </section>
      </div>
    }
  `,
})
export class TensorInspectorModalComponent {
  readonly inspection = inject(TensorInspectionCoordinator)
  readonly nodeSearch = signal('')

  async close(): Promise<void> { await this.inspection.close() }
  selectNode(event: Event): void { const value = (event.target as HTMLSelectElement).value; void this.inspection.selectNode(value) }
  selectTensor(event: Event): void { this.inspection.selectTensor((event.target as HTMLSelectElement).value) }
  setAxis(axis: number, event: Event): void { this.inspection.setFixedIndex(axis, Number((event.target as HTMLInputElement).value || 0)) }
  setDisplayAxis(slot: number, event: Event): void { this.inspection.setDisplayAxis(slot, Number((event.target as HTMLSelectElement).value)) }
  tensors(node: { inputs: TensorDescriptor[]; outputs: TensorDescriptor[] }): TensorDescriptor[] { return [...node.outputs, ...node.inputs] }
  visibleNodes(nodes: TensorInspectionNode[]): TensorInspectionNode[] {
    const query = this.nodeSearch().trim().toLocaleLowerCase()
    if (!query) return nodes
    return nodes.filter((node) => `${node.node_name} ${node.node_type} ${node.node_id}`.toLocaleLowerCase().includes(query))
  }
  branchContext(node: { predecessors?: string[]; successors?: string[] } | null): string { return node ? `${node.predecessors?.length ?? 0} in · ${node.successors?.length ?? 0} out` : 'graph context' }
  selectedTensor(node: { inputs: TensorDescriptor[]; outputs: TensorDescriptor[] }): TensorDescriptor | null { return this.tensors(node).find((tensor) => tensor.id === this.inspection.selectedTensorId()) ?? this.tensors(node)[0] ?? null }
  shape(tensor: TensorDescriptor): string { return `[${tensor.shape.join(', ')}]` }
  portShapes(tensors: TensorDescriptor[]): string { return tensors.map((tensor) => `${tensor.port_id}: ${this.shape(tensor)}`).join(' · ') }
  axisOptions(tensor: TensorDescriptor): number[] { return tensor.shape.map((_, index) => index) }
  displayAxisSlots(tensor: TensorDescriptor): number[] { return tensor.shape.length >= 2 ? [0, 1] : [] }
  fixedAxes(tensor: TensorDescriptor): number[] { const display = new Set(this.inspection.displayAxes()); return tensor.shape.map((_, index) => index).filter((index) => !display.has(index)) }
  views(node: { inputs: TensorDescriptor[]; outputs: TensorDescriptor[] }): Array<'values' | 'statistics' | 'histogram' | 'plot' | 'heatmap' | 'scatter'> {
    const tensor = this.selectedTensor(node)
    if (!tensor) return ['values']
    const result: Array<'values' | 'statistics' | 'histogram' | 'plot' | 'heatmap' | 'scatter'> = ['values', 'statistics', 'histogram']
    if (tensor.storage_mode !== 'summary_only' && tensor.shape.length === 1) result.push('plot')
    if (tensor.storage_mode === 'full' && tensor.shape.length >= 2) result.push('heatmap')
    if (tensor.storage_mode === 'full' && tensor.shape.length === 2 && tensor.shape[1] === 2) result.push('scatter')
    return result
  }
  label(view: string): string { return view === 'values' ? 'Values' : view === 'statistics' ? 'Statistics' : view === 'histogram' ? 'Histogram' : view === 'heatmap' ? 'Heatmap' : view === 'plot' ? 'Plot' : 'Scatter' }
  hasPrevious(): boolean { const nodes = (this.inspection.session()?.nodes ?? []).filter((item) => item.inspectable); const index = nodes.findIndex((item) => item.node_id === this.inspection.selectedNodeId()); return index > 0 }
  hasNext(): boolean { const nodes = (this.inspection.session()?.nodes ?? []).filter((item) => item.inspectable); const index = nodes.findIndex((item) => item.node_id === this.inspection.selectedNodeId()); return index >= 0 && index < nodes.length - 1 }
  format(value: number | null | undefined): string { return value === null || value === undefined ? '—' : Number.isFinite(value) ? value.toPrecision(6) : String(value) }
  formatPercent(value: number | null | undefined): string { return value === null || value === undefined ? '—' : `${(value * 100).toFixed(2)}%` }
  formatValues(value: unknown): string { return JSON.stringify(value, null, 2) }
  flatValueCount(value: unknown): number { return Array.isArray(value) ? value.reduce((total, item) => total + this.flatValueCount(item), 0) : value === undefined ? 0 : 1 }
  flatValues(value: unknown): number[] { return Array.isArray(value) ? value.flat(Infinity).filter((item): item is number => typeof item === 'number') : typeof value === 'number' ? [value] : [] }
  heatmapColumns(shape: number[]): string { const columns = shape.length > 1 ? shape[1] : shape[0] ?? 1; return `repeat(${Math.max(1, Math.min(columns, 64))}, minmax(2px, 1fr))` }
  scatterPoints(value: unknown): Array<{ x: number; y: number }> {
    const values = this.flatValues(value).filter(Number.isFinite)
    const points: Array<{ x: number; y: number }> = []
    for (let index = 0; index + 1 < values.length; index += 2) points.push({ x: values[index], y: values[index + 1] })
    if (!points.length) return []
    const minX = Math.min(...points.map((point) => point.x)); const maxX = Math.max(...points.map((point) => point.x))
    const minY = Math.min(...points.map((point) => point.y)); const maxY = Math.max(...points.map((point) => point.y))
    const width = Math.max(maxX - minX, 1e-9); const height = Math.max(maxY - minY, 1e-9)
    return points.map((point) => ({ x: 34 + ((point.x - minX) / width) * 594, y: 270 - ((point.y - minY) / height) * 258 }))
  }
  linePoints(value: unknown): string { const values = this.flatValues(value); const max = Math.max(...values.filter(Number.isFinite).map((item) => Math.abs(item)), 1); return values.map((item, index) => `${(index / Math.max(values.length - 1, 1)) * 640},${110 - (Number.isFinite(item) ? item / max : 0) * 100}`).join(' ') }
  barHeight(value: number, values: number[]): number { const max = Math.max(...values, 1); return Math.max(2, (value / max) * 100) }
  heatColor(value: number, tensor: TensorDescriptor): string { if (!Number.isFinite(value)) return 'hsl(195 12% 34%)'; const min = tensor.statistics?.minimum ?? -1; const max = tensor.statistics?.maximum ?? 1; const ratio = Math.max(0, Math.min(1, (value - min) / Math.max(max - min, 1e-9))); return `hsl(${210 - ratio * 180} 70% ${28 + ratio * 35}%)` }
}
