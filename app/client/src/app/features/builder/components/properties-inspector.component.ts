import { Component, inject } from '@angular/core'
import { GraphEditorStore } from '../state/graph-editor.store'

@Component({
  selector: 'nw-properties-inspector',
  standalone: true,
  template: `
    <aside class="builder-inspector panel" aria-label="Properties inspector">
      <div class="inspector-heading"><div><span class="eyebrow">INSPECTOR</span><h2>Properties</h2></div></div>
      @if (selectedNode(); as node) { <div class="inspector-node-heading"><span class="inspector-node-icon" [style.--node-accent]="node.data.manifest.ui.accent">{{ node.data.manifest.ui.icon || '·' }}</span><div><strong>{{ node.data.label }}</strong><small>{{ node.data.manifest.id }}</small></div></div><label class="field-label">Label<input [value]="node.data.label" (input)="updateLabel(node.id, $any($event.target).value)" /></label><p class="inspector-description">{{ node.data.manifest.description }}</p><div class="inspector-section"><span class="eyebrow">PORTS</span>@for (port of node.data.manifest.inputs; track port.id) { <div class="inspector-port"><span>Input</span><strong>{{ port.display_name }}</strong><small>{{ port.kind }}{{ port.required ? ' · required' : '' }}</small></div> }@for (port of node.data.manifest.outputs; track port.id) { <div class="inspector-port"><span>Output</span><strong>{{ port.display_name }}</strong><small>{{ port.kind }}</small></div> }</div><button class="secondary danger-button" type="button" (click)="deleteNode()">Delete node</button> } @else { <div class="inspector-empty"><span>Select an operation</span><small>Node parameters and ports will appear here.</small></div> }
    </aside>
  `,
})
export class PropertiesInspectorComponent {
  readonly store = inject(GraphEditorStore)
  selectedNode = () => this.store.nodes().find((node) => node.id === this.store.selectedNodeId()) ?? null
  updateLabel(nodeId: string, label: string): void { this.store.updateNode(nodeId, { label }) }
  deleteNode(): void { this.store.deleteSelected() }
}
