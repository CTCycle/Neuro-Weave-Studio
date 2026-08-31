import { Component, effect, inject, input } from '@angular/core'
import type { ProjectDocument } from '../../domain/types'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { GraphEditorStore } from '../../features/builder/state/graph-editor.store'
import { GraphCanvasComponent } from '../../features/builder/components/graph-canvas.component'

@Component({
  selector: 'nw-read-only-graph-preview',
  standalone: true,
  imports: [GraphCanvasComponent],
  providers: [GraphEditorStore],
  template: `<div class="readonly-graph-preview"><nw-graph-canvas [readOnly]="true" [modelName]="project().name"></nw-graph-canvas></div>`,
})
export class ReadOnlyGraphPreviewComponent {
  readonly project = input.required<ProjectDocument>()
  private readonly resources = inject(AppResourcesService)
  private readonly store = inject(GraphEditorStore)

  constructor() {
    effect(() => {
      const manifests = this.resources.manifests()
      if (!manifests.length) return
      this.store.setManifests(manifests)
      this.store.replaceGraph(this.project())
    })
  }
}
