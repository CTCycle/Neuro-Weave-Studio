import { Injectable, inject, signal } from '@angular/core'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import type { ProjectDocument } from '../../domain/types'

@Injectable()
export class ModelWorkspaceState {
  private readonly api = inject(ApiService)
  private readonly resources = inject(AppResourcesService)
  readonly project = signal<ProjectDocument | null>(null)
  readonly loading = signal(true)
  readonly error = signal<string | null>(null)

  async load(modelId: string): Promise<void> {
    this.loading.set(true)
    this.error.set(null)
    try {
      this.project.set(await this.api.getProject(modelId))
    } catch (reason) {
      this.project.set(null)
      this.error.set(reason instanceof Error ? reason.message : String(reason))
    } finally {
      this.loading.set(false)
    }
  }

  async refresh(): Promise<ProjectDocument> {
    const current = this.project()
    if (!current) throw new Error('Model is not loaded')
    const next = await this.api.getProject(current.id)
    this.project.set(next)
    await this.resources.refreshProjects()
    return next
  }

  async duplicate(): Promise<ProjectDocument> {
    const current = this.project()
    if (!current) throw new Error('Model is not loaded')
    const next = await this.api.duplicateProject(current.id, `${current.name} copy`)
    await this.resources.refreshProjects()
    return next
  }

  async remove(): Promise<void> {
    const current = this.project()
    if (!current) throw new Error('Model is not loaded')
    await this.api.deleteProject(current.id)
    await this.resources.refreshProjects()
  }
}
