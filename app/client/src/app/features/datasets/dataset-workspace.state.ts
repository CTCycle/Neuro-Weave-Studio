// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { Injectable, inject, signal } from '@angular/core'
import { ApiService } from '../../core/api/api.service'
import { AppResourcesService } from '../../core/state/app-resources.service'
import type { DatasetDefinitionMutation, DatasetDocument } from '../../domain/dataset.types'

@Injectable()
export class DatasetWorkspaceState {
  private readonly api = inject(ApiService)
  private readonly resources = inject(AppResourcesService)
  readonly dataset = signal<DatasetDocument | null>(null)
  readonly loading = signal(true)
  readonly error = signal<string | null>(null)

  async load(datasetId: string): Promise<void> {
    this.loading.set(true)
    this.error.set(null)
    try {
      this.dataset.set(await this.api.getDataset(datasetId))
    } catch (reason) {
      this.dataset.set(null)
      this.error.set(reason instanceof Error ? reason.message : String(reason))
    } finally {
      this.loading.set(false)
    }
  }

  async refresh(): Promise<DatasetDocument> {
    const current = this.dataset()
    if (!current) throw new Error('Dataset is not loaded')
    const next = await this.api.getDataset(current.id)
    this.dataset.set(next)
    await this.resources.refreshDatasets()
    return next
  }

  async updateDefinition(payload: DatasetDefinitionMutation): Promise<DatasetDocument> {
    const current = this.dataset()
    if (!current) throw new Error('Dataset is not loaded')
    const next = await this.api.updateDataset(current.id, payload)
    this.dataset.set(next)
    await this.resources.refreshDatasets()
    return next
  }

  async revalidate(): Promise<DatasetDocument> {
    const current = this.dataset()
    if (!current) throw new Error('Dataset is not loaded')
    const next = await this.api.revalidateDataset(current.id)
    this.dataset.set(next)
    await this.resources.refreshDatasets()
    return next
  }

  async remove(): Promise<void> {
    const current = this.dataset()
    if (!current) throw new Error('Dataset is not loaded')
    await this.api.deleteDataset(current.id)
    await this.resources.refreshDatasets()
  }
}
