import { Injectable, computed, signal } from '@angular/core'

export type DatasetKind = 'tabular' | 'arrays' | 'images' | 'paired' | 'documents' | 'audio' | 'timeseries' | 'custom'
export type DatasetSplitMode = 'none' | 'smart' | 'custom'

export interface DatasetSplit {
  mode: DatasetSplitMode
  train: number
  validation: number
  test: number
}

export interface DatasetReference {
  id: string
  name: string
  kind: DatasetKind
  primaryPath: string
  referencePath: string | null
  options: Record<string, string | number | boolean>
  previewEntries: string[]
  preprocessing: string[]
  split: DatasetSplit
  createdAt: string
}

export interface TrainingDraft {
  id: string
  name: string
  modelId: string
  modelName: string
  datasetId: string
  datasetName: string
  epochs: number
  batchSize: number
  status: 'draft'
  createdAt: string
}

export interface CustomMetricDefinition {
  id: string
  kind: 'metric' | 'loss'
  label: string
  description: string
  formula: string
  createdAt: string
}

@Injectable({ providedIn: 'root' })
export class ExperimentDraftsService {
  private readonly datasetState = signal<DatasetReference[]>([])
  private readonly trainingState = signal<TrainingDraft[]>([])
  private readonly definitionState = signal<CustomMetricDefinition[]>([])

  readonly datasets = this.datasetState.asReadonly()
  readonly trainings = this.trainingState.asReadonly()
  readonly customDefinitions = this.definitionState.asReadonly()
  readonly hasDatasets = computed(() => this.datasetState().length > 0)

  addDataset(dataset: DatasetReference): void {
    this.datasetState.update((items) => [dataset, ...items])
  }

  updateDataset(id: string, updater: (dataset: DatasetReference) => DatasetReference): void {
    this.datasetState.update((items) => items.map((item) => item.id === id ? updater(item) : item))
  }

  removeDataset(id: string): void {
    this.datasetState.update((items) => items.filter((item) => item.id !== id))
    this.trainingState.update((items) => items.filter((item) => item.datasetId !== id))
  }

  addTraining(training: TrainingDraft): void {
    this.trainingState.update((items) => [training, ...items])
  }

  removeTraining(id: string): void {
    this.trainingState.update((items) => items.filter((item) => item.id !== id))
  }

  addCustomDefinition(definition: CustomMetricDefinition): void {
    this.definitionState.update((items) => [definition, ...items])
  }
}
