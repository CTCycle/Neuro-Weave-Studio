import { Injectable, signal } from '@angular/core'

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
  private readonly trainingState = signal<TrainingDraft[]>([])
  private readonly definitionState = signal<CustomMetricDefinition[]>([])

  readonly trainings = this.trainingState.asReadonly()
  readonly customDefinitions = this.definitionState.asReadonly()

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
