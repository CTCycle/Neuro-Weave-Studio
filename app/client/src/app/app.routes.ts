import type { Routes } from '@angular/router'
import { builderResourcesResolver } from './core/navigation/builder-resources.resolver'
import { builderUnsavedGuard } from './core/navigation/builder-unsaved.guard'
import { AppShellComponent } from './layout/app-shell/app-shell.component'

export const routes: Routes = [
  {
    path: '',
    component: AppShellComponent,
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'models' },
      {
        path: 'models',
        loadComponent: () => import('./features/models/model-catalog-page.component').then((module) => module.ModelCatalogPageComponent),
      },
      {
        path: 'models/:modelId',
        loadComponent: () => import('./features/models/model-workspace-page.component').then((module) => module.ModelWorkspacePageComponent),
        children: [
          { path: '', pathMatch: 'full', loadComponent: () => import('./features/models/model-overview-page.component').then((module) => module.ModelOverviewPageComponent) },
          { path: 'compilation', loadComponent: () => import('./features/models/model-compilation-page.component').then((module) => module.ModelCompilationPageComponent) },
          { path: 'export', loadComponent: () => import('./features/models/model-export-page.component').then((module) => module.ModelExportPageComponent) },
          { path: 'validation', loadComponent: () => import('./features/models/model-validation-page.component').then((module) => module.ModelValidationPageComponent) },
        ],
      },
      {
        path: 'builder',
        canDeactivate: [builderUnsavedGuard],
        resolve: { resources: builderResourcesResolver },
        loadComponent: () => import('./features/builder/model-builder-page.component').then((module) => module.ModelBuilderPageComponent),
      },
      {
        path: 'builder/:modelId',
        canDeactivate: [builderUnsavedGuard],
        resolve: { resources: builderResourcesResolver },
        loadComponent: () => import('./features/builder/model-builder-page.component').then((module) => module.ModelBuilderPageComponent),
      },
      {
        path: 'nodes',
        loadComponent: () => import('./features/nodes/nodes-page.component').then((module) => module.NodesPageComponent),
      },
      {
        path: 'datasets',
        loadComponent: () => import('./features/datasets/datasets-page.component').then((module) => module.DatasetsPageComponent),
      },
      {
        path: 'datasets/:datasetId',
        loadComponent: () => import('./features/datasets/dataset-workspace-page.component').then((module) => module.DatasetWorkspacePageComponent),
        children: [
          { path: '', pathMatch: 'full', loadComponent: () => import('./features/datasets/dataset-overview-page.component').then((module) => module.DatasetOverviewPageComponent) },
          { path: 'structure', loadComponent: () => import('./features/datasets/dataset-structure-page.component').then((module) => module.DatasetStructurePageComponent) },
          { path: 'processing', data: { mode: 'processing' }, loadComponent: () => import('./features/datasets/dataset-future-page.component').then((module) => module.DatasetFuturePageComponent) },
          { path: 'split', data: { mode: 'split' }, loadComponent: () => import('./features/datasets/dataset-future-page.component').then((module) => module.DatasetFuturePageComponent) },
        ],
      },
      {
        path: 'training',
        loadComponent: () => import('./features/training/training-page.component').then((module) => module.TrainingPageComponent),
      },
      {
        path: 'environment',
        loadComponent: () => import('./features/environment/environment-page.component').then((module) => module.EnvironmentPageComponent),
      },
      {
        path: 'metrics',
        loadComponent: () => import('./features/metrics/metrics-page.component').then((module) => module.MetricsPageComponent),
      },
      {
        path: 'settings',
        loadComponent: () => import('./features/settings/settings-page.component').then((module) => module.SettingsPageComponent),
      },
      { path: '**', redirectTo: 'models' },
    ],
  },
]
