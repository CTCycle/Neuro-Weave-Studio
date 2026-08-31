import type { Routes } from '@angular/router'
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
          { path: 'inference', loadComponent: () => import('./features/models/model-inference-page.component').then((module) => module.ModelInferencePageComponent) },
        ],
      },
      {
        path: 'builder',
        canDeactivate: [builderUnsavedGuard],
        loadComponent: () => import('./features/builder/model-builder-page.component').then((module) => module.ModelBuilderPageComponent),
      },
      {
        path: 'builder/:modelId',
        canDeactivate: [builderUnsavedGuard],
        loadComponent: () => import('./features/builder/model-builder-page.component').then((module) => module.ModelBuilderPageComponent),
      },
      {
        path: 'environment',
        loadComponent: () => import('./features/environment/environment-page.component').then((module) => module.EnvironmentPageComponent),
      },
      {
        path: 'metrics',
        loadComponent: () => import('./features/metrics/metrics-page.component').then((module) => module.MetricsPageComponent),
      },
      { path: '**', redirectTo: 'models' },
    ],
  },
]
