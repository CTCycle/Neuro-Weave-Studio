import { createBrowserRouter, Navigate, RouterProvider } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { ModelBuilderPage } from './pages/ModelBuilderPage'
import { ModelCatalogPage } from './pages/ModelCatalogPage'
import { ModelCompilationPage } from './pages/ModelCompilationPage'
import { ModelInferencePage } from './pages/ModelInferencePage'
import { ModelOverviewPage, ModelWorkspacePage } from './pages/ModelWorkspacePage'
import { EnvironmentPage } from './pages/EnvironmentPage'
import { MetricsPage } from './pages/MetricsPage'
import { AppResourcesProvider } from './state/appContext'

const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/models" replace /> },
      { path: 'models', element: <ModelCatalogPage /> },
      {
        path: 'models/:modelId',
        element: <ModelWorkspacePage />,
        children: [
          { index: true, element: <ModelOverviewPage /> },
          { path: 'compilation', element: <ModelCompilationPage /> },
          { path: 'inference', element: <ModelInferencePage /> },
        ],
      },
      { path: 'builder', element: <ModelBuilderPage /> },
      { path: 'builder/:modelId', element: <ModelBuilderPage /> },
      { path: 'environment', element: <EnvironmentPage /> },
      { path: 'metrics', element: <MetricsPage /> },
      { path: '*', element: <Navigate to="/models" replace /> },
    ],
  },
])

export default function App() {
  return <AppResourcesProvider><RouterProvider router={router} /></AppResourcesProvider>
}
