import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { MetricsPage } from './MetricsPage'
import { useAppResources } from '../state/useAppResources'
import type { AppResources } from '../state/appResourcesContext'

vi.mock('../state/useAppResources', () => ({ useAppResources: vi.fn() }))

const mockedUseAppResources = vi.mocked(useAppResources)

const resources = {
  loading: false,
  error: null,
  manifests: [],
  projects: [],
  backendStatuses: [],
  runtimeApis: [],
  runtimePreferences: {},
  runtimePreferenceBackend: null,
  setRuntimePreferenceBackend: vi.fn(),
  setRuntimePreference: vi.fn(),
  refreshProjects: vi.fn(),
  clearError: vi.fn(),
  compilationCapabilities: {
    keras: {
      optimizers: [],
      initializers: [],
      regularizers: [],
      constraints: [],
      metrics: [{ id: 'accuracy', label: 'Accuracy', description: 'Classification accuracy.', parameters: [] }],
    losses: [{ id: 'mse', label: 'Mean squared error', description: 'Squared prediction error.', parameters: [], formula: String.raw`\operatorname{MSE}=\frac{1}{n}\sum_i(y_i-\hat{y}_i)^2` }],
    },
    native: {},
  },
} as unknown as AppResources

describe('MetricsPage', () => {
  afterEach(cleanup)

  it('exposes the shared read-only metrics and loss catalog sections', () => {
    mockedUseAppResources.mockReturnValue(resources)
    const { container } = render(<MetricsPage />)

    expect(screen.getByRole('heading', { level: 1, name: 'Metrics' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Loss functions' })).toBeInTheDocument()
    expect(screen.getByText('Accuracy')).toBeInTheDocument()
    expect(screen.getByText('Mean squared error')).toBeInTheDocument()
    expect(screen.getAllByText('READ ONLY')).toHaveLength(2)
    expect(container.querySelector('.metrics-formula .katex')).toBeInTheDocument()
    expect(container.querySelector('.metrics-formula code')).not.toBeInTheDocument()
    expect(container.textContent).not.toContain('\\operatorname')
  })
})
