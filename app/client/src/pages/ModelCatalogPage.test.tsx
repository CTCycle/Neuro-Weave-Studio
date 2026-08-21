import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ProjectSummary } from '../domain/types'
import { useAppResources } from '../state/useAppResources'
import { ModelCatalogPage } from './ModelCatalogPage'

vi.mock('../state/useAppResources', () => ({ useAppResources: vi.fn() }))

const projects: ProjectSummary[] = [
  {
    id: 'compiled-model',
    name: 'Compiled classifier',
    description: 'Ready for source inspection',
    execution_backend: 'pytorch',
    keras_enabled: true,
    current_revision: 3,
    updated_at: '2026-08-07T09:00:00Z',
    validation_status: 'valid',
    node_count: 4,
    edge_count: 3,
    has_generated_source: true,
  },
  {
    id: 'draft-model',
    name: 'Draft architecture',
    description: 'Still being edited',
    execution_backend: 'pytorch',
    keras_enabled: false,
    current_revision: 1,
    updated_at: '2026-08-06T09:00:00Z',
    validation_status: 'invalid',
    node_count: 2,
    edge_count: 1,
    has_generated_source: false,
  },
]

describe('ModelCatalogPage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('shows enriched model summaries and filters compiled models', async () => {
    vi.mocked(useAppResources).mockReturnValue({
      projects,
      manifests: [],
      backendStatuses: [],
      runtimeApis: [],
      runtimeSelection: null,
      selectedExecutionBackend: null,
      loading: false,
      error: null,
      refreshProjects: vi.fn(),
      clearError: vi.fn(),
      setSelectedExecutionBackend: vi.fn(),
      setRuntimeSelection: vi.fn(),
    })

    render(<MemoryRouter><ModelCatalogPage /></MemoryRouter>)
    const compiledCard = screen.getByRole('link', { name: /Compiled classifier/ })
    expect(compiledCard).toHaveTextContent('4 nodes')
    expect(compiledCard).toHaveTextContent('3 connections')
    expect(screen.getAllByText('Compiled').length).toBeGreaterThan(0)

    await userEvent.click(screen.getByRole('button', { name: 'Compiled' }))
    expect(screen.getByText('Compiled classifier')).toBeInTheDocument()
    expect(screen.queryByText('Draft architecture')).not.toBeInTheDocument()
    expect(screen.getByText('1 of 2')).toBeInTheDocument()
  })
})
