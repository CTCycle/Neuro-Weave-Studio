import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ModelConfigurationPanel } from './ModelConfigurationPanel'
import { useGraphStore } from '../state/graphStore'
import { useAppResources } from '../state/useAppResources'

vi.mock('../state/useAppResources', () => ({ useAppResources: vi.fn() }))

const catalog = {
  optimizers: [
    { id: 'adam', label: 'Adam', description: 'Adaptive optimizer.', parameters: [{ id: 'learning_rate', label: 'Learning rate', type: 'number' as const, default: 0.001, description: 'Step size.', constraints: {} }] },
    { id: 'sgd', label: 'SGD', description: 'Gradient descent.', parameters: [{ id: 'learning_rate', label: 'Learning rate', type: 'number' as const, default: 0.01, description: 'Step size.', constraints: {} }] },
  ],
  losses: [{ id: 'categorical_crossentropy', label: 'Categorical crossentropy', description: 'Categorical loss.', parameters: [], formula: null }],
  metrics: [{ id: 'accuracy', label: 'Accuracy', description: 'Accuracy.', parameters: [], formula: null }],
  initializers: ['glorot_uniform'],
  regularizers: ['none'],
  constraints: ['none'],
}

describe('ModelConfigurationPanel', () => {
  afterEach(() => {
    cleanup()
    useGraphStore.getState().reset()
  })

  it('keeps the primary summary compact and applies modal edits once', async () => {
    vi.mocked(useAppResources).mockReturnValue({ compilationCatalog: catalog } as unknown as ReturnType<typeof useAppResources>)
    render(<ModelConfigurationPanel />)
    expect(screen.getByText('COMPILATION')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Configure' })).toBeEnabled()
    expect(screen.queryByText('Model defaults')).not.toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Configure' }))
    expect(screen.getAllByRole('tab')).toHaveLength(4)
    expect(screen.getByRole('tab', { name: /Advanced Model defaults/ })).toBeVisible()
    const advancedPanel = document.getElementById('compilation-tabpanel-advanced') as HTMLElement
    expect(advancedPanel).not.toBeVisible()
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Optimizer' }), 'sgd')
    await userEvent.click(screen.getByRole('tab', { name: /Advanced Model defaults/ }))
    expect(advancedPanel).toBeVisible()
    expect(advancedPanel).toHaveTextContent('Model defaults')
    expect(screen.getByRole('heading', { name: 'Configure Keras compilation' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Apply configuration' })).toBeVisible()
    await userEvent.click(screen.getByRole('button', { name: 'Apply configuration' }))
    expect(useGraphStore.getState().modelConfiguration.optimizer).toBe('sgd')
  })
})
