import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import { ModelConfigurationPanel } from './ModelConfigurationPanel'
import { useGraphStore } from '../state/graphStore'

describe('ModelConfigurationPanel', () => {
  afterEach(() => {
    cleanup()
    useGraphStore.getState().reset()
  })

  it('keeps primary settings compact and advanced settings collapsed', async () => {
    render(<ModelConfigurationPanel />)
    expect(screen.getByRole('combobox', { name: 'Optimizer' })).toBeVisible()
    expect(screen.getByText('Advanced defaults')).toBeInTheDocument()
    expect(document.querySelector('details')).not.toHaveAttribute('open')

    await userEvent.click(screen.getByText('Advanced defaults'))
    expect(screen.getByRole('combobox', { name: 'Initializer' })).toBeVisible()
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Optimizer' }), 'sgd')
    expect(useGraphStore.getState().modelConfiguration.optimizer).toBe('sgd')
  })
})
