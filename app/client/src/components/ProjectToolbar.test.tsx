import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ProjectToolbar } from './ProjectToolbar'

describe('ProjectToolbar', () => {
  afterEach(cleanup)

  it('keeps Builder actions focused on authoring and persistence', () => {
    render(<ProjectToolbar name="Model" onNameChange={vi.fn()} description="" onDescriptionChange={vi.fn()} executionBackend="pytorch" onExecutionBackendChange={vi.fn()} kerasEnabled onKerasEnabledChange={vi.fn()} onSave={vi.fn()} onCompile={vi.fn()} onValidate={vi.fn()} busy={false} dirty={false} modelId="model-1" onOpenWorkspace={vi.fn()} />)
    expect(screen.getByRole('button', { name: /save model/i })).toBeEnabled()
    expect(screen.getByRole('button', { name: /save & compile/i })).toBeEnabled()
    expect(screen.getByRole('button', { name: /validate/i })).toBeEnabled()
    expect(screen.getByRole('button', { name: /open workspace/i })).toBeEnabled()
    expect(screen.queryByRole('button', { name: /smoke/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /export/i })).not.toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: /execution backend/i })).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: /enable keras/i })).toBeChecked()
    expect(screen.getByText('Saved')).toBeInTheDocument()
  })
})
