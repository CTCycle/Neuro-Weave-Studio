import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ProjectToolbar } from './ProjectToolbar'

const baseProps = {
  name: 'Classifier',
  executionBackend: 'pytorch' as const,
  onExecutionBackendChange: vi.fn(),
  kerasEnabled: true,
  onKerasEnabledChange: vi.fn(),
  onSave: vi.fn(),
  onValidate: vi.fn(),
  busy: false,
  dirty: false,
  modelId: null,
  onOpenWorkspace: vi.fn(),
  onNameChange: vi.fn(),
}

describe('ProjectToolbar', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('edits, cancels, saves, and clears the canonical model description', () => {
    const onDescriptionChange = vi.fn()
    const props = { ...baseProps, description: 'Existing purpose', onDescriptionChange }
    const { rerender } = render(<ProjectToolbar {...props} />)

    fireEvent.click(screen.getByRole('button', { name: 'Description' }))
    const editor = screen.getByRole('textbox', { name: 'Model description' })
    fireEvent.change(editor, { target: { value: 'Discarded edit' } })
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onDescriptionChange).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Description' }))
    expect(screen.getByRole('textbox', { name: 'Model description' })).toHaveValue('Existing purpose')
    fireEvent.change(screen.getByRole('textbox', { name: 'Model description' }), { target: { value: 'Saved purpose' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    expect(onDescriptionChange).toHaveBeenCalledWith('Saved purpose')

    rerender(<ProjectToolbar {...props} description="Saved purpose" />)
    fireEvent.click(screen.getByRole('button', { name: 'Description' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    expect(onDescriptionChange).toHaveBeenLastCalledWith('')
  })

  it('labels Keras as an optional API layer over the selected backend', () => {
    render(<ProjectToolbar {...baseProps} description="" onDescriptionChange={vi.fn()} />)

    expect(screen.getByText('Enable Keras')).toBeInTheDocument()
    expect(screen.getByText('Keras is an optional API layer over the selected native execution backend.')).toBeInTheDocument()
    expect(screen.queryByRole('textbox', { name: 'Optional model description' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Save & compile' })).not.toBeInTheDocument()
  })
})
