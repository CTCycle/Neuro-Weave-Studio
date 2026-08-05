import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ProjectToolbar } from './ProjectToolbar'

function renderToolbar(backends: ('tensorflow' | 'torch')[], backend: 'tensorflow' | 'torch' | null) {
  render(
    <ProjectToolbar
      name="Model"
      onNameChange={vi.fn()}
      description=""
      onDescriptionChange={vi.fn()}
      framework="keras"
      onFrameworkChange={vi.fn()}
      availableKerasBackends={backends}
      kerasBackend={backend}
      onKerasBackendChange={vi.fn()}
      projects={[]}
      currentProjectId="project-1"
      onNew={vi.fn()}
      onSave={vi.fn()}
      onOpen={vi.fn()}
      onDuplicate={vi.fn()}
      onValidate={vi.fn()}
      onCode={vi.fn()}
      onExport={vi.fn()}
      onBinaryExport={vi.fn()}
      onImport={vi.fn()}
      busy={false}
    />,
  )
}

describe('ProjectToolbar Keras backend selector', () => {
  afterEach(cleanup)

  it('disables selection and binary export with no operational backend', () => {
    renderToolbar([], null)
    expect(screen.getByRole('combobox', { name: 'Keras backend' })).toBeDisabled()
    expect(screen.getByRole('option', { name: 'Install Keras backend' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Export .keras' })).toBeDisabled()
  })

  it('uses the only operational backend for binary export', () => {
    renderToolbar(['tensorflow'], 'tensorflow')
    expect(screen.getByRole('combobox', { name: 'Keras backend' })).toBeEnabled()
    expect(screen.getByRole('option', { name: 'Keras · TensorFlow' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Export .keras' })).toBeEnabled()
  })

  it('requires an explicit choice when multiple backends are operational', () => {
    renderToolbar(['tensorflow', 'torch'], null)
    expect(screen.getByRole('combobox', { name: 'Keras backend' })).toHaveValue('')
    expect(screen.getByRole('option', { name: 'Keras · TensorFlow' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Keras · Torch' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Export .keras' })).toBeDisabled()
  })
})
