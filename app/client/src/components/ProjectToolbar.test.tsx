import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ProjectToolbar } from './ProjectToolbar'
import type { ExecutionBackend } from '../domain/types'

function renderToolbar(backendStatuses: Array<{ id: ExecutionBackend; available: boolean; reason_code?: string }>, backend: ExecutionBackend | null) {
  render(
    <ProjectToolbar
      name="Model"
      onNameChange={vi.fn()}
      description=""
      onDescriptionChange={vi.fn()}
      codeTarget="keras"
      onCodeTargetChange={vi.fn()}
      backendStatuses={backendStatuses}
      executionBackend={backend}
      onExecutionBackendChange={vi.fn()}
      projects={[]}
      currentProjectId="project-1"
      onNew={vi.fn()}
      onSave={vi.fn()}
      onOpen={vi.fn()}
      onDuplicate={vi.fn()}
      onValidate={vi.fn()}
      onSmokeTestOpen={vi.fn()}
      onCode={vi.fn()}
      onExport={vi.fn()}
      onBinaryExport={vi.fn()}
      onImport={vi.fn()}
      busy={false}
    />,
  )
}

describe('ProjectToolbar execution backend selector', () => {
  afterEach(cleanup)

  it('disables selection and binary export with no operational backend', () => {
    renderToolbar([
      { id: 'tensorflow', available: false, reason_code: 'dependency_missing' },
      { id: 'pytorch', available: false, reason_code: 'dependency_missing' },
      { id: 'jax', available: false, reason_code: 'dependency_missing' },
    ], null)
    expect(screen.getByRole('combobox', { name: 'Execution backend' })).toBeDisabled()
    expect(screen.getByRole('option', { name: 'None operational' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Export .keras' })).toBeDisabled()
  })

  it('uses the only operational backend for binary export', () => {
    renderToolbar([
      { id: 'tensorflow', available: true },
      { id: 'pytorch', available: false, reason_code: 'dependency_missing' },
      { id: 'jax', available: false, reason_code: 'dependency_missing' },
    ], 'tensorflow')
    expect(screen.getByRole('combobox', { name: 'Execution backend' })).toBeEnabled()
    expect(screen.getByRole('option', { name: /tensorflow/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Export .keras' })).toBeEnabled()
  })

  it('requires an explicit choice when multiple backends are operational', () => {
    renderToolbar([
      { id: 'tensorflow', available: true },
      { id: 'pytorch', available: true },
      { id: 'jax', available: false, reason_code: 'dependency_missing' },
    ], null)
    expect(screen.getByRole('combobox', { name: 'Execution backend' })).toHaveValue('')
    expect(screen.getByRole('option', { name: /tensorflow/ })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: /pytorch/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Export .keras' })).toBeDisabled()
  })
})
