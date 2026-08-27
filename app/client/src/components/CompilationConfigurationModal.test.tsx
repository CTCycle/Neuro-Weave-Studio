import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { CompilationConfigurationModal } from './CompilationConfigurationModal'
import { emptyModelConfiguration, type NativeCompilationCapability } from '../domain/types'

const nativeCapability: NativeCompilationCapability = {
  backend: 'pytorch',
  compiler_id: 'torch_compile',
  compiler: 'PyTorch torch.compile',
  available: true,
  parameters: [
    {
      id: 'compile_enabled',
      label: 'Enable torch.compile',
      type: 'boolean',
      default: false,
      description: 'Wrap the native module with PyTorch compiler.',
      constraints: {},
    },
    {
      id: 'mode',
      label: 'Compile mode',
      type: 'enum',
      default: 'default',
      options: ['default', 'max-autotune'],
      description: 'Select the compiler strategy.',
      constraints: {},
    },
  ],
}

describe('CompilationConfigurationModal', () => {
  afterEach(cleanup)

  it('updates native compiler settings together in one canonical configuration', () => {
    const onApply = vi.fn()
    render(
      <CompilationConfigurationModal
        configuration={emptyModelConfiguration()}
        kerasEnabled={false}
        nativeCapability={nativeCapability}
        outputIds={[]}
        onApply={onApply}
        onClose={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('checkbox', { name: /Enable torch\.compile/ }))
    fireEvent.change(screen.getByRole('combobox', { name: /Compile mode/ }), { target: { value: 'max-autotune' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply configuration' }))

    const configuration = onApply.mock.calls[0][0]
    expect(configuration.native.pytorch).toMatchObject({ compile_enabled: true, mode: 'max-autotune' })
    expect(configuration.native.tensorflow.jit_compile).toBe(false)
    expect(configuration.keras.optimizer).toBe('adam')
  })
})
