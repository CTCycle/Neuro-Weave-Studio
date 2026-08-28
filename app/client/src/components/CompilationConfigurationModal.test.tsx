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

const nativeCapabilityWithCatalog: NativeCompilationCapability = {
  ...nativeCapability,
  catalog: {
    optimizers: [{ id: 'adam', label: 'Adam', description: 'Adaptive optimizer.', parameters: [{ id: 'learning_rate', label: 'Learning rate', type: 'number', default: 0.001, description: 'Step size.', constraints: {} }] }],
    losses: [{ id: 'mean_squared_error', label: 'Mean squared error', description: 'Squared error.', parameters: [], formula: String.raw`\operatorname{MSE}=\frac{1}{n}\sum_i(y_i-\hat{y}_i)^2` }],
    metrics: [{ id: 'mean_squared_error', label: 'Mean squared error', description: 'Squared error metric.', parameters: [], formula: String.raw`\operatorname{MSE}=\frac{1}{n}\sum_i(y_i-\hat{y}_i)^2` }],
    initializers: [],
    regularizers: [],
    constraints: [],
  },
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

  it('exposes native PyTorch optimizer, loss, and metric tabs', () => {
    const onApply = vi.fn()
    render(
      <CompilationConfigurationModal
        configuration={emptyModelConfiguration()}
        kerasEnabled={false}
        nativeCapability={nativeCapabilityWithCatalog}
        outputIds={['output']}
        onApply={onApply}
        onClose={vi.fn()}
      />,
    )

    expect(screen.getByRole('tab', { name: /Optimizer/ })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: /^Loss/ }))
    fireEvent.change(screen.getByRole('combobox', { name: 'Loss' }), { target: { value: 'mean_squared_error' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply configuration' }))

    const configuration = onApply.mock.calls[0][0]
    expect(configuration.native.pytorch.losses).toMatchObject({ __default__: 'mean_squared_error' })
    expect(configuration.native.pytorch.optimizer).toBe('adam')
  })
})
