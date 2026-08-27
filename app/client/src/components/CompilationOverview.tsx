import type { CompilationCatalog, ExecutionBackend, ModelConfiguration, NativeCompilationCapability } from '../domain/types'

interface Props {
  configuration: ModelConfiguration
  executionBackend: ExecutionBackend
  kerasEnabled: boolean
  kerasCatalog: CompilationCatalog | null | undefined
  nativeCapability: NativeCompilationCapability | null | undefined
  onConfigure: () => void
}

function nativeSummary(configuration: ModelConfiguration, backend: ExecutionBackend): string {
  if (backend === 'tensorflow') {
    return `tf.function · XLA ${configuration.native.tensorflow.jit_compile ? 'enabled' : 'off'}`
  }
  if (backend === 'pytorch') {
    const native = configuration.native.pytorch
    return native.compile_enabled ? `torch.compile · ${native.mode}` : 'torch.compile off'
  }
  return `jax.jit ${configuration.native.jax.jit_enabled ? 'enabled' : 'off'}`
}

export function CompilationOverview({ configuration, executionBackend, kerasEnabled, kerasCatalog, nativeCapability, onConfigure }: Props) {
  if (!kerasEnabled) {
    return (
      <div className="compilation-overview">
        <div>
          <span className="eyebrow">COMPILATION · NATIVE / {executionBackend}</span>
          <strong>{nativeCapability?.compiler ?? `Native ${executionBackend}`} · {nativeSummary(configuration, executionBackend)}</strong>
          <small>{nativeCapability?.available ? 'Native compiler settings are applied to execution.' : nativeCapability?.reason ?? 'Native compiler capability is unavailable.'}</small>
        </div>
        <button type="button" className="secondary" onClick={onConfigure} disabled={!nativeCapability}>Configure</button>
      </div>
    )
  }
  const keras = configuration.keras
  const optimizer = kerasCatalog?.optimizers.find((item) => item.id === keras.optimizer)?.label ?? keras.optimizer
  const lossId = keras.losses.__default__ ?? 'categorical_crossentropy'
  const loss = kerasCatalog?.losses.find((item) => item.id === lossId)?.label ?? lossId
  const metrics = (keras.metrics.__default__ ?? []).map((metric) => kerasCatalog?.metrics.find((item) => item.id === metric)?.label ?? metric)
  const learningRate = keras.optimizer_parameters.learning_rate ?? 0.001
  return (
    <div className="compilation-overview">
      <div>
        <span className="eyebrow">COMPILATION · KERAS / {executionBackend}</span>
        <strong>{optimizer} · lr {String(learningRate)} · {loss} · {metrics.join(', ') || 'No metrics'}</strong>
        <small>Applied when the Keras model is compiled or checked.</small>
      </div>
      <button type="button" className="secondary" onClick={onConfigure} disabled={!kerasCatalog}>Configure</button>
    </div>
  )
}
