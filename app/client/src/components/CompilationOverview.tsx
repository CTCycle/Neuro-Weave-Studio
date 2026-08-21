import type { CompilationCatalog, ModelConfiguration } from '../domain/types'

interface Props {
  configuration: ModelConfiguration
  catalog: CompilationCatalog | null | undefined
  kerasEnabled: boolean
  onConfigure: () => void
}
export function CompilationOverview({ configuration, catalog, kerasEnabled, onConfigure }: Props) {
  if (!kerasEnabled) {
    return <div className="compilation-overview"><div><span className="eyebrow">COMPILATION</span><strong>Native backend mode</strong><small>Keras compilation inactive for this model.</small></div></div>
  }
  const optimizer = catalog?.optimizers.find((item) => item.id === configuration.optimizer)?.label ?? configuration.optimizer
  const lossId = configuration.losses.__default__ ?? 'categorical_crossentropy'
  const loss = catalog?.losses.find((item) => item.id === lossId)?.label ?? lossId
  const metrics = (configuration.metrics.__default__ ?? []).map((metric) => catalog?.metrics.find((item) => item.id === metric)?.label ?? metric)
  const learningRate = configuration.optimizer_parameters.learning_rate ?? 0.001
  return (
    <div className="compilation-overview"><div><span className="eyebrow">COMPILATION</span><strong>{optimizer} · lr {String(learningRate)} · {loss} · {metrics.join(', ') || 'No metrics'}</strong><small>Applied when the Keras model is compiled or checked.</small></div><button type="button" className="secondary" onClick={onConfigure} disabled={!catalog}>Configure</button></div>
  )
}
