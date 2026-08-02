import { useGraphStore } from '../state/graphStore'

export function ModelConfigurationPanel() {
  const configuration = useGraphStore((state) => state.modelConfiguration)
  const update = useGraphStore((state) => state.updateConfiguration)
  const defaultLoss = configuration.losses.__default__ ?? 'categorical_crossentropy'
  const defaultMetrics = configuration.metrics.__default__ ?? ['accuracy']
  return (
    <section className="configuration-strip panel">
      <div><span className="eyebrow">MODEL SETUP</span><strong>Compilation defaults</strong></div>
      <label className="compact-field"><span>Optimizer</span><select value={configuration.optimizer} onChange={(event) => update({ optimizer: event.target.value })}><option value="adam">Adam</option><option value="adamw">AdamW</option><option value="sgd">SGD</option><option value="rmsprop">RMSprop</option></select></label>
      <label className="compact-field"><span>Learning rate</span><input type="number" step="0.0001" value={Number(configuration.optimizer_parameters.learning_rate ?? 0.001)} onChange={(event) => update({ optimizer_parameters: { ...configuration.optimizer_parameters, learning_rate: Number(event.target.value) } })} /></label>
      <label className="compact-field"><span>Default loss</span><select value={defaultLoss} onChange={(event) => update({ losses: { ...configuration.losses, __default__: event.target.value } })}><option value="categorical_crossentropy">Categorical CE</option><option value="sparse_categorical_crossentropy">Sparse categorical CE</option><option value="binary_crossentropy">Binary CE</option><option value="mean_squared_error">MSE</option><option value="mean_absolute_error">MAE</option></select></label>
      <label className="compact-field"><span>Metrics</span><input value={defaultMetrics.join(', ')} onChange={(event) => update({ metrics: { ...configuration.metrics, __default__: event.target.value.split(',').map((item) => item.trim()).filter(Boolean) } })} /></label>
      <label className="compact-field"><span>Initializer</span><select value={configuration.default_initializer} onChange={(event) => update({ default_initializer: event.target.value })}><option value="glorot_uniform">Glorot uniform</option><option value="he_normal">He normal</option><option value="zeros">Zeros</option><option value="ones">Ones</option></select></label>
      <label className="compact-field"><span>Regularizer</span><select value={configuration.default_regularizer} onChange={(event) => update({ default_regularizer: event.target.value })}><option value="none">None</option><option value="l1">L1</option><option value="l2">L2</option><option value="l1_l2">L1 + L2</option></select></label>
      <label className="compact-field"><span>Constraint</span><select value={configuration.default_constraint} onChange={(event) => update({ default_constraint: event.target.value })}><option value="none">None</option><option value="max_norm">Max norm</option><option value="non_neg">Non-negative</option></select></label>
      <label className="compact-toggle"><input type="checkbox" checked={configuration.mixed_precision} onChange={(event) => update({ mixed_precision: event.target.checked })} /><span>Mixed precision</span></label>
    </section>
  )
}
