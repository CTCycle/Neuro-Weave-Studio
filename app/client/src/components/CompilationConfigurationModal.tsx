import { useEffect, useMemo, useState } from 'react'
import type { CompilationCatalog, ModelConfiguration } from '../domain/types'
import { MathFormula } from './MathFormula'

interface Props {
  configuration: ModelConfiguration
  catalog: CompilationCatalog
  outputIds: string[]
  onApply: (configuration: ModelConfiguration) => void
  onClose: () => void
}

type CompilationTab = 'optimizer' | 'objective' | 'metrics' | 'outputs' | 'advanced'

interface CompilationTabDefinition {
  id: CompilationTab
  label: string
  detail: string
}

export function CompilationConfigurationModal({ configuration, catalog, outputIds, onApply, onClose }: Props) {
  const [draft, setDraft] = useState<ModelConfiguration>(() => structuredClone(configuration))
  const [activeTab, setActiveTab] = useState<CompilationTab>('optimizer')
  useEffect(() => setDraft(structuredClone(configuration)), [configuration])
  useEffect(() => {
    if (activeTab === 'outputs' && outputIds.length === 0) setActiveTab('optimizer')
  }, [activeTab, outputIds.length])

  const optimizer = catalog.optimizers.find((item) => item.id === draft.optimizer) ?? catalog.optimizers[0]
  const loss = catalog.losses.find((item) => item.id === (draft.losses.__default__ ?? 'categorical_crossentropy')) ?? catalog.losses[0]
  const selectedMetrics = draft.metrics.__default__ ?? []
  const optimizerParameters = useMemo(() => optimizer?.parameters ?? [], [optimizer])
  const tabs: CompilationTabDefinition[] = [
    { id: 'optimizer', label: 'Optimizer', detail: optimizer?.label ?? 'Update rule' },
    { id: 'objective', label: 'Loss', detail: loss?.label ?? 'Default objective' },
    { id: 'metrics', label: 'Metrics', detail: selectedMetrics.length ? String(selectedMetrics.length) + ' selected' : 'None selected' },
    ...(outputIds.length > 0 ? [{ id: 'outputs' as const, label: 'Outputs', detail: String(outputIds.length) + ' available' }] : []),
    { id: 'advanced', label: 'Advanced', detail: 'Model defaults' },
  ]

  const chooseOptimizer = (optimizerId: string) => {
    const next = catalog.optimizers.find((item) => item.id === optimizerId)
    if (!next) return
    const parameters = Object.fromEntries(next.parameters.map((parameter) => [parameter.id, draft.optimizer_parameters[parameter.id] ?? parameter.default]))
    setDraft((current) => ({ ...current, optimizer: optimizerId, optimizer_parameters: parameters }))
  }

  const setOptimizerParameter = (id: string, value: string | boolean) => {
    const parameter = optimizerParameters.find((item) => item.id === id)
    const parsed = parameter?.type === 'boolean' ? value === true : Number(value)
    setDraft((current) => ({ ...current, optimizer_parameters: { ...current.optimizer_parameters, [id]: parsed } }))
  }

  const toggleMetric = (metricId: string, enabled: boolean) => {
    const metrics = enabled ? [...new Set([...selectedMetrics, metricId])] : selectedMetrics.filter((item) => item !== metricId)
    setDraft((current) => ({ ...current, metrics: { ...current.metrics, __default__: metrics } }))
  }

  const tabPanelId = (tab: CompilationTab) => 'compilation-tabpanel-' + tab
  const tabButtonId = (tab: CompilationTab) => 'compilation-tab-' + tab

  return (
    <div className="dialog-backdrop compilation-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <section className="compilation-modal panel" role="dialog" aria-modal="true" aria-labelledby="compilation-modal-title">
        <header className="compilation-modal-header"><div><span className="eyebrow">COMPILATION CATALOGUE</span><h2 id="compilation-modal-title">Configure Keras compilation</h2><p>These controls are validated and serialized by the backend catalogue.</p></div><button type="button" className="panel-close" onClick={onClose} aria-label="Close compilation configuration">×</button></header>
        <div className="compilation-modal-tabs" role="tablist" aria-label="Compilation configuration sections">
          {tabs.map((tab) => (
            <button
              type="button"
              role="tab"
              key={tab.id}
              id={tabButtonId(tab.id)}
              aria-selected={activeTab === tab.id}
              aria-controls={tabPanelId(tab.id)}
              className={'compilation-tab ' + (activeTab === tab.id ? 'active' : '')}
              onClick={() => setActiveTab(tab.id)}
            >
              <span>{tab.label}</span>
              <small>{tab.detail}</small>
            </button>
          ))}
        </div>
        <div className="compilation-modal-body">
          <section id={tabPanelId('optimizer')} role="tabpanel" tabIndex={0} aria-labelledby={tabButtonId('optimizer')} hidden={activeTab !== 'optimizer'} className="compilation-section compilation-tab-panel">
            <div className="section-heading"><div><span className="eyebrow">OPTIMIZER</span><h3>Update rule</h3></div></div>
            <label className="field"><span>Optimizer</span><select value={optimizer?.id ?? ''} onChange={(event) => chooseOptimizer(event.target.value)}>{catalog.optimizers.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
            {optimizer && <p className="compilation-description">{optimizer.description}</p>}
            <div className="compilation-parameter-grid">{optimizerParameters.map((parameter) => <label className="field" key={parameter.id}><span>{parameter.label}<small>{parameter.description}</small></span>{parameter.type === 'boolean' ? <input type="checkbox" checked={Boolean(draft.optimizer_parameters[parameter.id] ?? parameter.default)} onChange={(event) => setOptimizerParameter(parameter.id, event.target.checked)} /> : <input type="number" value={String(draft.optimizer_parameters[parameter.id] ?? parameter.default)} min={typeof parameter.constraints.min === 'number' ? parameter.constraints.min : undefined} max={typeof parameter.constraints.max === 'number' ? parameter.constraints.max : undefined} step="any" onChange={(event) => setOptimizerParameter(parameter.id, event.target.value)} />}</label>)}</div>
          </section>
          <section id={tabPanelId('objective')} role="tabpanel" tabIndex={0} aria-labelledby={tabButtonId('objective')} hidden={activeTab !== 'objective'} className="compilation-section compilation-tab-panel">
            <div className="section-heading"><div><span className="eyebrow">LOSS</span><h3>Default objective</h3></div></div>
            <label className="field"><span>Loss</span><select value={loss?.id ?? ''} onChange={(event) => setDraft((current) => ({ ...current, losses: { ...current.losses, __default__: event.target.value } }))}>{catalog.losses.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
            {loss?.description && <p className="compilation-description">{loss.description}</p>}
            {loss?.formula && <MathFormula formula={loss.formula} />}
          </section>
          <section id={tabPanelId('metrics')} role="tabpanel" tabIndex={0} aria-labelledby={tabButtonId('metrics')} hidden={activeTab !== 'metrics'} className="compilation-section compilation-tab-panel">
            <div className="section-heading"><div><span className="eyebrow">METRICS</span><h3>Default measurements</h3></div></div>
            <div className="compilation-option-grid">{catalog.metrics.map((metric) => <label className="compilation-option" key={metric.id}><input type="checkbox" checked={selectedMetrics.includes(metric.id)} onChange={(event) => toggleMetric(metric.id, event.target.checked)} /><span><strong>{metric.label}</strong><small>{metric.description}</small>{metric.formula && <MathFormula formula={metric.formula} />}</span></label>)}</div>
          </section>
          {outputIds.length > 0 && <section id={tabPanelId('outputs')} role="tabpanel" tabIndex={0} aria-labelledby={tabButtonId('outputs')} hidden={activeTab !== 'outputs'} className="compilation-section compilation-tab-panel">
            <div className="section-heading"><div><span className="eyebrow">OUTPUT OVERRIDES</span><h3>Per-output loss</h3></div></div>
            {outputIds.map((outputId) => <label className="field" key={outputId}><span>{outputId}</span><select value={draft.losses[outputId] ?? ''} onChange={(event) => setDraft((current) => { const losses = { ...current.losses }; if (event.target.value) losses[outputId] = event.target.value; else delete losses[outputId]; return { ...current, losses } })}><option value="">Use default loss</option>{catalog.losses.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>)}
          </section>}
          <section id={tabPanelId('advanced')} role="tabpanel" tabIndex={0} aria-labelledby={tabButtonId('advanced')} hidden={activeTab !== 'advanced'} className="compilation-section compilation-tab-panel">
            <div className="section-heading"><div><span className="eyebrow">ADVANCED OPTIONS</span><h3>Model defaults</h3></div><small className="compilation-description">Initializers, regularization, precision</small></div>
            <div className="compilation-parameter-grid"><label className="field"><span>Initializer</span><select value={draft.default_initializer} onChange={(event) => setDraft((current) => ({ ...current, default_initializer: event.target.value }))}>{catalog.initializers.map((item) => <option key={item} value={item}>{item}</option>)}</select></label><label className="field"><span>Regularizer</span><select value={draft.default_regularizer} onChange={(event) => setDraft((current) => ({ ...current, default_regularizer: event.target.value }))}>{catalog.regularizers.map((item) => <option key={item} value={item}>{item}</option>)}</select></label><label className="field"><span>Constraint</span><select value={draft.default_constraint} onChange={(event) => setDraft((current) => ({ ...current, default_constraint: event.target.value }))}>{catalog.constraints.map((item) => <option key={item} value={item}>{item}</option>)}</select></label><label className="toggle-field"><span>Mixed precision</span><input type="checkbox" checked={draft.mixed_precision} onChange={(event) => setDraft((current) => ({ ...current, mixed_precision: event.target.checked }))} /></label></div>
          </section>
        </div>
        <footer className="dialog-actions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button type="button" className="primary" onClick={() => onApply(draft)}>Apply configuration</button></footer>
      </section>
    </div>
  )
}
