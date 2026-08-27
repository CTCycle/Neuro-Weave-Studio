import { useEffect, useMemo, useState } from 'react'
import type {
  CompilationCatalog,
  CompilationParameter,
  ExecutionBackend,
  KerasCompilationConfiguration,
  ModelConfiguration,
  NativeCompilationCapability,
} from '../domain/types'
import { MathFormula } from './MathFormula'

interface Props {
  configuration: ModelConfiguration
  kerasEnabled: boolean
  kerasCatalog?: CompilationCatalog | null
  nativeCapability?: NativeCompilationCapability | null
  outputIds: string[]
  onApply: (configuration: ModelConfiguration) => void
  onClose: () => void
}

type CompilationTab = 'optimizer' | 'objective' | 'metrics' | 'outputs' | 'advanced' | 'native'

interface CompilationTabDefinition {
  id: CompilationTab
  label: string
  detail: string
}

function parameterValue(configuration: Record<string, unknown>, parameter: CompilationParameter): string | number | boolean {
  const value = configuration[parameter.id] ?? parameter.default
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value
  return String(parameter.default)
}

function nativeBranch(configuration: ModelConfiguration, backend: ExecutionBackend): Record<string, unknown> {
  return configuration.native[backend] as unknown as Record<string, unknown>
}

export function CompilationConfigurationModal({ configuration, kerasEnabled, kerasCatalog, nativeCapability, outputIds, onApply, onClose }: Props) {
  const [draft, setDraft] = useState<ModelConfiguration>(() => structuredClone(configuration))
  const [activeTab, setActiveTab] = useState<CompilationTab>(kerasEnabled ? 'optimizer' : 'native')
  useEffect(() => setDraft(structuredClone(configuration)), [configuration])
  useEffect(() => setActiveTab(kerasEnabled ? 'optimizer' : 'native'), [kerasEnabled])
  useEffect(() => {
    if (activeTab === 'outputs' && outputIds.length === 0) setActiveTab('optimizer')
  }, [activeTab, outputIds.length])

  const optimizer = kerasCatalog?.optimizers.find((item) => item.id === draft.keras.optimizer) ?? kerasCatalog?.optimizers[0]
  const lossId = draft.keras.losses.__default__ ?? 'categorical_crossentropy'
  const loss = kerasCatalog?.losses.find((item) => item.id === lossId) ?? kerasCatalog?.losses[0]
  const selectedMetrics = draft.keras.metrics.__default__ ?? []
  const optimizerParameters = useMemo(() => optimizer?.parameters ?? [], [optimizer])
  const tabs: CompilationTabDefinition[] = kerasEnabled
    ? [
        { id: 'optimizer', label: 'Optimizer', detail: optimizer?.label ?? 'Update rule' },
        { id: 'objective', label: 'Loss', detail: loss?.label ?? 'Default objective' },
        { id: 'metrics', label: 'Metrics', detail: selectedMetrics.length ? `${selectedMetrics.length} selected` : 'None selected' },
        ...(outputIds.length > 0 ? [{ id: 'outputs' as const, label: 'Outputs', detail: `${outputIds.length} available` }] : []),
        { id: 'advanced', label: 'Advanced', detail: 'Model defaults' },
      ]
    : [{ id: 'native', label: 'Native compiler', detail: nativeCapability?.compiler ?? 'Backend settings' }]

  const chooseOptimizer = (optimizerId: string) => {
    const next = kerasCatalog?.optimizers.find((item) => item.id === optimizerId)
    if (!next) return
    const parameters = Object.fromEntries(next.parameters.map((parameter) => [parameter.id, draft.keras.optimizer_parameters[parameter.id] ?? parameter.default]))
    setDraft((current) => ({ ...current, keras: { ...current.keras, optimizer: optimizerId, optimizer_parameters: parameters } }))
  }

  const setKeras = (patch: Partial<KerasCompilationConfiguration>) => {
    setDraft((current) => ({ ...current, keras: { ...current.keras, ...patch } }))
  }

  const setOptimizerParameter = (id: string, value: string | boolean) => {
    const parameter = optimizerParameters.find((item) => item.id === id)
    const parsed = parameter?.type === 'boolean' ? value === true : Number(value)
    setDraft((current) => ({
      ...current,
      keras: {
        ...current.keras,
        optimizer_parameters: { ...current.keras.optimizer_parameters, [id]: parsed },
      },
    }))
  }

  const toggleMetric = (metricId: string, enabled: boolean) => {
    const metrics = enabled ? [...new Set([...selectedMetrics, metricId])] : selectedMetrics.filter((item) => item !== metricId)
    setKeras({ metrics: { ...draft.keras.metrics, __default__: metrics } })
  }

  const setNativeParameter = (id: string, value: string | boolean) => {
    if (!nativeCapability) return
    const backend = nativeCapability.backend
    const parameter = nativeCapability.parameters.find((item) => item.id === id)
    const parsed = parameter?.type === 'boolean' ? value === true : value
    setDraft((current) => {
      const branch = nativeBranch(current, backend)
      return {
        ...current,
        native: { ...current.native, [backend]: { ...branch, [id]: parsed } },
      }
    })
  }

  const tabPanelId = (tab: CompilationTab) => `compilation-tabpanel-${tab}`
  const tabButtonId = (tab: CompilationTab) => `compilation-tab-${tab}`
  const nativeValues = nativeCapability ? nativeBranch(draft, nativeCapability.backend) : {}

  return (
    <div className="dialog-backdrop compilation-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <section className="compilation-modal panel" role="dialog" aria-modal="true" aria-labelledby="compilation-modal-title">
        <header className="compilation-modal-header">
          <div>
            <span className="eyebrow">COMPILATION CONFIGURATION</span>
            <h2 id="compilation-modal-title">{kerasEnabled ? 'Configure Keras compilation' : `Configure native ${nativeCapability?.backend ?? 'backend'}`}</h2>
            <p>{kerasEnabled ? 'These controls are validated and serialized by the Keras catalogue.' : 'These controls map directly to the selected backend compiler.'}</p>
          </div>
          <button type="button" className="panel-close" onClick={onClose} aria-label="Close compilation configuration">×</button>
        </header>
        <div className="compilation-modal-tabs" role="tablist" aria-label="Compilation configuration sections">
          {tabs.map((tab) => (
            <button type="button" role="tab" key={tab.id} id={tabButtonId(tab.id)} aria-selected={activeTab === tab.id} aria-controls={tabPanelId(tab.id)} className={`compilation-tab ${activeTab === tab.id ? 'active' : ''}`} onClick={() => setActiveTab(tab.id)}>
              <span>{tab.label}</span>
              <small>{tab.detail}</small>
            </button>
          ))}
        </div>
        <div className="compilation-modal-body">
          {kerasEnabled && kerasCatalog && <>
            <section id={tabPanelId('optimizer')} role="tabpanel" tabIndex={0} aria-labelledby={tabButtonId('optimizer')} hidden={activeTab !== 'optimizer'} className="compilation-section compilation-tab-panel">
              <div className="section-heading"><div><span className="eyebrow">OPTIMIZER</span><h3>Update rule</h3></div></div>
              <label className="field"><span>Optimizer</span><select value={optimizer?.id ?? ''} onChange={(event) => chooseOptimizer(event.target.value)}>{kerasCatalog.optimizers.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
              {optimizer && <p className="compilation-description">{optimizer.description}</p>}
              <div className="compilation-parameter-grid">{optimizerParameters.map((parameter) => <label className="field" key={parameter.id}><span>{parameter.label}<small>{parameter.description}</small></span>{parameter.type === 'boolean' ? <input type="checkbox" checked={Boolean(draft.keras.optimizer_parameters[parameter.id] ?? parameter.default)} onChange={(event) => setOptimizerParameter(parameter.id, event.target.checked)} /> : <input type="number" value={String(draft.keras.optimizer_parameters[parameter.id] ?? parameter.default)} min={typeof parameter.constraints.min === 'number' ? parameter.constraints.min : undefined} max={typeof parameter.constraints.max === 'number' ? parameter.constraints.max : undefined} step="any" onChange={(event) => setOptimizerParameter(parameter.id, event.target.value)} />}</label>)}</div>
            </section>
            <section id={tabPanelId('objective')} role="tabpanel" tabIndex={0} aria-labelledby={tabButtonId('objective')} hidden={activeTab !== 'objective'} className="compilation-section compilation-tab-panel">
              <div className="section-heading"><div><span className="eyebrow">LOSS</span><h3>Default objective</h3></div></div>
              <label className="field"><span>Loss</span><select value={loss?.id ?? ''} onChange={(event) => setKeras({ losses: { ...draft.keras.losses, __default__: event.target.value } })}>{kerasCatalog.losses.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
              {loss?.description && <p className="compilation-description">{loss.description}</p>}
              {loss?.formula && <MathFormula formula={loss.formula} />}
            </section>
            <section id={tabPanelId('metrics')} role="tabpanel" tabIndex={0} aria-labelledby={tabButtonId('metrics')} hidden={activeTab !== 'metrics'} className="compilation-section compilation-tab-panel">
              <div className="section-heading"><div><span className="eyebrow">METRICS</span><h3>Default measurements</h3></div></div>
              <div className="compilation-option-grid">{kerasCatalog.metrics.map((metric) => <label className="compilation-option" key={metric.id}><input type="checkbox" checked={selectedMetrics.includes(metric.id)} onChange={(event) => toggleMetric(metric.id, event.target.checked)} /><span><strong>{metric.label}</strong><small>{metric.description}</small>{metric.formula && <MathFormula formula={metric.formula} />}</span></label>)}</div>
            </section>
            {outputIds.length > 0 && <section id={tabPanelId('outputs')} role="tabpanel" tabIndex={0} aria-labelledby={tabButtonId('outputs')} hidden={activeTab !== 'outputs'} className="compilation-section compilation-tab-panel">
              <div className="section-heading"><div><span className="eyebrow">OUTPUT OVERRIDES</span><h3>Per-output loss</h3></div></div>
              {outputIds.map((outputId) => <label className="field" key={outputId}><span>{outputId}</span><select value={draft.keras.losses[outputId] ?? ''} onChange={(event) => setDraft((current) => { const losses = { ...current.keras.losses }; if (event.target.value) losses[outputId] = event.target.value; else delete losses[outputId]; return { ...current, keras: { ...current.keras, losses } } })}><option value="">Use default loss</option>{kerasCatalog.losses.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>)}
            </section>}
            <section id={tabPanelId('advanced')} role="tabpanel" tabIndex={0} aria-labelledby={tabButtonId('advanced')} hidden={activeTab !== 'advanced'} className="compilation-section compilation-tab-panel">
              <div className="section-heading"><div><span className="eyebrow">ADVANCED OPTIONS</span><h3>Model defaults</h3></div><small className="compilation-description">Initializers, regularization, precision</small></div>
              <div className="compilation-parameter-grid"><label className="field"><span>Initializer</span><select value={draft.keras.default_initializer} onChange={(event) => setKeras({ default_initializer: event.target.value })}>{kerasCatalog.initializers.map((item) => <option key={item} value={item}>{item}</option>)}</select></label><label className="field"><span>Regularizer</span><select value={draft.keras.default_regularizer} onChange={(event) => setKeras({ default_regularizer: event.target.value })}>{kerasCatalog.regularizers.map((item) => <option key={item} value={item}>{item}</option>)}</select></label><label className="field"><span>Constraint</span><select value={draft.keras.default_constraint} onChange={(event) => setKeras({ default_constraint: event.target.value })}>{kerasCatalog.constraints.map((item) => <option key={item} value={item}>{item}</option>)}</select></label><label className="toggle-field"><span>Mixed precision</span><input type="checkbox" checked={draft.keras.mixed_precision} onChange={(event) => setKeras({ mixed_precision: event.target.checked })} /></label></div>
            </section>
          </>}
          {!kerasEnabled && nativeCapability && <section id={tabPanelId('native')} role="tabpanel" tabIndex={0} aria-labelledby={tabButtonId('native')} className="compilation-section compilation-tab-panel">
            <div className="section-heading"><div><span className="eyebrow">{nativeCapability.compiler}</span><h3>Backend-native execution</h3></div></div>
            {!nativeCapability.available && <p className="compilation-unavailable">{nativeCapability.reason ?? 'This compiler is not operational in the current environment. Settings can be saved for a later runtime.'}</p>}
            <div className="compilation-parameter-grid">{nativeCapability.parameters.map((parameter) => {
              const value = parameterValue(nativeValues, parameter)
              if (parameter.type === 'boolean') return <label className="toggle-field" key={parameter.id}><span>{parameter.label}<small>{parameter.description}</small></span><input type="checkbox" checked={Boolean(value)} onChange={(event) => setNativeParameter(parameter.id, event.target.checked)} /></label>
              if (parameter.type === 'enum') return <label className="field" key={parameter.id}><span>{parameter.label}<small>{parameter.description}</small></span><select value={String(value)} onChange={(event) => setNativeParameter(parameter.id, event.target.value)}>{(parameter.options ?? []).map((option) => <option key={option} value={option}>{option}</option>)}</select></label>
              return <label className="field" key={parameter.id}><span>{parameter.label}<small>{parameter.description}</small></span><input type="number" value={String(value)} onChange={(event) => setNativeParameter(parameter.id, event.target.value)} /></label>
            })}</div>
          </section>}
        </div>
        <footer className="dialog-actions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button type="button" className="primary" onClick={() => onApply(draft)}>Apply configuration</button></footer>
      </section>
    </div>
  )
}
