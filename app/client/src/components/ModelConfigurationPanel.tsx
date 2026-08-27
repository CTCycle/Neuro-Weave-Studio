import { useState } from 'react'
import type { ExecutionBackend } from '../domain/types'
import { useAppResources } from '../state/useAppResources'
import { CompilationConfigurationModal } from './CompilationConfigurationModal'
import { CompilationOverview } from './CompilationOverview'
import { useGraphStore } from '../state/graphStore'

interface Props {
  executionBackend: ExecutionBackend
  kerasEnabled: boolean
}

export function ModelConfigurationPanel({ executionBackend, kerasEnabled }: Props) {
  const configuration = useGraphStore((state) => state.modelConfiguration)
  const update = useGraphStore((state) => state.updateConfiguration)
  const nodes = useGraphStore((state) => state.nodes)
  const { compilationCapabilities } = useAppResources()
  const [modalOpen, setModalOpen] = useState(false)
  const outputIds = nodes.filter((node) => node.data.manifest.id.startsWith('output.')).map((node) => node.id)
  const kerasCatalog = compilationCapabilities?.keras
  const nativeCapability = compilationCapabilities?.native[executionBackend]
  return (
    <>
      <section className="configuration-strip panel">
        <CompilationOverview configuration={configuration} executionBackend={executionBackend} kerasEnabled={kerasEnabled} kerasCatalog={kerasCatalog} nativeCapability={nativeCapability} onConfigure={() => setModalOpen(true)} />
      </section>
      {modalOpen && (kerasCatalog || nativeCapability) && <CompilationConfigurationModal configuration={configuration} kerasEnabled={kerasEnabled} kerasCatalog={kerasCatalog} nativeCapability={nativeCapability} outputIds={outputIds} onClose={() => setModalOpen(false)} onApply={(next) => { update(next); setModalOpen(false) }} />}
    </>
  )
}
