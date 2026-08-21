import { useState } from 'react'
import { useAppResources } from '../state/useAppResources'
import { CompilationConfigurationModal } from './CompilationConfigurationModal'
import { CompilationOverview } from './CompilationOverview'
import { useGraphStore } from '../state/graphStore'

export function ModelConfigurationPanel({ kerasEnabled = true }: { kerasEnabled?: boolean }) {
  const configuration = useGraphStore((state) => state.modelConfiguration)
  const update = useGraphStore((state) => state.updateConfiguration)
  const nodes = useGraphStore((state) => state.nodes)
  const { compilationCatalog } = useAppResources()
  const [modalOpen, setModalOpen] = useState(false)
  const outputIds = nodes.filter((node) => node.data.manifest.id.startsWith('output.')).map((node) => node.id)
  return (
    <>
      <section className="configuration-strip panel">
        <CompilationOverview configuration={configuration} catalog={compilationCatalog} kerasEnabled={kerasEnabled} onConfigure={() => setModalOpen(true)} />
      </section>
      {modalOpen && compilationCatalog && <CompilationConfigurationModal configuration={configuration} catalog={compilationCatalog} outputIds={outputIds} onClose={() => setModalOpen(false)} onApply={(next) => { update(next); setModalOpen(false) }} />}
    </>
  )
}
