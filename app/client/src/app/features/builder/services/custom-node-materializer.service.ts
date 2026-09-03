import { Injectable, inject } from '@angular/core'
import type { CompositeNodeComponent, CustomNodeDefinition } from '../../../domain/custom-node.types'
import type { NodeManifest } from '../../../domain/types'
import { GraphEditorStore } from '../state/graph-editor.store'

interface ResolvedComponent {
  definition: CompositeNodeComponent
  manifest: NodeManifest
}

@Injectable({ providedIn: 'root' })
export class CustomNodeMaterializerService {
  private readonly store = inject(GraphEditorStore)

  materialize(definition: CustomNodeDefinition, position: { x: number; y: number }): void {
    if (definition.spec.type === 'mathematical') {
      this.materializeMathematical(definition, position)
      return
    }
    if (definition.spec.type === 'composite') {
      this.materializeComposite(definition, position)
      return
    }
    throw new Error(
      'Repository references are verified metadata only until a trusted execution adapter is available.',
    )
  }

  private materializeMathematical(
    definition: CustomNodeDefinition,
    position: { x: number; y: number },
  ): void {
    if (definition.spec.type !== 'mathematical') return
    if (!definition.spec.expression) {
      throw new Error('The mathematical definition has not been normalized by the server.')
    }
    const manifest = this.manifest('math.expression', 1)
    const nodeId = this.store.addNode(manifest, position)
    const node = this.store.nodes().find((item) => item.id === nodeId)
    this.store.updateNode(nodeId, {
      label: definition.display_name,
      parameters: {
        ...(node?.data.parameters ?? {}),
        expression: definition.spec.expression,
      },
    })
  }

  private materializeComposite(
    definition: CustomNodeDefinition,
    position: { x: number; y: number },
  ): void {
    if (definition.spec.type !== 'composite') return
    const resolved: ResolvedComponent[] = definition.spec.components.map((component) => ({
      definition: component,
      manifest: this.manifest(component.definition_id, component.definition_version),
    }))
    const portPairs = resolved.slice(0, -1).map((component, index) =>
      this.compatiblePorts(component.manifest, resolved[index + 1].manifest),
    )
    const nodeIds = resolved.map((component, index) => {
      const nodeId = this.store.addNode(component.manifest, {
        x: position.x + index * 280,
        y: position.y,
      })
      this.store.updateNode(nodeId, {
        label: component.definition.label || (index === 0
          ? definition.display_name
          : component.manifest.display_name),
        parameters: structuredClone(component.definition.parameters),
      })
      return nodeId
    })
    portPairs.forEach((pair, index) => {
      const result = this.store.connect({
        source: nodeIds[index],
        sourceHandle: pair.sourceHandle,
        target: nodeIds[index + 1],
        targetHandle: pair.targetHandle,
      })
      if (!result.valid) {
        throw new Error(`Could not connect sequential template components: ${result.message}`)
      }
    })
  }

  private compatiblePorts(
    source: NodeManifest,
    target: NodeManifest,
  ): { sourceHandle: string; targetHandle: string } {
    for (const output of source.outputs) {
      for (const input of target.inputs) {
        const expected = input.kind === 'tensor_list'
          ? input.element_kind ?? 'tensor'
          : input.kind
        if (output.kind === expected) {
          return { sourceHandle: output.id, targetHandle: input.id }
        }
      }
    }
    throw new Error(
      `Sequential template cannot connect ${source.display_name} to ${target.display_name}.`,
    )
  }

  private manifest(id: string, version: number): NodeManifest {
    const manifest = this.store.manifests().find(
      (item) => item.id === id && item.version === version,
    )
    if (!manifest) throw new Error(`Built-in node ${id}@${version} is no longer available.`)
    return manifest
  }
}
