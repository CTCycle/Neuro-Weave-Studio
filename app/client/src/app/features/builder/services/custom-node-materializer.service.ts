import { Injectable, inject } from '@angular/core'
import type { CompositeNodeComponent, CustomNodeDefinition } from '../../../domain/custom-node.types'
import type { NodeManifest, PortDefinition } from '../../../domain/types'
import { GraphEditorStore } from '../state/graph-editor.store'

interface ResolvedComponent {
  definition: CompositeNodeComponent
  manifest: NodeManifest
}

@Injectable({ providedIn: 'root' })
export class CustomNodeMaterializerService {
  private readonly store = inject(GraphEditorStore)

  materialize(definition: CustomNodeDefinition, position: { x: number; y: number }): void {
    const before = this.store.captureEditorSession()
    try {
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
    } catch (error) {
      this.store.restoreEditorSession(before)
      throw error
    }
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
      this.compatiblePorts(component, resolved[index + 1]),
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
    source: ResolvedComponent,
    target: ResolvedComponent,
  ): { sourceHandle: string; targetHandle: string } {
    const outputs = source.manifest.outputs.filter((port) =>
      this.portEnabled(port, source.definition.parameters),
    )
    const inputs = target.manifest.inputs.filter((port) =>
      this.portEnabled(port, target.definition.parameters),
    )
    for (const output of outputs) {
      for (const input of inputs) {
        const expected = input.kind === 'tensor_list'
          ? input.element_kind ?? 'tensor'
          : input.kind
        if (output.kind === expected) {
          return { sourceHandle: output.id, targetHandle: input.id }
        }
      }
    }
    throw new Error(
      `Sequential template cannot connect ${source.manifest.display_name} to ${target.manifest.display_name}.`,
    )
  }

  private portEnabled(port: PortDefinition, parameters: Record<string, unknown>): boolean {
    const condition = port.enabled_when
    if (!condition) return true
    return Object.entries(condition).every(([parameterId, expected]) => {
      const actual = parameters[parameterId]
      if (typeof expected === 'object' && expected !== null && !Array.isArray(expected)) {
        const rule = expected as Record<string, unknown>
        if ('equals' in rule && !this.sameValue(actual, rule['equals'])) return false
        if ('not_equals' in rule && this.sameValue(actual, rule['not_equals'])) return false
        if (
          'in' in rule
          && Array.isArray(rule['in'])
          && !rule['in'].some((item) => this.sameValue(actual, item))
        ) return false
        if (
          'not_in' in rule
          && Array.isArray(rule['not_in'])
          && rule['not_in'].some((item) => this.sameValue(actual, item))
        ) return false
        return true
      }
      if (Array.isArray(expected)) {
        return expected.some((item) => this.sameValue(actual, item))
      }
      return this.sameValue(actual, expected)
    })
  }

  private sameValue(left: unknown, right: unknown): boolean {
    return JSON.stringify(left) === JSON.stringify(right)
  }

  private manifest(id: string, version: number): NodeManifest {
    const manifest = this.store.manifests().find(
      (item) => item.id === id && item.version === version,
    )
    if (!manifest) throw new Error(`Built-in node ${id}@${version} is no longer available.`)
    return manifest
  }
}
