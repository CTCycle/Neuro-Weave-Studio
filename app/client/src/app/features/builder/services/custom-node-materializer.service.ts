import { Inject, Injectable } from '@angular/core'
import type { CompositeNodeComponent, CustomNodeDefinition } from '../../../domain/custom-node.types'
import { effectiveParameterDefault, resolveParameterCompatibility, type NodeManifest, type PortDefinition } from '../../../domain/types'
import { GraphEditorStore } from '../state/graph-editor.store'

interface ResolvedComponent {
  definition: CompositeNodeComponent
  manifest: NodeManifest
  parameters: Record<string, unknown>
}

export interface CustomNodeCompatibility {
  available: boolean
  reason: string
}

@Injectable({ providedIn: 'root' })
export class CustomNodeMaterializerService {
  constructor(@Inject(GraphEditorStore) private readonly store: GraphEditorStore) {}

  compatibility(definition: CustomNodeDefinition): CustomNodeCompatibility {
    if (!definition.executable_template) {
      return {
        available: false,
        reason: 'Repository references are verified metadata only until a trusted execution adapter is available.',
      }
    }
    try {
      if (definition.spec.type === 'mathematical') {
        if (!definition.spec.expression) {
          return { available: false, reason: 'The mathematical definition has not been normalized by the server.' }
        }
        const manifest = this.manifest('math.expression', 1)
        return this.componentCompatibility(
          manifest,
          this.store.nodeParametersFor(manifest, { expression: definition.spec.expression }),
        )
      }
      if (definition.spec.type === 'composite') {
        if (!definition.spec.components.length) {
          return { available: false, reason: 'A composite template must contain at least one built-in component.' }
        }
        const resolved = this.resolveComponents(definition.spec.components)
        for (const component of resolved) {
          const result = this.componentCompatibility(component.manifest, component.parameters)
          if (!result.available) return result
        }
        this.resolvedPortPairs(resolved)
        return { available: true, reason: '' }
      }
      return {
        available: false,
        reason: 'Repository references are verified metadata only until a trusted execution adapter is available.',
      }
    } catch (error) {
      return { available: false, reason: error instanceof Error ? error.message : String(error) }
    }
  }

  materialize(definition: CustomNodeDefinition, position: { x: number; y: number }): void {
    const compatibility = this.compatibility(definition)
    if (!compatibility.available) throw new Error(compatibility.reason)
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
    const parameters = this.store.nodeParametersFor(manifest, { expression: definition.spec.expression })
    const nodeId = this.store.addNode(manifest, position)
    if (nodeId === null) throw new Error(this.componentCompatibility(manifest, parameters).reason)
    if (!this.store.updateNode(nodeId, {
      label: definition.display_name,
      parameters,
    })) throw new Error('The mathematical template could not be applied to the active execution profile.')
  }

  private materializeComposite(
    definition: CustomNodeDefinition,
    position: { x: number; y: number },
  ): void {
    if (definition.spec.type !== 'composite') return
    const resolved = this.resolveComponents(definition.spec.components)
    if (!resolved.length) throw new Error('A composite template must contain at least one built-in component.')
    const portPairs = this.resolvedPortPairs(resolved)
    const nodeIds = resolved.map((component, index) => {
      const nodeId = this.store.addNode(component.manifest, {
        x: position.x + index * 280,
        y: position.y,
      })
      if (nodeId === null) throw new Error(this.componentCompatibility(component.manifest, component.parameters).reason)
      if (!this.store.updateNode(nodeId, {
        label: component.definition.label || (index === 0
          ? definition.display_name
          : component.manifest.display_name),
        parameters: structuredClone(component.parameters),
      })) throw new Error(`Could not apply ${component.manifest.display_name} to the active execution profile.`)
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

  private resolveComponents(components: CompositeNodeComponent[]): ResolvedComponent[] {
    return components.map((component) => {
      const manifest = this.manifest(component.definition_id, component.definition_version)
      return {
        definition: component,
        manifest,
        parameters: this.store.nodeParametersFor(manifest, component.parameters),
      }
    })
  }

  private resolvedPortPairs(resolved: ResolvedComponent[]): { sourceHandle: string; targetHandle: string }[] {
    return resolved.slice(0, -1).map((component, index) =>
      this.compatiblePorts(component, resolved[index + 1]),
    )
  }

  private componentCompatibility(
    manifest: NodeManifest,
    parameters: Record<string, unknown>,
  ): CustomNodeCompatibility {
    const node = this.store.nodeCompatibility(manifest)
    if (!node.available) return { available: false, reason: node.reason ?? 'Node is unavailable for the active execution profile.' }
    if (this.store.canMaterializeNode(manifest, parameters)) return { available: true, reason: '' }
    const profile = this.store.executionProfile()
    const mapping = node.mapping
    for (const definition of manifest.parameters) {
      const value = Object.prototype.hasOwnProperty.call(parameters, definition.id)
        ? parameters[definition.id]
        : effectiveParameterDefault(definition, mapping?.parameters?.[definition.id])
      const compatibility = resolveParameterCompatibility(
        manifest,
        definition,
        value,
        parameters,
        profile.executionBackend,
        profile.kerasEnabled,
      )
      if (!compatibility.valueCompatible) {
        return {
          available: false,
          reason: compatibility.reason ?? `${definition.display_name} is unavailable for the active execution profile.`,
        }
      }
    }
    return { available: false, reason: 'This template contains parameters that are unavailable for the active execution profile.' }
  }

  private compatiblePorts(
    source: ResolvedComponent,
    target: ResolvedComponent,
  ): { sourceHandle: string; targetHandle: string } {
    const outputs = source.manifest.outputs.filter((port) =>
      this.portEnabled(port, source.parameters),
    )
    const inputs = target.manifest.inputs.filter((port) =>
      this.portEnabled(port, target.parameters),
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
