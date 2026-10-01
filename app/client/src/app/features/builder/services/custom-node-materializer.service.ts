// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { Inject, Injectable } from '@angular/core'
import type {
  CustomNodeDefinition,
  GraphTemplateComponent,
  GraphTemplateNodeSpec,
} from '../../../domain/custom-node.types'
import { materializedInputs, materializedOutputs } from '../../../domain/ports'
import { effectiveParameterDefault, resolveParameterCompatibility, type NodeManifest, type PortDefinition } from '../../../domain/types'
import { GraphEditorStore } from '../state/graph-editor.store'

interface ResolvedComponent {
  definition: GraphTemplateComponent
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
      if (definition.spec.type === 'graph_template') {
        if (!definition.spec.components.length) {
          return { available: false, reason: 'A graph template must contain at least one built-in component.' }
        }
        const resolved = this.resolveComponents(definition.spec.components)
        for (const component of resolved) {
          const result = this.componentCompatibility(component.manifest, component.parameters)
          if (!result.available) return result
        }
        this.validateGraphTemplate(definition.spec, resolved)
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
      if (definition.spec.type === 'graph_template') {
        this.materializeGraphTemplate(definition, position)
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

  private materializeGraphTemplate(
    definition: CustomNodeDefinition,
    position: { x: number; y: number },
  ): void {
    if (definition.spec.type !== 'graph_template') return
    const resolved = this.resolveComponents(definition.spec.components)
    if (!resolved.length) throw new Error('A graph template must contain at least one built-in component.')
    this.validateGraphTemplate(definition.spec, resolved)
    const byId = new Map(resolved.map((component) => [component.definition.id, component]))
    const nodeIds = new Map<string, string>()
    this.topologicalComponents(definition.spec, resolved).forEach((component, index) => {
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
      nodeIds.set(component.definition.id, nodeId)
    })
    definition.spec.edges.forEach((edge) => {
      const sourceNodeId = nodeIds.get(edge.source_component_id)
      const targetNodeId = nodeIds.get(edge.target_component_id)
      if (!sourceNodeId || !targetNodeId || !byId.has(edge.source_component_id) || !byId.has(edge.target_component_id)) {
        throw new Error('Graph template edge references a component that was not materialized.')
      }
      const result = this.store.connect({
        source: sourceNodeId,
        sourceHandle: edge.source_port_id,
        target: targetNodeId,
        targetHandle: edge.target_port_id,
      })
      if (!result.valid) {
        throw new Error(`Could not connect graph template components: ${result.message}`)
      }
    })
  }

  private resolveComponents(components: GraphTemplateComponent[]): ResolvedComponent[] {
    return components.map((component) => {
      const manifest = this.manifest(component.definition_id, component.definition_version)
      return {
        definition: component,
        manifest,
        parameters: this.store.nodeParametersFor(manifest, component.parameters),
      }
    })
  }

  private validateGraphTemplate(spec: GraphTemplateNodeSpec, resolved: ResolvedComponent[]): void {
    const byId = new Map(resolved.map((component) => [component.definition.id, component]))
    for (const edge of spec.edges) {
      const source = byId.get(edge.source_component_id)
      const target = byId.get(edge.target_component_id)
      if (!source || !target) throw new Error('Graph template edge references an unknown component.')
      const sourcePort = materializedOutputs(source.manifest, source.parameters).find(
        (port) => port.id === edge.source_port_id && this.portEnabled(port, source.parameters),
      )
      const targetPort = materializedInputs(target.manifest, target.parameters).find(
        (port) => port.id === edge.target_port_id && this.portEnabled(port, target.parameters),
      )
      if (!sourcePort || !targetPort) {
        throw new Error(
          `Graph template edge references an unavailable port: ${edge.source_component_id}.${edge.source_port_id} -> ${edge.target_component_id}.${edge.target_port_id}`,
        )
      }
      const expected = targetPort.kind === 'tensor_list'
        ? targetPort.element_kind ?? 'tensor'
        : targetPort.kind
      if (sourcePort.kind !== expected) {
        throw new Error(
          `Graph template edge ${edge.source_component_id}.${edge.source_port_id} -> ${edge.target_component_id}.${edge.target_port_id} has incompatible port kinds.`,
        )
      }
    }
    for (const binding of [...spec.input_bindings, ...spec.output_bindings]) {
      const component = byId.get(binding.component_id)
      if (!component) throw new Error('Graph template binding references an unknown component.')
      const ports = spec.input_bindings.includes(binding)
        ? materializedInputs(component.manifest, component.parameters)
        : materializedOutputs(component.manifest, component.parameters)
      if (!ports.some((port) => port.id === binding.port_id && this.portEnabled(port, component.parameters))) {
        throw new Error(`Graph template binding references an unavailable port: ${binding.component_id}.${binding.port_id}`)
      }
    }
  }

  private topologicalComponents(
    spec: GraphTemplateNodeSpec,
    resolved: ResolvedComponent[],
  ): ResolvedComponent[] {
    const byId = new Map(resolved.map((component) => [component.definition.id, component]))
    const indegree = new Map(resolved.map((component) => [component.definition.id, 0]))
    const adjacency = new Map(resolved.map((component) => [component.definition.id, new Set<string>()]))
    for (const edge of spec.edges) {
      const targets = adjacency.get(edge.source_component_id)
      if (!targets || !indegree.has(edge.target_component_id)) throw new Error('Graph template edge references an unknown component.')
      if (!targets.has(edge.target_component_id)) {
        targets.add(edge.target_component_id)
        indegree.set(edge.target_component_id, (indegree.get(edge.target_component_id) ?? 0) + 1)
      }
    }
    const pending = resolved.filter((component) => indegree.get(component.definition.id) === 0)
    const ordered: ResolvedComponent[] = []
    while (pending.length) {
      const component = pending.shift()
      if (!component) continue
      ordered.push(component)
      for (const targetId of adjacency.get(component.definition.id) ?? []) {
        const nextDegree = (indegree.get(targetId) ?? 0) - 1
        indegree.set(targetId, nextDegree)
        if (nextDegree === 0) {
          const target = byId.get(targetId)
          if (target) pending.push(target)
        }
      }
    }
    if (ordered.length !== resolved.length) throw new Error('Graph template contains a cycle.')
    return ordered
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
