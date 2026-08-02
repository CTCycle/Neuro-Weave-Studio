import { describe, expect, it } from 'vitest'
import { wouldCreateCycle } from './graphStore'
import type { EditorEdge, EditorNode } from '../domain/types'

const node = (id: string): EditorNode => ({
  id,
  type: 'modelNode',
  position: { x: 0, y: 0 },
  data: {
    manifest: { id: 'layer.dense', version: 1, display_name: 'Dense', category: 'core', description: '', inputs: [], outputs: [], parameters: [], shape_rule: { kind: 'dense', config: {} }, frameworks: { keras: { supported: true }, tensorflow_keras: { supported: true }, pytorch: { supported: true } }, ui: { icon: 'Box', accent: 'violet', width: 250, height: 120 }, deprecated: false, documentation_ref: '' },
    parameters: {}, label: id, diagnostics: [],
  },
})

describe('wouldCreateCycle', () => {
  it('rejects an edge that closes a cycle', () => {
    const nodes = [node('a'), node('b'), node('c')]
    const edges: EditorEdge[] = [{ id: 'ab', source: 'a', target: 'b' }, { id: 'bc', source: 'b', target: 'c' }]
    expect(wouldCreateCycle(nodes, edges, 'c', 'a')).toBe(true)
    expect(wouldCreateCycle(nodes, edges, 'a', 'c')).toBe(false)
  })
})
