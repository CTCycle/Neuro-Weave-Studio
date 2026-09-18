import { describe, expect, it } from 'vitest'
import { DATASET_CAPABILITIES, arrayFormatFromPath, datasetCapability, formatFromPath, sourceExtensions, sourceMode, tableFormatFromPath } from './dataset-capabilities'

describe('dataset capability registry', () => {
  it('exposes the supported organizations and keeps Auto Detect disabled', () => {
    expect(DATASET_CAPABILITIES.filter((item) => item.enabled).map((item) => item.id)).toEqual([
      'tabular', 'images', 'text', 'time_series', 'array', 'composed',
    ])
    const autoDetect = DATASET_CAPABILITIES.find((item) => item.id === 'auto_detect')
    expect(autoDetect?.enabled).toBe(false)
    expect(autoDetect?.sourceLabel).toContain('Coming later')
  })

  it('centralizes source layout behavior for composed and text sources', () => {
    expect(sourceMode('composed', 0)).toBe('directory')
    expect(sourceMode('composed', 1)).toBe('file')
    expect(sourceMode('text', 0)).toBe('file')
    expect(datasetCapability('time_series').formats).toEqual(['csv', 'tsv', 'parquet'])
  })

  it('derives unambiguous file formats from the selected path', () => {
    expect(tableFormatFromPath('C:\\data\\events.PARQUET')).toBe('parquet')
    expect(tableFormatFromPath('/tmp/events.tsv')).toBe('tsv')
    expect(arrayFormatFromPath('/tmp/features.NPZ')).toBe('npz')
    expect(formatFromPath('array', '/tmp/features.npy')).toBe('npy')
    expect(formatFromPath('images', '/tmp/images')).toBeNull()
  })

  it('derives browser filters from the source capability', () => {
    expect(sourceExtensions('tabular')).toEqual(['csv', 'tsv', 'parquet'])
    expect(sourceExtensions('array')).toEqual(['npy', 'npz'])
    expect(sourceExtensions('array', 2)).toEqual(['npy', 'npz'])
    expect(sourceExtensions('text', 0, { textTable: true })).toEqual(['csv', 'tsv', 'parquet'])
    expect(sourceExtensions('images')).toEqual([])
    expect(sourceMode('text', 0, { textPathKind: 'folder' })).toBe('directory')
  })
})
