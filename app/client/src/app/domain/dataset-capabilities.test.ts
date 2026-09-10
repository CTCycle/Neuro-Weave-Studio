import { describe, expect, it } from 'vitest'
import { DATASET_CAPABILITIES, datasetCapability, sourceMode } from './dataset-capabilities'

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
})
