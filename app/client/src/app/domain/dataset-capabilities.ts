import type { DatasetType } from './dataset.types'

export type DatasetTypeChoice = DatasetType | 'auto_detect'
export type DatasetSourceMode = 'file' | 'directory'

export interface DatasetTypeCapability {
  id: DatasetTypeChoice
  label: string
  description: string
  sourceLabel: string
  enabled: boolean
  formats: string[]
  sourceModes: DatasetSourceMode[]
  futureFeatures: string[]
}

export const DATASET_CAPABILITIES: readonly DatasetTypeCapability[] = [
  {
    id: 'tabular',
    label: 'Tabular',
    description: 'Rows and columns from CSV, TSV, or Parquet.',
    sourceLabel: 'One table file',
    enabled: true,
    formats: ['csv', 'tsv', 'parquet'],
    sourceModes: ['file'],
    futureFeatures: ['missing values', 'normalization', 'categorical encoding'],
  },
  {
    id: 'images',
    label: 'Images',
    description: 'Image folders, with optional folder labels or a label table.',
    sourceLabel: 'Folder or folder + table',
    enabled: true,
    formats: ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tif', 'tiff', 'webp'],
    sourceModes: ['directory', 'file'],
    futureFeatures: ['resize', 'normalization', 'augmentation'],
  },
  {
    id: 'text',
    label: 'Text',
    description: 'Text columns or plain-text files and folders.',
    sourceLabel: 'Table, file, or folder',
    enabled: true,
    formats: ['csv', 'tsv', 'parquet', 'txt', 'text', 'md', 'jsonl'],
    sourceModes: ['file', 'directory'],
    futureFeatures: ['tokenization', 'sequence handling'],
  },
  {
    id: 'time_series',
    label: 'Time series',
    description: 'Sequential tabular records with optional time and group fields.',
    sourceLabel: 'One table file',
    enabled: true,
    formats: ['csv', 'tsv', 'parquet'],
    sourceModes: ['file'],
    futureFeatures: ['windowing', 'sequence generation'],
  },
  {
    id: 'array',
    label: 'Array / tensor',
    description: 'NumPy NPY sources or NPZ collections inspected from headers.',
    sourceLabel: 'One or more array files',
    enabled: true,
    formats: ['npy', 'npz'],
    sourceModes: ['file'],
    futureFeatures: ['shape-aware loading', 'batching'],
  },
  {
    id: 'composed',
    label: 'Composed / multimodal',
    description: 'Images or text combined with table features or targets.',
    sourceLabel: 'Folder + table',
    enabled: true,
    formats: ['csv', 'tsv', 'parquet', 'txt', 'text', 'md', 'jsonl'],
    sourceModes: ['directory', 'file'],
    futureFeatures: ['modality alignment', 'joint transforms'],
  },
  {
    id: 'auto_detect',
    label: 'Auto Detect',
    description: 'Automatic source interpretation is planned for a later release.',
    sourceLabel: 'Coming later',
    enabled: false,
    formats: [],
    sourceModes: [],
    futureFeatures: [],
  },
]

export function datasetCapability(type: DatasetType): DatasetTypeCapability {
  const capability = DATASET_CAPABILITIES.find((item) => item.id === type)
  if (!capability || !capability.enabled || capability.id === 'auto_detect') {
    throw new Error(`Unsupported dataset type: ${type}`)
  }
  return capability
}

export function datasetTypeLabel(type: DatasetType): string {
  return datasetCapability(type).label
}

export function sourceExtensions(type: DatasetType, sourceIndex = 0): string[] {
  const capability = datasetCapability(type)
  if (type === 'images' && sourceIndex === 0) return []
  if (type === 'text' && sourceIndex === 0) return capability.formats
  if (type === 'composed' && sourceIndex === 0) return []
  return capability.formats
}

export function sourceMode(type: DatasetType, sourceIndex = 0): DatasetSourceMode {
  if (type === 'images' || type === 'composed') return sourceIndex === 0 ? 'directory' : 'file'
  if (type === 'text') return sourceIndex === 0 ? 'file' : 'file'
  return 'file'
}
