import type { ArraySource, DatasetSource, DatasetType, TableSource, TextSource } from './dataset.types'

export type DatasetTypeChoice = DatasetType | 'auto_detect'
export type DatasetSourceMode = 'file' | 'directory'
export type DatasetSourceKind = DatasetSource['type']

export interface DatasetSourceCapability {
  kind: DatasetSourceKind
  label: string
  mode: DatasetSourceMode
  formats: readonly string[]
  optional?: boolean
}

export interface DatasetTypeCapability {
  id: DatasetTypeChoice
  label: string
  description: string
  sourceLabel: string
  enabled: boolean
  formats: string[]
  sourceModes: DatasetSourceMode[]
  sourceSlots: readonly DatasetSourceCapability[]
  futureFeatures: string[]
}

const TABLE_FORMATS = ['csv', 'tsv', 'parquet'] as const
const IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tif', 'tiff', 'webp'] as const
const TEXT_FORMATS = ['txt', 'text', 'md', 'jsonl'] as const
const TABLE_SOURCE: DatasetSourceCapability = { kind: 'table', label: 'Table file', mode: 'file', formats: TABLE_FORMATS }
const IMAGE_SOURCE: DatasetSourceCapability = { kind: 'image_folder', label: 'Image folder', mode: 'directory', formats: [] }
const TEXT_SOURCE: DatasetSourceCapability = { kind: 'text', label: 'Text file or folder', mode: 'file', formats: TEXT_FORMATS }
const ARRAY_SOURCE: DatasetSourceCapability = { kind: 'array', label: 'Array file', mode: 'file', formats: ['npy', 'npz'] }

export const DATASET_CAPABILITIES: readonly DatasetTypeCapability[] = [
  {
    id: 'tabular', label: 'Tabular', description: 'Rows and columns from CSV, TSV, or Parquet.', sourceLabel: 'One table file', enabled: true,
    formats: [...TABLE_FORMATS], sourceModes: ['file'], sourceSlots: [TABLE_SOURCE], futureFeatures: ['missing values', 'normalization', 'categorical encoding'],
  },
  {
    id: 'images', label: 'Images', description: 'Image folders, with optional folder labels or a label table.', sourceLabel: 'Folder or folder + table', enabled: true,
    formats: [...IMAGE_FORMATS], sourceModes: ['directory', 'file'], sourceSlots: [IMAGE_SOURCE, TABLE_SOURCE], futureFeatures: ['resize', 'normalization', 'augmentation'],
  },
  {
    id: 'text', label: 'Text', description: 'Text columns or plain-text files and folders.', sourceLabel: 'Table, file, or folder', enabled: true,
    formats: [...TABLE_FORMATS, ...TEXT_FORMATS], sourceModes: ['file', 'directory'], sourceSlots: [TEXT_SOURCE], futureFeatures: ['tokenization', 'sequence handling'],
  },
  {
    id: 'time_series', label: 'Time series', description: 'Sequential tabular records with optional time and group fields.', sourceLabel: 'One table file', enabled: true,
    formats: [...TABLE_FORMATS], sourceModes: ['file'], sourceSlots: [TABLE_SOURCE], futureFeatures: ['windowing', 'sequence generation'],
  },
  {
    id: 'array', label: 'Array / tensor', description: 'NumPy NPY sources or NPZ collections inspected from headers.', sourceLabel: 'One or more array files', enabled: true,
    formats: ['npy', 'npz'], sourceModes: ['file'], sourceSlots: [ARRAY_SOURCE], futureFeatures: ['shape-aware loading', 'batching'],
  },
  {
    id: 'composed', label: 'Composed / multimodal', description: 'Images or text combined with table features or targets.', sourceLabel: 'Folder + table', enabled: true,
    formats: [...TABLE_FORMATS, ...TEXT_FORMATS], sourceModes: ['directory', 'file'], sourceSlots: [IMAGE_SOURCE, TABLE_SOURCE], futureFeatures: ['modality alignment', 'joint transforms'],
  },
  {
    id: 'auto_detect', label: 'Auto Detect', description: 'Automatic source interpretation is planned for a later release.', sourceLabel: 'Coming later', enabled: false,
    formats: [], sourceModes: [], sourceSlots: [], futureFeatures: [],
  },
]

export interface DatasetCapabilityOptions {
  sourceKind?: DatasetSourceKind
  textTable?: boolean
  textPathKind?: TextSource['path_kind']
  composedMode?: 'images' | 'text'
}

export function datasetCapability(type: DatasetType): DatasetTypeCapability {
  const capability = DATASET_CAPABILITIES.find((item) => item.id === type)
  if (!capability || !capability.enabled || capability.id === 'auto_detect') throw new Error(`Unsupported dataset type: ${type}`)
  return capability
}

export function datasetTypeLabel(type: DatasetType): string { return datasetCapability(type).label }

export function sourceCapability(type: DatasetType, sourceIndex = 0, options: DatasetCapabilityOptions = {}): DatasetSourceCapability {
  const capability = datasetCapability(type)
  if (type === 'array') return ARRAY_SOURCE
  if (type === 'text' && options.textTable) return TABLE_SOURCE
  if (type === 'composed' && sourceIndex === 0 && options.composedMode === 'text') return TEXT_SOURCE
  const source = capability.sourceSlots[sourceIndex]
  if (!source) throw new Error(`No source capability for ${type} source ${sourceIndex}`)
  if (options.sourceKind && source.kind !== options.sourceKind) {
    const matching = capability.sourceSlots.find((item) => item.kind === options.sourceKind)
    if (matching) return matching
  }
  return source
}

export function sourceExtensions(type: DatasetType, sourceIndex = 0, options: DatasetCapabilityOptions = {}): string[] {
  return [...sourceCapability(type, sourceIndex, options).formats]
}

export function sourceMode(type: DatasetType, sourceIndex = 0, options: DatasetCapabilityOptions = {}): DatasetSourceMode {
  const source = sourceCapability(type, sourceIndex, options)
  if (source.kind === 'text' && options.textPathKind === 'folder') return 'directory'
  return source.mode
}

export function tableFormatFromPath(path: string): TableSource['format'] | null {
  const extension = extensionFromPath(path)
  return extension === 'csv' || extension === 'tsv' || extension === 'parquet' ? extension : null
}

export function arrayFormatFromPath(path: string): ArraySource['format'] | null {
  const extension = extensionFromPath(path)
  return extension === 'npy' || extension === 'npz' ? extension : null
}

export function formatFromPath(type: DatasetType, path: string, sourceIndex = 0, options: DatasetCapabilityOptions = {}): TableSource['format'] | ArraySource['format'] | null {
  const source = sourceCapability(type, sourceIndex, options)
  if (source.kind === 'table') return tableFormatFromPath(path)
  if (source.kind === 'array') return arrayFormatFromPath(path)
  return null
}

export function extensionFromPath(path: string): string {
  const normalized = path.trim().replace(/[\\/]+$/, '')
  const filename = normalized.split(/[\\/]/).at(-1) ?? normalized
  const index = filename.lastIndexOf('.')
  return index >= 0 ? filename.slice(index + 1).toLowerCase() : ''
}
