// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

export interface FilesystemEntry {
  name: string
  path: string
  is_dir: boolean
  size_bytes: number | null
  suffix: string | null
}

export interface FilesystemBrowseResponse {
  current_path: string
  parent_path: string | null
  items: FilesystemEntry[]
  roots: FilesystemEntry[]
  truncated: boolean
}
