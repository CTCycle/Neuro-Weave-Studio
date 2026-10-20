// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

export interface ApplicationSettings {
  app_name: string
  app_version: string
  cors_origins: string
  log_level: string
  log_file: string
  export_root: string
  ml_job_root: string
  tensor_inspection_root: string
  tensor_inspection_ttl_seconds: number
  ml_worker_timeout_seconds: number
  runtime_discovery_timeout_seconds: number
  ml_cancel_grace_seconds: number
  ml_terminate_grace_seconds: number
  ml_job_history_limit: number
  runtime_discovery_path: string
}

export interface DatabaseSettings {
  backend: 'sqlite'
  schema_version: 1
  sqlite_path: string
  connect_timeout_seconds: number
}

export interface DatabaseStatus {
  backend: string
  sqlite_path: string | null
  connect_timeout_seconds: number | null
  available: boolean
  switching: boolean
  message: string | null
}

export interface DatabaseTestResult {
  success: boolean
  backend: 'sqlite'
  sqlite_path: string
  message: string
}

export interface DatabaseSwitchResult {
  status: DatabaseStatus
  settings: ApplicationSettings
}
