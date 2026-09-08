import type {
  ExecutionBackend,
  ExecutionBackendStatus,
  RuntimeApi,
  RuntimeDevice,
  WorkerConfiguration,
} from '../../domain/types'

export function backendSupportsApi(status: ExecutionBackendStatus, api: RuntimeApi): boolean {
  if (!status.available) return false
  if (!status.runtime_apis?.length) return api === 'keras'
  return status.runtime_apis.some((item) => item.id === api && item.available)
}

export function preferredRuntimeApi(status: ExecutionBackendStatus): RuntimeApi | null {
  if (!status.runtime_apis?.length) return status.available ? 'keras' : null
  const available = status.runtime_apis.filter((item) => item.available)
  return (available.find((item) => item.id === 'keras') ?? available[0])?.id ?? null
}

export function preferredBackendForApi(
  statuses: ExecutionBackendStatus[],
  api: RuntimeApi,
): ExecutionBackend | null {
  const available = statuses.filter((item) => backendSupportsApi(item, api))
  return (available.length === 1 ? available[0] : available.find((item) => item.id === 'pytorch') ?? available[0])?.id ?? null
}

export function preferredBackend(statuses: ExecutionBackendStatus[]): ExecutionBackend | null {
  const available = statuses.filter((item) => item.available && preferredRuntimeApi(item) !== null)
  return (available.length === 1 ? available[0] : available.find((item) => item.id === 'pytorch') ?? available[0])?.id ?? null
}

export function defaultRuntimePreference(status: ExecutionBackendStatus): WorkerConfiguration | null {
  const api = preferredRuntimeApi(status)
  if (!api) return null
  const device: RuntimeDevice = status.devices?.find((item) => item.available) ?? status.devices?.[0] ?? {
    device_id: 'cpu',
    lease_key: 'cpu',
    kind: 'cpu',
    name: 'CPU',
    accelerator: 'cpu',
    available: true,
    supported_memory_policies: ['default'],
  }
  return {
    api,
    backend: status.id,
    device_id: device.device_id,
    memory_policy: { kind: device.supported_memory_policies[0] ?? 'default' },
    precision: 'float32',
  }
}
