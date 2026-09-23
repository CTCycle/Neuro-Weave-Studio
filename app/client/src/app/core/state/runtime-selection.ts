import type {
  ExecutionBackend,
  ExecutionBackendStatus,
  MemoryPolicy,
  MemoryPolicyKind,
  RuntimeApi,
  RuntimeDevice,
  WorkerConfiguration,
} from '../../domain/types'

export function backendSupportsApi(status: ExecutionBackendStatus, api: RuntimeApi): boolean {
  if (!status.available || !status.runtime_apis?.length) return false
  return status.runtime_apis.some((item) => item.id === api && item.available)
}

export function preferredRuntimeApi(status: ExecutionBackendStatus): RuntimeApi | null {
  if (!status.runtime_apis?.length) return null
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
  const device: RuntimeDevice | undefined = status.devices?.find((item) => item.available)
  if (!device || !device.supported_memory_policies.length) return null
  const memoryPolicy = defaultMemoryPolicy(device.supported_memory_policies[0])
  return {
    api,
    backend: status.id,
    device_id: device.device_id,
    memory_policy: memoryPolicy,
    precision: 'float32',
  }
}

export function isRuntimePreferenceValid(
  status: ExecutionBackendStatus,
  preference: WorkerConfiguration,
): boolean {
  if (!isWorkerConfiguration(preference)) return false
  if (preference.backend !== status.id || !backendSupportsApi(status, preference.api)) return false
  const device = status.devices?.find(
    (item) => item.device_id === preference.device_id && item.available,
  )
  return Boolean(
    device
      && device.supported_memory_policies.includes(preference.memory_policy.kind)
      && isMemoryPolicyValid(preference.memory_policy),
  )
}

export function isWorkerConfiguration(value: unknown): value is WorkerConfiguration {
  if (!isRecord(value)) return false
  if (!isRuntimeApi(value['api']) || !isExecutionBackend(value['backend']) || typeof value['device_id'] !== 'string') return false
  if (value['precision'] !== 'float32' && value['precision'] !== 'mixed_float16' && value['precision'] !== 'mixed_bfloat16') return false
  const memoryPolicy = value['memory_policy']
  if (!isRecord(memoryPolicy) || !isMemoryPolicyKind(memoryPolicy['kind'])) return false
  return isMemoryPolicyValid(memoryPolicy as unknown as MemoryPolicy)
}

export function defaultMemoryPolicy(kind: MemoryPolicyKind): MemoryPolicy {
  if (kind === 'tensorflow_limit_mb') return { kind, limit_mb: 1024 }
  if (kind === 'jax_preallocate_fraction' || kind === 'pytorch_fraction') return { kind, fraction: 0.5 }
  return { kind }
}

export function isMemoryPolicyValid(policy: MemoryPolicy): boolean {
  if (typeof policy !== 'object' || policy === null) return false
  if (policy.kind === 'tensorflow_limit_mb') {
    return Number.isInteger(policy.limit_mb) && (policy.limit_mb ?? 0) >= 1 && policy.fraction == null
  }
  if (policy.kind === 'jax_preallocate_fraction' || policy.kind === 'pytorch_fraction') {
    return typeof policy.fraction === 'number' && policy.fraction > 0 && policy.fraction <= 1 && policy.limit_mb == null
  }
  if (policy.kind === 'default' || policy.kind === 'tensorflow_growth' || policy.kind === 'jax_no_preallocation' || policy.kind === 'jax_platform_allocator') {
    return policy.limit_mb == null && policy.fraction == null
  }
  return false
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isRuntimeApi(value: unknown): value is RuntimeApi {
  return value === 'keras' || value === 'native'
}

function isExecutionBackend(value: unknown): value is ExecutionBackend {
  return value === 'tensorflow' || value === 'pytorch' || value === 'jax'
}

function isMemoryPolicyKind(value: unknown): value is MemoryPolicyKind {
  return value === 'default'
    || value === 'tensorflow_growth'
    || value === 'tensorflow_limit_mb'
    || value === 'jax_no_preallocation'
    || value === 'jax_preallocate_fraction'
    || value === 'jax_platform_allocator'
    || value === 'pytorch_fraction'
}
