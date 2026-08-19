"""Fresh-interpreter ML worker.

This module deliberately imports no ML framework at module import time.  The
worker reads its immutable configuration, sets startup environment variables,
and only then imports Keras and its selected execution backend.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, cast

from common.process_control import process_rss_bytes
from domain.enums import ExecutionBackend
from domain.graph import ModelGraph
from domain.runtime import MemoryPolicyKind, RuntimeApi, WorkerConfiguration

###############################################################################
class WorkerCancelled(RuntimeError):
    pass

###############################################################################
class WorkerContext:
    def __init__(self, job_dir: Path, config: WorkerConfiguration) -> None:
        self.job_dir = job_dir
        self.config = config
        self.cancel_file = job_dir / "cancel.requested"
        self.status_file = job_dir / "status.json"
        self.started = time.time()

    def checkpoint(self, phase: str, message: str, percent: int | None = None) -> None:
        self.write_status(
            phase=phase,
            message=message,
            percent=percent,
            state="running",
        )
        if self.cancel_file.exists():
            self.write_status(
                phase="cancelling",
                message="Cancellation observed at a safe checkpoint",
                percent=percent,
                state="stop_requested",
            )
            raise WorkerCancelled("Worker cancellation requested")

    def write_status(
        self,
        *,
        phase: str,
        message: str,
        state: str,
        percent: int | None = None,
        **extra: object,
    ) -> None:
        payload: dict[str, object] = {
            "state": state,
            "phase": phase,
            "message": message,
            "percent": percent,
            "pid": os.getpid(),
            "updated_at": time.time(),
            "process_rss_bytes": process_rss_bytes(os.getpid()),
            "backend": self.config.backend.value,
            "api": self.config.api.value,
            "device_id": self.config.device_id,
            **extra,
        }
        temporary = self.status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.status_file)

###############################################################################
def _read_request(job_dir: Path) -> tuple[WorkerContext, dict[str, Any]]:
    payload = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    try:
        (job_dir / "request.json").unlink()
    except OSError:
        pass
    config = WorkerConfiguration.model_validate(payload["worker"])
    return WorkerContext(job_dir, config), cast(dict[str, Any], payload)

###############################################################################
def _set_startup_environment(config: WorkerConfiguration) -> None:
    if config.api is RuntimeApi.NATIVE:
        raise RuntimeError("Native framework inference is not implemented")
    backend_value = "torch" if config.backend is ExecutionBackend.PYTORCH else config.backend.value
    os.environ["KERAS_BACKEND"] = backend_value

    if config.device_id == "auto":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    elif config.device_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if config.backend is ExecutionBackend.JAX:
            os.environ["JAX_PLATFORMS"] = "cpu"
    elif config.device_id.startswith(("cuda:", "rocm:")):
        index = config.device_id.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = index
    elif config.device_id.startswith("mps:"):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    elif config.device_id.startswith("xpu:"):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        raise RuntimeError(f"Unsupported worker device '{config.device_id}'")

    policy = config.memory_policy.kind
    if policy is MemoryPolicyKind.JAX_NO_PREALLOCATION:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    elif policy is MemoryPolicyKind.JAX_PREALLOCATE_FRACTION:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.memory_policy.fraction)
    elif policy is MemoryPolicyKind.JAX_PLATFORM_ALLOCATOR:
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

###############################################################################
def _configure_startup_backend(config: WorkerConfiguration) -> None:
    """Apply allocator and visibility settings before importing Keras."""

    if config.backend is ExecutionBackend.TENSORFLOW:
        tensorflow = importlib.import_module("tensorflow")
        gpus = list(tensorflow.config.list_physical_devices("GPU"))
        if config.device_id == "cpu":
            tensorflow.config.set_visible_devices([], "GPU")
        elif gpus and config.device_id != "auto":
            visible_gpu = gpus[0]
            tensorflow.config.set_visible_devices(visible_gpu, "GPU")
            policy = config.memory_policy.kind
            if policy is MemoryPolicyKind.TENSORFLOW_GROWTH:
                tensorflow.config.experimental.set_memory_growth(visible_gpu, True)
            elif policy is MemoryPolicyKind.TENSORFLOW_LIMIT_MB:
                tensorflow.config.set_logical_device_configuration(
                    visible_gpu,
                    [tensorflow.config.LogicalDeviceConfiguration(memory_limit=config.memory_policy.limit_mb)],
                )
    elif config.backend is ExecutionBackend.PYTORCH:
        torch = importlib.import_module("torch")
        if config.device_id.startswith(("cuda:", "rocm:")) and bool(torch.cuda.is_available()):
            torch.cuda.set_device(0)
            if config.memory_policy.kind is MemoryPolicyKind.PYTORCH_FRACTION:
                fraction = config.memory_policy.fraction
                if fraction is None:
                    raise RuntimeError("pytorch_fraction requires a fraction")
                torch.cuda.set_per_process_memory_fraction(
                    float(fraction), device=0
                )
    elif config.backend is ExecutionBackend.JAX:
        importlib.import_module("jax")

###############################################################################
def _configure_imported_backend(config: WorkerConfiguration, keras: Any) -> dict[str, Any]:
    """Read the selected backend after startup configuration without allocating it."""

    del config
    return {"backend": str(keras.backend.backend())}

###############################################################################
def _device_inventory(backend: ExecutionBackend, keras: Any) -> list[dict[str, object]]:
    devices: list[dict[str, object]] = [
        {
            "device_id": "cpu",
            "lease_key": "cpu",
            "kind": "cpu",
            "index": None,
            "name": "CPU",
            "accelerator": "cpu",
            "supported_memory_policies": [MemoryPolicyKind.DEFAULT.value],
        }
    ]
    if backend is ExecutionBackend.PYTORCH:
        torch = importlib.import_module("torch")
        if bool(torch.cuda.is_available()):
            for index in range(int(torch.cuda.device_count())):
                devices.append(
                    {
                        "device_id": f"cuda:{index}",
                        "lease_key": f"cuda:{index}",
                        "kind": "cuda",
                        "index": index,
                        "name": str(torch.cuda.get_device_name(index)),
                        "accelerator": "cuda",
                        "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                        "supported_memory_policies": [
                            MemoryPolicyKind.DEFAULT.value,
                            MemoryPolicyKind.PYTORCH_FRACTION.value,
                        ],
                    }
                )
        if hasattr(torch, "xpu") and bool(torch.xpu.is_available()):
            devices.append(
                {
                    "device_id": "xpu:0",
                    "lease_key": "xpu:0",
                    "kind": "xpu",
                    "index": 0,
                    "name": str(torch.xpu.get_device_name(0)),
                    "accelerator": "xpu",
                    "supported_memory_policies": [MemoryPolicyKind.DEFAULT.value],
                }
            )
        if hasattr(torch.backends, "mps") and bool(torch.backends.mps.is_available()):
            devices.append(
                {
                    "device_id": "mps:0",
                    "lease_key": "mps:0",
                    "kind": "mps",
                    "index": 0,
                    "name": "Apple Metal Performance Shaders",
                    "accelerator": "mps",
                    "supported_memory_policies": [MemoryPolicyKind.DEFAULT.value],
                }
            )
    elif backend is ExecutionBackend.TENSORFLOW:
        tensorflow = importlib.import_module("tensorflow")
        for index, device in enumerate(tensorflow.config.list_physical_devices("GPU")):
            devices.append(
                {
                    "device_id": f"cuda:{index}",
                    "lease_key": f"cuda:{index}",
                    "kind": "cuda",
                    "index": index,
                    "name": str(device.name),
                    "accelerator": "cuda",
                    "supported_memory_policies": [
                        MemoryPolicyKind.DEFAULT.value,
                        MemoryPolicyKind.TENSORFLOW_GROWTH.value,
                        MemoryPolicyKind.TENSORFLOW_LIMIT_MB.value,
                    ],
                }
            )
    elif backend is ExecutionBackend.JAX:
        jax = importlib.import_module("jax")
        jax_devices = list(jax.devices())
        devices = []
        for index, device in enumerate(jax_devices):
            platform = str(device.platform)
            if platform == "cpu":
                device_id = "cpu"
                accelerator = "cpu"
            else:
                accelerator = "rocm" if "rocm" in platform else "cuda" if platform in {"gpu", "cuda"} else platform
                device_id = f"{accelerator}:{index}"
            devices.append(
                {
                    "device_id": device_id,
                    "lease_key": device_id,
                    "kind": platform,
                    "index": None if platform == "cpu" else index,
                    "name": str(device),
                    "accelerator": accelerator,
                    "supported_memory_policies": [
                        MemoryPolicyKind.DEFAULT.value,
                        MemoryPolicyKind.JAX_NO_PREALLOCATION.value,
                        MemoryPolicyKind.JAX_PREALLOCATE_FRACTION.value,
                        MemoryPolicyKind.JAX_PLATFORM_ALLOCATOR.value,
                    ],
                }
            )
    return devices

###############################################################################
def _memory_metrics(backend: ExecutionBackend, keras: Any) -> dict[str, int | None]:
    metrics: dict[str, int | None] = {
        "framework_allocated_bytes": None,
        "framework_reserved_bytes": None,
        "framework_peak_bytes": None,
        "device_used_bytes": None,
        "device_total_bytes": None,
    }
    try:
        if backend is ExecutionBackend.PYTORCH:
            torch = importlib.import_module("torch")
            if bool(torch.cuda.is_available()):
                metrics["framework_allocated_bytes"] = int(torch.cuda.memory_allocated(0))
                metrics["framework_reserved_bytes"] = int(torch.cuda.memory_reserved(0))
                metrics["framework_peak_bytes"] = int(torch.cuda.max_memory_allocated(0))
                metrics["device_used_bytes"] = int(torch.cuda.device_memory_used(0))
                metrics["device_total_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
        elif backend is ExecutionBackend.TENSORFLOW:
            tensorflow = importlib.import_module("tensorflow")
            if tensorflow.config.list_logical_devices("GPU"):
                info = tensorflow.config.experimental.get_memory_info("GPU:0")
                metrics["framework_allocated_bytes"] = int(info.get("current", 0))
                metrics["framework_peak_bytes"] = int(info.get("peak", 0))
        elif backend is ExecutionBackend.JAX:
            jax = importlib.import_module("jax")
            devices = list(jax.devices())
            if devices and hasattr(devices[0], "memory_stats"):
                stats = devices[0].memory_stats() or {}
                metrics["framework_allocated_bytes"] = int(
                    stats.get("bytes_in_use", 0) or 0
                )
                metrics["framework_reserved_bytes"] = int(
                    stats.get("bytes_reserved", 0) or 0
                )
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return metrics
    return metrics

###############################################################################
def _cleanup(backend: ExecutionBackend, keras: Any) -> None:
    try:
        if backend is ExecutionBackend.PYTORCH:
            torch = importlib.import_module("torch")
            if bool(torch.cuda.is_available()):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        elif backend is ExecutionBackend.JAX:
            jax = importlib.import_module("jax")
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
        if hasattr(keras.backend, "clear_session"):
            keras.backend.clear_session()
    except (AttributeError, ImportError, RuntimeError, TypeError):
        pass
    finally:
        gc.collect()

###############################################################################
def _run_job(context: WorkerContext, payload: dict[str, Any]) -> dict[str, object]:
    global _IMPORTED_KERAS
    _IMPORTED_KERAS = None
    try:
        return _run_job_inner(context, payload)
    finally:
        if _IMPORTED_KERAS is not None:
            _cleanup(context.config.backend, _IMPORTED_KERAS)
        _IMPORTED_KERAS = None

###############################################################################
def _run_job_inner(context: WorkerContext, payload: dict[str, Any]) -> dict[str, object]:
    global _IMPORTED_KERAS
    context.checkpoint("importing", "Configuring the selected backend", 5)
    _set_startup_environment(context.config)
    context.checkpoint("importing", "Importing Keras and the selected backend", 15)
    if context.config.api is not RuntimeApi.KERAS:
        raise RuntimeError("Native framework inference is not implemented")
    _configure_startup_backend(context.config)
    from models import keras_worker

    import keras  # pyright: ignore[reportMissingImports]
    _IMPORTED_KERAS = keras

    keras_worker.set_checkpoint(
        lambda: context.checkpoint("model_loading", "Building the model", 35)
    )
    context.checkpoint("initializing", "Applying device and memory policy", 25)
    _configure_imported_backend(context.config, keras)
    operation = payload["operation"]
    operation_type = str(operation.get("type"))
    graph = ModelGraph.model_validate(operation.get("graph")) if operation_type == "smoke_inference" else ModelGraph.model_validate(operation["graph"])
    context.checkpoint("model_loading", "Constructing the configured model", 40)
    capabilities = keras_worker._backend_capabilities()

    if operation_type == "smoke_inference":
        result = keras_worker._smoke_test(graph)
        context.checkpoint("publishing", "Forward pass completed", 90)
        result.update(_memory_metrics(context.config.backend, keras))
        return result

    if operation_type == "keras_export":
        destination = context.job_dir / "artifact.keras"
        with keras_worker._device_scope(capabilities):
            model = keras_worker._build_model(graph)
            context.checkpoint("compiling", "Compiling the Keras artifact", 55)
            from models.keras_runtime import compile_model

            compile_model(model, graph)
            context.checkpoint("saving", "Saving the temporary Keras artifact", 75)
            destination.parent.mkdir(parents=True, exist_ok=True)
            model.save(destination)
            keras_worker._verify_saved_model(destination, len(model.outputs))
        context.checkpoint("publishing", "Artifact verification completed", 95)
        result = {
            "artifact_name": destination.name,
            "parameter_count": int(model.count_params()),
            "accelerator": capabilities.get("accelerator", "cpu"),
            "device_name": capabilities.get("device_name", "CPU"),
        }
        result.update(_memory_metrics(context.config.backend, keras))
        return result
    raise RuntimeError(f"Unsupported operation '{operation_type}'")


_IMPORTED_KERAS: Any | None = None

###############################################################################
def _probe(backend: ExecutionBackend) -> None:
    config = WorkerConfiguration(backend=backend, device_id="auto")
    _set_startup_environment(config)
    _configure_startup_backend(config)
    import keras  # pyright: ignore[reportMissingImports]

    _configure_imported_backend(config, keras)
    payload = {
        "backend": keras.backend.backend(),
        "available": True,
        "devices": _device_inventory(backend, keras),
        "accelerator": "cpu",
        "device_name": "CPU",
    }
    for device in payload["devices"]:
        if isinstance(device, dict) and device.get("accelerator") != "cpu":
            payload["accelerator"] = device.get("accelerator")
            payload["device_name"] = device.get("name")
            break
    print(json.dumps(payload), flush=True)

###############################################################################
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--backend", choices=[item.value for item in ExecutionBackend])
    parser.add_argument("--job-dir", type=Path)
    return parser.parse_args()

###############################################################################
def main() -> None:
    arguments = _parse_arguments()
    if arguments.probe:
        if arguments.backend is None:
            raise RuntimeError("--backend is required for a probe")
        try:
            _probe(ExecutionBackend(arguments.backend))
        except Exception as exc:  # noqa: BLE001 - probe errors become structured parent state
            reason = "dependency_missing" if isinstance(exc, ModuleNotFoundError) else "probe_failed"
            print(json.dumps({"available": False, "reason_code": reason, "message": str(exc)}))
            raise
        return
    if arguments.job_dir is None:
        raise RuntimeError("--job-dir is required")
    context, payload = _read_request(arguments.job_dir)
    context.write_status(
        phase="starting",
        message="Worker process started",
        state="running",
    )
    try:
        result = _run_job(context, payload)
        context.write_status(
            phase="finished",
            message="Worker completed successfully",
            state="succeeded",
            percent=100,
            result=result,
        )
    except WorkerCancelled as exc:
        context.write_status(
            phase="cancelled",
            message=str(exc),
            state="terminated",
            error={"code": "cancelled", "message": str(exc)},
        )
        raise SystemExit(20) from exc
    except Exception as exc:  # noqa: BLE001 - worker boundary serializes all failures
        context.write_status(
            phase="failed",
            message=str(exc) or "Worker failed without a diagnostic",
            state="failed",
            error={
                "code": "worker_failed",
                "message": str(exc) or "Worker failed without a diagnostic",
                "traceback": traceback.format_exc(limit=12),
            },
        )
        raise

###############################################################################
if __name__ == "__main__":
    main()
