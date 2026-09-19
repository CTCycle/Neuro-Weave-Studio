# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportPrivateUsage=false

"""Fresh-interpreter dispatcher for one isolated execution backend."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from domain.enums import ExecutionBackend
from domain.runtime import RuntimeApi, WorkerConfiguration
from models.worker_backend import (
    BackendWorker,
    configure_startup_environment,
    load_backend_worker,
)
from models.worker_core import WorkerCancelled, WorkerContext, read_request, run_job

###############################################################################
def _probe(backend: ExecutionBackend, api: RuntimeApi) -> None:
    config = WorkerConfiguration(backend=backend, api=api, device_id="auto")
    configure_startup_environment(config)
    worker = load_backend_worker(backend)
    try:
        worker.configure(config)
        payload: dict[str, object] = {
            "api": api.value,
            "backend": worker.capabilities().get("backend"),
            "available": True,
            "devices": worker.device_inventory(),
            "accelerator": "cpu",
            "device_name": "CPU",
        }
        capabilities = worker.capabilities()
        payload["accelerator"] = capabilities.get("accelerator", "cpu")
        payload["device_name"] = capabilities.get("device_name", "CPU")
        payload["native_compilers"] = worker.native_compilation_capabilities()
        print(json.dumps(payload), flush=True)
    finally:
        worker.cleanup()

###############################################################################
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--backend", choices=[item.value for item in ExecutionBackend])
    parser.add_argument(
        "--api", choices=[item.value for item in RuntimeApi], default=RuntimeApi.KERAS.value
    )
    parser.add_argument("--job-dir", type=Path)
    return parser.parse_args()

###############################################################################
def _run_job_process(job_dir: Path) -> None:
    context: WorkerContext
    payload: dict[str, object]
    context, payload = read_request(job_dir)
    configure_startup_environment(context.config)
    worker: BackendWorker | None = None
    try:
        worker = load_backend_worker(context.config.backend)
        context.write_status(
            phase="starting",
            message="Worker process started",
            state="running",
        )
        result = run_job(context, payload, worker)
        context.write_status(
            phase="completed",
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
    except Exception as exc:
        operation = payload.get("operation", {})
        operation_type = operation.get("type") if isinstance(operation, dict) else None
        error_code = "worker_failed"
        if operation_type == "tensor_inspection":
            # Preserve the stable inspection contract when a backend rejects
            # a dtype, shape, or capture strategy.  Ordinary framework errors
            # remain an explicit execution failure rather than an opaque
            # generic worker failure.
            detail = str(exc) or "Tensor inspection execution failed"
            known_codes = {
                "capture_unsupported",
                "dynamic_dimension_unresolved",
                "execution_failed",
                "node_uninspectable",
                "payload_limit",
                "unsupported_dtype",
            }
            error_code = next(
                (code for code in known_codes if detail.startswith(f"{code}:")),
                "execution_failed",
            )
        context.write_status(
            phase="failed",
            message=str(exc) or "Worker failed without a diagnostic",
            state="failed",
            error={
                "code": error_code,
                "message": str(exc) or "Worker failed without a diagnostic",
                "traceback": traceback.format_exc(limit=12),
            },
        )
        raise
    finally:
        if worker is not None:
            worker.cleanup()

###############################################################################
def main() -> None:
    arguments = _parse_arguments()
    if arguments.probe:
        if arguments.backend is None:
            raise RuntimeError("--backend is required for a probe")
        try:
            _probe(ExecutionBackend(arguments.backend), RuntimeApi(arguments.api))
        except Exception as exc:
            reason = (
                "dependency_missing" if isinstance(exc, ModuleNotFoundError) else "probe_failed"
            )
            print(json.dumps({"available": False, "reason_code": reason, "message": str(exc)}))
            raise
        return
    if arguments.job_dir is None:
        raise RuntimeError("--job-dir is required")
    _run_job_process(arguments.job_dir)


###############################################################################
if __name__ == "__main__":
    main()
