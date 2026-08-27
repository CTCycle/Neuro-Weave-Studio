from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from common.process_control import (
    process_rss_bytes,
    process_start_options,
    sleep_poll,
    terminate_process_tree,
    wait_for_process,
)
from common.timestamps import utc_now
from configurations.settings import Settings
from domain.compilation_capabilities import get_native_compilation_capabilities
from domain.enums import ExecutionBackend, keras_backend_value
from domain.exports import ExportArtifactManifest, ExportFormat, export_suffix
from domain.graph import ModelGraph
from domain.jobs import (
    JobDiagnostics,
    JobError,
    JobOperation,
    JobProgress,
    JobResponse,
    JobState,
    JobSubmitRequest,
    ModelExportOperation,
    SmokeInferenceOperation,
)
from domain.metadata import ExecutionBackendStatus
from domain.runtime import (
    DeviceLeaseStatus,
    MemoryPolicyKind,
    RuntimeApi,
    RuntimeApiStatus,
    RuntimeDevice,
    WorkerConfiguration,
)
from models.export_artifacts import sha256_file

LOGGER = logging.getLogger(__name__)

###############################################################################
def _empty_diagnostics() -> dict[str, int | None]:
    return {}

###############################################################################
class RuntimeControllerError(RuntimeError):

    # -------------------------------------------------------------------------
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        self.code = code
        self.details = details or {}
        super().__init__(message)

###############################################################################
@dataclass
class _JobRecord:
    job_id: str
    operation: JobOperation
    worker: WorkerConfiguration
    device: RuntimeDevice
    job_dir: Path
    created_at: Any
    process: subprocess.Popen[bytes] | None = None
    state: JobState = JobState.STARTING
    pid: int | None = None
    started_at: Any | None = None
    updated_at: Any = field(default_factory=utc_now)
    finished_at: Any | None = None
    exit_code: int | None = None
    cancellation_requested: bool = False
    forced_termination: bool = False
    stop_started_at: float | None = None
    terminate_sent_at: float | None = None
    progress_phase: str = "queued"
    progress_message: str = "Worker is starting"
    progress_percent: int | None = None
    diagnostics: dict[str, int | None] = field(default_factory=_empty_diagnostics)
    result: dict[str, Any] | None = None
    error: JobError | None = None
    artifact_destination: Path | None = None
    artifact_filename: str | None = None
    export_format: ExportFormat | None = None
    artifact: ExportArtifactManifest | None = None
    project_revision: int | None = None
    revision_checker: Callable[[], bool] | None = None
    stderr_tail: str | None = None
    restart_reason: str = "new_job"
    monitor: threading.Thread | None = None

###############################################################################
class RuntimeController:
    """Owns short-lived ML workers, leases, lifecycle state, and diagnostics."""

    # -------------------------------------------------------------------------
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._server_root = Path(__file__).resolve().parents[1]
        self._job_root = settings.ml_job_root
        self._job_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, _JobRecord] = {}
        self._leases: dict[str, str] = {}
        self._last_worker_by_lease: dict[str, WorkerConfiguration] = {}
        self._backend_statuses: dict[ExecutionBackend, ExecutionBackendStatus] = {}
        self._api_statuses: dict[tuple[ExecutionBackend, RuntimeApi], RuntimeApiStatus] = {}
        self._shutting_down = False
        self._sweep_orphaned_jobs()

    # -------------------------------------------------------------------------
    @property
    def backend_statuses(self) -> dict[ExecutionBackend, ExecutionBackendStatus]:
        with self._lock:
            return dict(self._backend_statuses)

    # -------------------------------------------------------------------------
    def api_status(self, backend: ExecutionBackend, api: RuntimeApi) -> RuntimeApiStatus | None:
        with self._lock:
            return self._api_statuses.get((backend, api))

    # -------------------------------------------------------------------------
    def probe_statuses(self) -> dict[ExecutionBackend, dict[str, object]]:
        statuses: dict[ExecutionBackend, dict[str, object]] = {}
        for backend in ExecutionBackend:
            installed, version = self._installed_package(backend)
            api_results: dict[RuntimeApi, tuple[bool, str | None, dict[str, object]]] = {}
            for api in RuntimeApi:
                started = time.perf_counter()
                capabilities: dict[str, object] = {}
                reason: str | None = None
                try:
                    result = self._run_probe(backend, api)
                    capabilities = self._parse_probe_output(result)
                    expected_backend = (
                        keras_backend_value(backend)
                        if api is RuntimeApi.KERAS
                        else backend.value
                    )
                    available = bool(
                        result.returncode == 0
                        and capabilities.get("backend") == expected_backend
                        and capabilities.get("available", False)
                    )
                    if not available:
                        reason = str(capabilities.get("reason_code") or self._reason_code(result))
                except (OSError, subprocess.SubprocessError) as exc:
                    available = False
                    reason = "worker_failed"
                    capabilities = {"message": str(exc)}
                api_status = RuntimeApiStatus(
                    id=api,
                    available=available,
                    reason_code=reason,
                )
                with self._lock:
                    self._api_statuses[(backend, api)] = api_status
                api_results[api] = (available, reason, capabilities)
                LOGGER.info(
                    "backend_probe requested=%s api=%s duration_ms=%s outcome=%s",
                    backend.value,
                    api.value,
                    round((time.perf_counter() - started) * 1000, 1),
                    "available" if available else reason,
                )
            selected_api = next(
                (api for api in RuntimeApi if api_results[api][0]),
                RuntimeApi.KERAS,
            )
            available, reason, capabilities = api_results[selected_api]
            native_capabilities = api_results[RuntimeApi.NATIVE][2].get("native_compilers", {})
            native_compilers = (
                {str(key): bool(value) for key, value in cast(dict[object, object], native_capabilities).items()}
                if isinstance(native_capabilities, dict)
                else {}
            )
            raw_devices = capabilities.get("devices", [])
            device_items = cast(list[object], raw_devices) if isinstance(raw_devices, list) else []
            devices = [
                RuntimeDevice.model_validate(item)
                for item in device_items
                if isinstance(item, dict)
            ]
            if available and not devices:
                devices = [
                    RuntimeDevice(
                        device_id="cpu",
                        lease_key="cpu",
                        kind="cpu",
                        name="CPU",
                        accelerator="cpu",
                        supported_memory_policies=[MemoryPolicyKind.DEFAULT],
                    )
                ]
            runtime_api_statuses = [
                RuntimeApiStatus(
                    id=api,
                    available=result[0],
                    reason_code=None if result[0] else result[1],
                )
                for api, result in api_results.items()
            ]
            status = ExecutionBackendStatus(
                id=backend,
                installed=installed,
                version=version,
                available=any(result[0] for result in api_results.values()),
                reason_code=reason if not any(result[0] for result in api_results.values()) else None,
                accelerator=(str(capabilities.get("accelerator")) if available else None),
                device_name=(str(capabilities.get("device_name")) if available else None),
                devices=devices,
                runtime_apis=runtime_api_statuses,
                native_compilers=native_compilers,
            )
            with self._lock:
                self._backend_statuses[backend] = status
            statuses[backend] = status.model_dump(mode="json")
        return statuses

    # -------------------------------------------------------------------------
    def submit(
        self,
        request: JobSubmitRequest,
        *,
        graph: ModelGraph | None = None,
        artifact_destination: Path | None = None,
        artifact_filename: str | None = None,
        project_revision: int | None = None,
        revision_checker: Callable[[], bool] | None = None,
    ) -> JobResponse:
        with self._lock:
            if self._shutting_down:
                raise RuntimeControllerError("controller_shutting_down", "The ML runtime is shutting down")
            if isinstance(request.operation, SmokeInferenceOperation):
                operation = JobOperation.SMOKE_INFERENCE
                if graph is None:
                    graph = request.operation.graph
            else:
                operation = JobOperation.MODEL_EXPORT
                if graph is None:
                    raise RuntimeControllerError(
                        "missing_graph_snapshot", "Model export submission requires a graph snapshot"
                    )
            device = self._validate_worker(request.worker)
            self._validate_native_compiler(request.worker, graph)
            lease_key = device.lease_key
            if lease_key in self._leases:
                raise RuntimeControllerError(
                    "device_busy",
                    f"Device '{lease_key}' already has an active ML job",
                    details={"device_id": device.device_id, "job_id": self._leases[lease_key]},
                )
            export_format = (
                request.operation.format
                if isinstance(request.operation, ModelExportOperation)
                else None
            )
            if operation is JobOperation.MODEL_EXPORT and artifact_filename is None:
                artifact_filename = f"model{export_suffix(export_format or ExportFormat.KERAS)}"
            job_id = uuid4().hex
            job_dir = self._job_root / job_id
            job_dir.mkdir(parents=True, exist_ok=False)
            restart_reason = self._restart_reason(lease_key, request.worker)
            record = _JobRecord(
                job_id=job_id,
                operation=operation,
                worker=request.worker,
                device=device,
                job_dir=job_dir,
                created_at=utc_now(),
                updated_at=utc_now(),
                artifact_destination=artifact_destination,
                artifact_filename=artifact_filename,
                export_format=export_format,
                project_revision=project_revision,
                revision_checker=revision_checker,
                restart_reason=restart_reason,
            )
            operation_payload: dict[str, object] = {
                "type": operation.value,
                "graph": graph.model_dump(mode="json"),
            }
            operation_payload["project_id"] = str(request.operation.project_id)
            if isinstance(request.operation, ModelExportOperation):
                operation_payload["format"] = request.operation.format.value
                operation_payload["filename_stem"] = request.operation.filename_stem
                operation_payload["options"] = request.operation.options.model_dump(mode="json")
                operation_payload["artifact_filename"] = artifact_filename
                operation_payload["project_revision"] = project_revision
            self._write_json(
                job_dir / "request.json",
                {"operation": operation_payload, "worker": request.worker.model_dump(mode="json")},
            )
            self._write_json(
                job_dir / "status.json",
                {
                    "state": "starting",
                    "phase": "queued",
                    "message": "Worker is starting",
                    "percent": 0,
                    "updated_at": time.time(),
                },
            )
            try:
                process = self._start_process(job_dir)
            except Exception:
                self._remove_job_workspace(job_dir)
                raise
            record.process = process
            record.pid = process.pid
            record.started_at = utc_now()
            record.updated_at = utc_now()
            self._jobs[job_id] = record
            self._leases[lease_key] = job_id
            self._last_worker_by_lease[lease_key] = request.worker
            monitor = threading.Thread(
                target=self._monitor_job,
                args=(record,),
                name=f"neuro-weave-job-{job_id[:8]}",
                daemon=True,
            )
            record.monitor = monitor
            monitor.start()
            return self._response(record)

    # -------------------------------------------------------------------------
    def stop(self, job_id: str) -> JobResponse:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise RuntimeControllerError("job_not_found", f"ML job '{job_id}' was not found")
            if record.state in {JobState.SUCCEEDED, JobState.TERMINATED, JobState.FAILED}:
                return self._response(record)
            record.cancellation_requested = True
            record.stop_started_at = record.stop_started_at or time.monotonic()
            record.state = JobState.STOP_REQUESTED
            record.progress_phase = "cancelling"
            record.progress_message = "Cancellation requested; waiting for the worker to exit"
            record.updated_at = utc_now()
            try:
                (record.job_dir / "cancel.requested").write_text("requested", encoding="utf-8")
            except OSError:
                LOGGER.warning("Could not write cancellation marker for job %s", job_id)
            return self._response(record)

    # -------------------------------------------------------------------------
    def get(self, job_id: str) -> JobResponse:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise RuntimeControllerError("job_not_found", f"ML job '{job_id}' was not found")
            return self._response(record)

    # -------------------------------------------------------------------------
    def list(self, *, active_only: bool = False) -> list[JobResponse]:
        with self._lock:
            records = list(self._jobs.values())
            if active_only:
                records = [
                    record
                    for record in records
                    if record.state not in {JobState.SUCCEEDED, JobState.TERMINATED, JobState.FAILED}
                ]
            records.sort(key=lambda item: item.created_at, reverse=True)
            return [self._response(record) for record in records]

    # -------------------------------------------------------------------------
    def artifact_path(self, job_id: str) -> tuple[Path, str]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise RuntimeControllerError("job_not_found", f"ML job '{job_id}' was not found")
            if record.operation is not JobOperation.MODEL_EXPORT:
                raise RuntimeControllerError("artifact_not_available", "This job does not produce an artifact")
            if record.state is not JobState.SUCCEEDED or record.artifact_destination is None:
                raise RuntimeControllerError("artifact_not_ready", "The export artifact is not ready")
            if not record.artifact_destination.is_file():
                raise RuntimeControllerError("artifact_not_found", "The export artifact is no longer available")
            return record.artifact_destination, record.artifact_filename or record.artifact_destination.name

    # -------------------------------------------------------------------------
    def shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True
            active = [
                record.job_id
                for record in self._jobs.values()
                if record.state not in {JobState.SUCCEEDED, JobState.TERMINATED, JobState.FAILED}
            ]
        for job_id in active:
            try:
                self.stop(job_id)
            except RuntimeControllerError:
                continue
        deadline = time.monotonic() + self._settings.ml_cancel_grace_seconds + self._settings.ml_terminate_grace_seconds + 2
        while time.monotonic() < deadline:
            with self._lock:
                monitors = [record.monitor for record in self._jobs.values() if record.monitor and record.monitor.is_alive()]
            if not monitors:
                break
            for monitor in monitors:
                monitor.join(timeout=0.1)
        with self._lock:
            remaining = [record for record in self._jobs.values() if record.process and record.process.poll() is None]
        for record in remaining:
            if record.pid is not None:
                terminate_process_tree(record.pid, force=True)
            if record.process is not None:
                wait_for_process(record.process, 2)
        for record in remaining:
            if record.monitor is not None:
                record.monitor.join(timeout=2)

    # -------------------------------------------------------------------------
    def _validate_worker(self, worker: WorkerConfiguration) -> RuntimeDevice:
        status = self._backend_statuses.get(worker.backend)
        api_status = self._api_statuses.get((worker.backend, worker.api))
        if api_status is None or not api_status.available:
            reason_code = api_status.reason_code if api_status is not None else "unknown_api"
            raise RuntimeControllerError(
                "api_unavailable",
                f"Runtime API '{worker.api.value}' is not operational for {worker.backend.value}",
                details={"backend": worker.backend.value, "api": worker.api.value, "reason_code": reason_code},
            )
        if status is None or not status.available:
            raise RuntimeControllerError(
                "backend_unavailable",
                f"Execution backend '{worker.backend.value}' is not operational",
                details={"backend": worker.backend.value, "reason_code": status.reason_code if status else "unknown_backend"},
            )
        device = next((item for item in status.devices if item.device_id == worker.device_id), None)
        if device is None or not device.available:
            raise RuntimeControllerError(
                "device_unavailable",
                f"Device '{worker.device_id}' is not available for {worker.backend.value}",
                details={"backend": worker.backend.value, "device_id": worker.device_id},
            )
        if worker.memory_policy.kind not in device.supported_memory_policies:
            raise RuntimeControllerError(
                "memory_policy_unsupported",
                f"Memory policy '{worker.memory_policy.kind.value}' is not supported on '{worker.device_id}'",
                details={"backend": worker.backend.value, "device_id": worker.device_id},
            )
        return device

    # -------------------------------------------------------------------------
    def _validate_native_compiler(
        self, worker: WorkerConfiguration, graph: ModelGraph | None
    ) -> None:
        if worker.api is not RuntimeApi.NATIVE or graph is None:
            return
        capability = get_native_compilation_capabilities().get(worker.backend)
        if capability is None:
            return
        backend_configuration = getattr(graph.model_configuration.native, worker.backend.value)
        enabled_field = {
            ExecutionBackend.TENSORFLOW: "jit_compile",
            ExecutionBackend.PYTORCH: "compile_enabled",
            ExecutionBackend.JAX: "jit_enabled",
        }[worker.backend]
        if not bool(getattr(backend_configuration, enabled_field)):
            return
        status = self._backend_statuses.get(worker.backend)
        if status is not None and status.native_compilers.get(capability.compiler_id, False):
            return
        raise RuntimeControllerError(
            "native_compiler_unavailable",
            f"{capability.compiler} is not operational for {worker.backend.value}",
            details={
                "backend": worker.backend.value,
                "compiler": capability.compiler_id,
                "reason_code": "compiler_unavailable",
            },
        )

    # -------------------------------------------------------------------------
    def _start_process(self, job_dir: Path) -> subprocess.Popen[bytes]:
        log_path = job_dir / "worker.log"
        log_handle = log_path.open("wb")
        environment = os.environ.copy()
        environment.pop("KERAS_BACKEND", None)
        # Torch ONNX diagnostics can contain Unicode symbols on Windows. Keep
        # worker logs UTF-8 so a diagnostic cannot turn a valid export into a
        # locale-dependent encoding failure.
        environment["PYTHONIOENCODING"] = "utf-8"
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "models.ml_worker", "--job-dir", str(job_dir)],
                cwd=self._server_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                **process_start_options(),
            )
        finally:
            log_handle.close()
        return cast(subprocess.Popen[bytes], process)

    # -------------------------------------------------------------------------
    def _monitor_job(self, record: _JobRecord) -> None:
        process = record.process
        if process is None:
            return
        timeout_started = time.monotonic()
        last_log_bound = timeout_started
        while process.poll() is None:
            status = self._read_status(record.job_dir / "status.json")
            with self._lock:
                self._apply_worker_status(record, status)
                if record.cancellation_requested:
                    record.state = JobState.STOP_REQUESTED
                    elapsed = time.monotonic() - (record.stop_started_at or time.monotonic())
                    if elapsed >= self._settings.ml_cancel_grace_seconds and record.terminate_sent_at is None:
                        record.state = JobState.STOPPING
                        record.terminate_sent_at = time.monotonic()
                        terminate_process_tree(record.pid or -1, force=False)
                    elif (
                        record.terminate_sent_at is not None
                        and time.monotonic() - record.terminate_sent_at >= self._settings.ml_terminate_grace_seconds
                    ):
                        record.state = JobState.STOPPING
                        record.forced_termination = True
                        terminate_process_tree(record.pid or -1, force=True)
                        with suppress(OSError):
                            process.kill()
                elif time.monotonic() - timeout_started >= self._settings.ml_worker_timeout_seconds:
                    record.cancellation_requested = True
                    record.stop_started_at = time.monotonic()
                    record.error = JobError(
                        code="worker_timeout",
                        message="The ML worker exceeded its configured timeout",
                    )
            if time.monotonic() - last_log_bound >= 1:
                self._bound_log(record.job_dir / "worker.log")
                last_log_bound = time.monotonic()
            sleep_poll(0.1)
        process.wait()
        status = self._read_status(record.job_dir / "status.json")
        with self._lock:
            self._apply_worker_status(record, status)
            record.exit_code = process.returncode
            record.finished_at = utc_now()
            record.updated_at = utc_now()
            if record.exit_code == 0 and status.get("state") == "succeeded" and not record.forced_termination:
                if record.operation is JobOperation.MODEL_EXPORT:
                    record.progress_phase = "publishing"
                    record.progress_message = "Publishing verified artifact"
                    if self._publish_artifact(record):
                        record.state = JobState.SUCCEEDED
                        record.progress_phase = "completed"
                        record.progress_message = "Export completed and published"
                        record.progress_percent = 100
                    else:
                        record.state = JobState.FAILED
                else:
                    record.state = JobState.SUCCEEDED
                    record.progress_phase = "completed"
                    record.progress_message = "Worker completed successfully"
                    record.progress_percent = 100
            elif record.cancellation_requested or record.exit_code == 20 or status.get("state") == "terminated":
                record.state = JobState.TERMINATED
                if record.error is None:
                    record.error = JobError(code="cancelled", message="The ML worker was terminated")
            else:
                record.state = JobState.FAILED
                if record.error is None:
                    error_payload = status.get("error")
                    if isinstance(error_payload, dict):
                        record.error = JobError.model_validate(error_payload)
                    else:
                        record.error = JobError(
                            code="worker_failed",
                            message=self._tail_log(record.job_dir / "worker.log") or "ML worker failed",
                        )
            self._bound_log(record.job_dir / "worker.log")
            record.stderr_tail = self._tail_log(record.job_dir / "worker.log")
            self._leases.pop(record.device.lease_key, None)
            self._trim_history()
        self._remove_job_workspace(record.job_dir)

    # -------------------------------------------------------------------------
    def _apply_worker_status(self, record: _JobRecord, status: Mapping[str, object]) -> None:
        if not status:
            return
        state = str(status.get("state", ""))
        if (
            not record.cancellation_requested
            and state == "running"
            and record.state is JobState.STARTING
        ):
            record.state = JobState.RUNNING
        record.progress_phase = str(status.get("phase", record.progress_phase))
        record.progress_message = str(status.get("message", record.progress_message))
        percent = status.get("percent")
        record.progress_percent = int(percent) if isinstance(percent, int) else record.progress_percent
        for key in (
            "process_rss_bytes",
            "framework_allocated_bytes",
            "framework_reserved_bytes",
            "framework_peak_bytes",
            "device_used_bytes",
            "device_total_bytes",
        ):
            value = status.get(key)
            if isinstance(value, int) or value is None:
                record.diagnostics[key] = value
        if record.pid is not None:
            record.diagnostics["process_rss_bytes"] = process_rss_bytes(record.pid)
        record.updated_at = utc_now()
        result = status.get("result")
        if isinstance(result, dict):
            record.result = cast(dict[str, Any], result)
        error = status.get("error")
        if isinstance(error, dict):
            error_payload = cast(dict[str, Any], error)
            try:
                record.error = JobError.model_validate(error_payload)
            except ValueError:
                record.error = JobError(code="worker_failed", message=str(error_payload))

    # -------------------------------------------------------------------------
    def _publish_artifact(self, record: _JobRecord) -> bool:
        if record.artifact_destination is None:
            record.error = JobError(code="artifact_destination_missing", message="No export destination was assigned")
            return False
        if record.revision_checker is not None:
            try:
                if not record.revision_checker():
                    record.error = JobError(
                        code="revision_conflict",
                        message="The project changed while the artifact was being generated",
                    )
                    return False
            except Exception as exc:
                record.error = JobError(code="revision_check_failed", message=str(exc))
                return False
        result_path = (record.result or {}).get("artifact_path")
        if not isinstance(result_path, str) or not result_path:
            record.error = JobError(code="artifact_missing", message="Worker completed without an artifact path")
            return False
        candidate = Path(result_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            record.error = JobError(code="artifact_path_invalid", message="Worker returned an unsafe artifact path")
            return False
        source = (record.job_dir / candidate).resolve()
        try:
            source.relative_to(record.job_dir.resolve())
        except ValueError:
            record.error = JobError(code="artifact_path_invalid", message="Worker returned an artifact outside its workspace")
            return False
        if not source.is_file():
            record.error = JobError(code="artifact_missing", message="Worker completed without an artifact")
            return False
        manifest_payload = (record.result or {}).get("artifact_manifest")
        if not isinstance(manifest_payload, dict):
            record.error = JobError(code="artifact_manifest_missing", message="Worker completed without an artifact manifest")
            return False
        try:
            manifest = ExportArtifactManifest.model_validate(manifest_payload)
        except ValueError as exc:
            record.error = JobError(code="artifact_manifest_invalid", message=str(exc))
            return False
        if record.export_format is not None and manifest.format is not record.export_format:
            record.error = JobError(code="artifact_manifest_mismatch", message="Artifact manifest format does not match the requested export")
            return False
        if record.project_revision is not None and manifest.project_revision != record.project_revision:
            record.error = JobError(code="artifact_manifest_mismatch", message="Artifact manifest revision does not match the requested export")
            return False
        try:
            self._validate_staged_artifact(source, manifest)
        except (OSError, RuntimeError, ValueError) as exc:
            record.error = JobError(code="artifact_validation_failed", message=str(exc))
            return False
        temporary_destination = record.artifact_destination.with_name(
            f".{record.artifact_destination.name}.{record.job_id}.tmp"
        )
        try:
            record.artifact_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, temporary_destination)
            os.replace(temporary_destination, record.artifact_destination)
        except OSError as exc:
            with suppress(OSError):
                temporary_destination.unlink()
            record.error = JobError(code="artifact_publish_failed", message=str(exc))
            return False
        record.result = record.result or {}
        record.result["artifact_name"] = record.artifact_filename or record.artifact_destination.name
        record.artifact = manifest
        return True

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_staged_artifact(
        source: Path,
        manifest: ExportArtifactManifest,
    ) -> None:
        if manifest.layout.value == "file":
            if len(manifest.files) != 1 or manifest.files[0].path != source.name:
                raise RuntimeError("File artifact manifest does not describe the staged artifact")
            declared = manifest.files[0]
            if declared.size_bytes != source.stat().st_size or declared.sha256 != sha256_file(source):
                raise RuntimeError("Staged artifact hash or size does not match its manifest")
            return
        with zipfile.ZipFile(source) as archive:
            names = set(archive.namelist())
            if "manifest.json" not in names:
                raise RuntimeError("Archive artifact is missing manifest.json")
            for declared in manifest.files:
                path = declared.path.replace("\\", "/")
                if not path or path.startswith("/") or ".." in Path(path).parts or path not in names:
                    raise RuntimeError(f"Archive artifact is missing declared file '{declared.path}'")
                payload = archive.read(path)
                if len(payload) != declared.size_bytes or hashlib.sha256(payload).hexdigest() != declared.sha256:
                    raise RuntimeError(f"Archive file '{declared.path}' does not match its manifest")

    # -------------------------------------------------------------------------
    def _response(self, record: _JobRecord) -> JobResponse:
        lease_state = "idle"
        if record.state in {JobState.STARTING, JobState.RUNNING}:
            lease_state = record.state.value
        elif record.state is JobState.STOP_REQUESTED:
            lease_state = "stop_requested"
        elif record.state is JobState.STOPPING:
            lease_state = "stopping"
        elif record.state is JobState.FAILED:
            lease_state = "failed"
        return JobResponse(
            id=record.job_id,
            operation=record.operation,
            state=record.state,
            worker=record.worker,
            device_lease=DeviceLeaseStatus(
                device_id=record.device.device_id,
                state=cast(Any, lease_state),
                job_id=record.job_id if lease_state != "idle" else None,
                pid=record.pid if lease_state != "idle" else None,
            ),
            pid=record.pid,
            created_at=record.created_at,
            started_at=record.started_at,
            updated_at=record.updated_at,
            finished_at=record.finished_at,
            exit_code=record.exit_code,
            cancellation_requested=record.cancellation_requested,
            forced_termination=record.forced_termination,
            restart_reason=record.restart_reason,
            progress=JobProgress(
                phase=record.progress_phase,
                message=record.progress_message,
                percent=record.progress_percent,
            ),
            diagnostics=JobDiagnostics.model_validate(record.diagnostics),
            result=record.result,
            artifact=record.artifact,
            error=record.error,
        )

    # -------------------------------------------------------------------------
    def _restart_reason(self, lease_key: str, worker: WorkerConfiguration) -> str:
        previous = self._last_worker_by_lease.get(lease_key)
        if previous is None:
            return "new_job"
        if previous.api != worker.api:
            return "api_changed"
        if previous.backend != worker.backend:
            return "backend_changed"
        if previous.device_id != worker.device_id:
            return "device_changed"
        if previous.memory_policy != worker.memory_policy:
            return "memory_policy_changed"
        if previous.precision != worker.precision:
            return "precision_changed"
        return "replacement_job"

    # -------------------------------------------------------------------------
    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    # -------------------------------------------------------------------------
    @staticmethod
    def _read_status(path: Path) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cast(dict[str, object], payload) if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    # -------------------------------------------------------------------------
    @staticmethod
    def _tail_log(path: Path, limit: int = 4_000) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip() or None
        except OSError:
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _bound_log(path: Path, limit: int = 64_000) -> None:
        try:
            data = path.read_bytes()
            if len(data) <= limit:
                return
            temporary = path.with_suffix(".bounded")
            temporary.write_bytes(data[-limit:])
            os.replace(temporary, path)
        except OSError:
            return

    # -------------------------------------------------------------------------
    def _trim_history(self) -> None:
        terminal = [
            record
            for record in self._jobs.values()
            if record.state in {JobState.SUCCEEDED, JobState.TERMINATED, JobState.FAILED}
        ]
        terminal.sort(key=lambda item: item.created_at)
        for record in terminal[: max(0, len(terminal) - self._settings.ml_job_history_limit)]:
            self._jobs.pop(record.job_id, None)

    # -------------------------------------------------------------------------
    @staticmethod
    def _remove_job_workspace(path: Path) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            return

    # -------------------------------------------------------------------------
    def _sweep_orphaned_jobs(self) -> None:
        for path in self._job_root.iterdir():
            if path.is_dir():
                self._remove_job_workspace(path)

    # -------------------------------------------------------------------------
    def _run_probe(
        self,
        backend: ExecutionBackend,
        api: RuntimeApi,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("KERAS_BACKEND", None)
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "models.ml_worker",
                "--probe",
                "--backend",
                backend.value,
                "--api",
                api.value,
            ],
            cwd=self._server_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=self._settings.ml_worker_timeout_seconds,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_probe_output(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
        for line in reversed(result.stdout.splitlines()):
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    return cast(dict[str, object], payload)
            except json.JSONDecodeError:
                continue
        return {}

    # -------------------------------------------------------------------------
    @staticmethod
    def _reason_code(result: subprocess.CompletedProcess[str]) -> str:
        detail = f"{result.stderr}\n{result.stdout}".lower()
        if "no module named" in detail or "modulenotfounderror" in detail:
            return "dependency_missing"
        if "backend" in detail:
            return "backend_import_failed"
        return "probe_failed"

    # -------------------------------------------------------------------------
    @staticmethod
    def _installed_package(backend: ExecutionBackend) -> tuple[bool, str | None]:
        package_name = {
            ExecutionBackend.TENSORFLOW: "tensorflow",
            ExecutionBackend.PYTORCH: "torch",
            ExecutionBackend.JAX: "jax",
        }[backend]
        try:
            installed = importlib.util.find_spec(package_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            return False, None
        try:
            return True, importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return True, None
