# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.process_control import process_start_options
from configurations.settings import Settings
from domain.enums import ExecutionBackend, keras_backend_value
from domain.graph import ModelConfiguration, ModelGraph, NativeCompilationConfiguration
from domain.jobs import JobResponse, JobSubmitRequest, SmokeInferenceOperation
from domain.metadata import ExecutionBackendStatus
from domain.runtime import (
    MemoryPolicyKind,
    RuntimeApi,
    RuntimeApiStatus,
    RuntimeDevice,
    RuntimeDiscoveryState,
    WorkerConfiguration,
)
from services.runtime_controller import (
    DISCOVERY_SNAPSHOT_SCHEMA_VERSION,
    RuntimeController,
    RuntimeControllerError,
)

# Consolidated from test_runtime_controller.py

###############################################################################
def status(*devices: RuntimeDevice) -> ExecutionBackendStatus:
    return ExecutionBackendStatus(
        id=ExecutionBackend.PYTORCH,
        installed=True,
        version="test",
        available=True,
        accelerator="cpu",
        devices=list(devices),
    )

###############################################################################
def fake_start(job_dir: Path, sleep_seconds: float) -> subprocess.Popen[bytes]:
    code = (
        "import json, pathlib, sys, time; "
        "d=pathlib.Path(sys.argv[1]); "
        "(d/'status.json').write_text(json.dumps({'state':'running','phase':'inference',"
        "'message':'fake worker','percent':50,'pid':__import__('os').getpid()})); "
        f"time.sleep({sleep_seconds}); "
        "(d/'status.json').write_text(json.dumps({'state':'succeeded','phase':'finished',"
        "'message':'fake complete','percent':100,'pid':__import__('os').getpid(),"
        "'result':{'ok':True}}))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(job_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **process_start_options(),
    )
    return cast(subprocess.Popen[bytes], process)

###############################################################################
def controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    sleep_seconds: float = 0.2,
    history_limit: int = 10,
) -> RuntimeController:
    settings = Settings(
        ml_job_root=tmp_path / "jobs",
        export_root=tmp_path / "exports",
        ml_cancel_grace_seconds=1,
        ml_terminate_grace_seconds=1,
        ml_job_history_limit=history_limit,
    )
    runtime = RuntimeController(settings)
    cpu = RuntimeDevice(
        device_id="cpu",
        lease_key="cpu",
        kind="cpu",
        name="CPU",
        accelerator="cpu",
        supported_memory_policies=[MemoryPolicyKind.DEFAULT],
    )
    gpu = RuntimeDevice(
        device_id="cuda:0",
        lease_key="cuda:0",
        kind="cuda",
        name="Test GPU",
        accelerator="cuda",
        supported_memory_policies=[MemoryPolicyKind.DEFAULT],
    )
    runtime._backend_statuses = {  # pyright: ignore[reportPrivateUsage]
        ExecutionBackend.PYTORCH: status(cpu, gpu)
    }
    runtime._api_statuses = {  # pyright: ignore[reportPrivateUsage]
        (ExecutionBackend.PYTORCH, RuntimeApi.KERAS): RuntimeApiStatus(
            id=RuntimeApi.KERAS,
            available=True,
        )
    }
    monkeypatch.setattr(
        runtime,
        "_start_process",
        partial(fake_start, sleep_seconds=sleep_seconds),
    )
    return runtime

###############################################################################
def request(device_id: str = "cpu") -> JobSubmitRequest:
    return JobSubmitRequest(
        operation=SmokeInferenceOperation(project_id=uuid4(), graph=ModelGraph()),
        worker=WorkerConfiguration(
            backend=ExecutionBackend.PYTORCH,
            device_id=device_id,
        ),
    )

###############################################################################
def wait(runtime: RuntimeController, job_id: str) -> JobResponse:
    for _ in range(320):
        job = runtime.get(job_id)
        if job.state.value in {"succeeded", "terminated", "failed"}:
            return job
        time.sleep(0.02)
    pytest.fail(f"job {job_id} did not finish")

###############################################################################
def test_device_leases_are_exclusive_but_distinct_devices_can_run_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = controller(tmp_path, monkeypatch, sleep_seconds=0.35)
    try:
        first = runtime.submit(request("cpu"), graph=ModelGraph())
        with pytest.raises(RuntimeControllerError, match="already has an active"):
            runtime.submit(request("cpu"), graph=ModelGraph())

        second = runtime.submit(request("cuda:0"), graph=ModelGraph())
        assert wait(runtime, first.id).state.value == "succeeded"
        assert wait(runtime, second.id).state.value == "succeeded"

        replacement = runtime.submit(request("cpu"), graph=ModelGraph())
        assert replacement.restart_reason == "replacement_job"
        wait(runtime, replacement.id)
    finally:
        runtime.shutdown()

###############################################################################
def test_stop_is_idempotent_and_releases_lease_only_after_worker_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = controller(tmp_path, monkeypatch, sleep_seconds=10)
    try:
        job = runtime.submit(request(), graph=ModelGraph())

        assert runtime.stop(job.id).cancellation_requested is True
        assert runtime.stop(job.id).cancellation_requested is True

        terminal = wait(runtime, job.id)
        assert terminal.state.value == "terminated"
        assert terminal.device_lease.state == "idle"

        replacement = runtime.submit(request(), graph=ModelGraph())
        runtime.stop(replacement.id)
        wait(runtime, replacement.id)
    finally:
        runtime.shutdown()

###############################################################################
def test_terminal_history_is_bounded_and_job_workspaces_are_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = controller(
        tmp_path,
        monkeypatch,
        sleep_seconds=0.01,
        history_limit=10,
    )
    try:
        for _ in range(11):
            job = runtime.submit(request(), graph=ModelGraph())
            wait(runtime, job.id)

        assert len(runtime.list()) == 10
        for _ in range(50):
            if not any(path.is_dir() for path in (tmp_path / "jobs").iterdir()):
                break
            time.sleep(0.02)
        assert not any(path.is_dir() for path in (tmp_path / "jobs").iterdir())
    finally:
        runtime.shutdown()

###############################################################################
def test_environment_discovery_skips_missing_backends_and_reuses_matching_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = {
        ExecutionBackend.TENSORFLOW: (False, None),
        ExecutionBackend.PYTORCH: (True, "2.1.0"),
        ExecutionBackend.JAX: (False, None),
    }
    probes: list[tuple[ExecutionBackend, RuntimeApi]] = []

    def fake_installed(backend: ExecutionBackend) -> tuple[bool, str | None]:
        return installed[backend]

    def fake_probe(
        _runtime: RuntimeController,
        backend: ExecutionBackend,
        api: RuntimeApi,
    ) -> subprocess.CompletedProcess[str]:
        probes.append((backend, api))
        expected_backend = (
            keras_backend_value(backend) if api is RuntimeApi.KERAS else backend.value
        )
        payload = {
            "backend": expected_backend,
            "available": True,
            "accelerator": "cpu",
            "device_name": "CPU",
            "devices": [
                {
                    "device_id": "cpu",
                    "lease_key": "cpu",
                    "kind": "cpu",
                    "name": "CPU",
                    "accelerator": "cpu",
                    "supported_memory_policies": ["default"],
                }
            ],
            "native_compilers": {"pytorch_compile": True},
        }
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    monkeypatch.setattr(RuntimeController, "_installed_package", staticmethod(fake_installed))
    monkeypatch.setattr(
        RuntimeController, "_hardware_fingerprint", staticmethod(lambda: {"nvidia_smi": None})
    )
    monkeypatch.setattr(RuntimeController, "_run_probe", fake_probe)
    settings = Settings(
        ml_job_root=tmp_path / "jobs",
        export_root=tmp_path / "exports",
        runtime_discovery_path=tmp_path / "environment-discovery.json",
    )
    settings.runtime_discovery_path.write_text(
        json.dumps(
            {
                "schema_version": DISCOVERY_SNAPSHOT_SCHEMA_VERSION - 1,
                "fingerprint": "legacy-snapshot",
                "statuses": {"pytorch": {"native_compilers": {"torch_compile": True}}},
            }
        ),
        encoding="utf-8",
    )

    first = RuntimeController(settings)
    try:
        first.probe_statuses()
        assert probes == [
            (ExecutionBackend.PYTORCH, RuntimeApi.KERAS),
            (ExecutionBackend.PYTORCH, RuntimeApi.NATIVE),
        ]
        assert (
            first.backend_statuses[ExecutionBackend.TENSORFLOW].reason_code == "dependency_missing"
        )
        refreshed_snapshot = json.loads(settings.runtime_discovery_path.read_text(encoding="utf-8"))
        assert refreshed_snapshot["schema_version"] == DISCOVERY_SNAPSHOT_SCHEMA_VERSION
        assert refreshed_snapshot["statuses"]["pytorch"]["native_compilers"] == {
            "pytorch_compile": True
        }
    finally:
        first.shutdown()

    second = RuntimeController(settings)
    try:
        second.probe_statuses()
        assert len(probes) == 2

        installed[ExecutionBackend.PYTORCH] = (True, "2.2.0")
        second.probe_statuses()
        assert len(probes) == 4
    finally:
        second.shutdown()

###############################################################################
def test_native_compiler_availability_ignores_obsolete_capability_keys() -> None:
    from domain.compilation_capabilities import get_native_compilation_capabilities

    statuses = {
        ExecutionBackend.TENSORFLOW: ExecutionBackendStatus(
            id=ExecutionBackend.TENSORFLOW,
            available=True,
            native_compilers={"tensorflow_jit": True},
        ),
        ExecutionBackend.PYTORCH: ExecutionBackendStatus(
            id=ExecutionBackend.PYTORCH,
            available=True,
            native_compilers={"torch_compile": True},
        ),
        ExecutionBackend.JAX: ExecutionBackendStatus(
            id=ExecutionBackend.JAX,
            available=True,
            native_compilers={"jax_jit": False},
        ),
    }

    capabilities = get_native_compilation_capabilities(statuses)

    assert capabilities[ExecutionBackend.TENSORFLOW].available is False
    assert capabilities[ExecutionBackend.PYTORCH].available is False
    assert capabilities[ExecutionBackend.JAX].available is False

###############################################################################
@pytest.mark.parametrize(
    ("backend", "compiler_id", "compiler_configuration"),
    [
        (
            ExecutionBackend.TENSORFLOW,
            "tensorflow_function",
            {"tensorflow": {"compiler": {"id": "tensorflow_function", "parameters": {"jit_compile": True}}}},
        ),
        (
            ExecutionBackend.PYTORCH,
            "pytorch_compile",
            {"pytorch": {"compiler": {"id": "pytorch_compile", "parameters": {"enabled": True}}}},
        ),
        (
            ExecutionBackend.JAX,
            "jax_jit",
            {"jax": {"compiler": {"id": "jax_jit", "parameters": {"enabled": True}}}},
        ),
    ],
)
def test_unavailable_native_compilers_are_rejected_before_worker_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: ExecutionBackend,
    compiler_id: str,
    compiler_configuration: dict[str, object],
) -> None:
    runtime = controller(tmp_path, monkeypatch)
    cpu = RuntimeDevice(
        device_id="cpu",
        lease_key="cpu",
        kind="cpu",
        name="CPU",
        accelerator="cpu",
        supported_memory_policies=[MemoryPolicyKind.DEFAULT],
    )
    runtime._backend_statuses[backend] = ExecutionBackendStatus(  # pyright: ignore[reportPrivateUsage]
        id=backend,
        installed=True,
        version="test",
        available=True,
        accelerator="cpu",
        devices=[cpu],
        native_compilers={compiler_id: False},
    )
    runtime._api_statuses[(backend, RuntimeApi.NATIVE)] = RuntimeApiStatus(  # pyright: ignore[reportPrivateUsage]
        id=RuntimeApi.NATIVE,
        available=True,
    )
    graph = ModelGraph(
        model_configuration=ModelConfiguration(
            native=NativeCompilationConfiguration.model_validate(compiler_configuration)
        )
    )
    submission = JobSubmitRequest(
        operation=SmokeInferenceOperation(project_id=uuid4(), graph=graph),
        worker=WorkerConfiguration(api=RuntimeApi.NATIVE, backend=backend, device_id="cpu"),
    )

    try:
        with pytest.raises(RuntimeControllerError) as raised:
            runtime.submit(submission, graph=graph)

        assert raised.value.code == "native_compiler_unavailable"
        assert raised.value.details == {
            "backend": backend.value,
            "compiler": compiler_id,
            "reason_code": "compiler_unavailable",
        }
        assert list((tmp_path / "jobs").iterdir()) == []
    finally:
        runtime.shutdown()

# Consolidated from test_startup_regressions.py

# pyright: reportUnknownLambdaType=false, reportUnknownArgumentType=false




###############################################################################
def test_controller_startup_does_not_probe_optional_frameworks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_probe(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("optional framework probe ran during controller startup")

    monkeypatch.setattr(RuntimeController, "_run_probe", fail_probe)
    runtime = RuntimeController(Settings(ml_job_root=tmp_path / "jobs", runtime_discovery_path=tmp_path / "missing.json"))
    try:
        assert all(status.discovery_state is RuntimeDiscoveryState.UNCHECKED for status in runtime.backend_statuses.values())
        assert all(not status.available for status in runtime.backend_statuses.values())
    finally:
        runtime.shutdown()

###############################################################################
def test_discovery_requests_are_deduplicated_and_complete_in_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RuntimeController(Settings(ml_job_root=tmp_path / "jobs", runtime_discovery_path=tmp_path / "discovery.json"))
    try:
        monkeypatch.setattr(runtime, "_environment_fingerprint", lambda: ("test-fingerprint", {}))
        monkeypatch.setattr(runtime, "_load_discovery_snapshot", lambda _fingerprint: None)

        def fake_probe(backend: ExecutionBackend, api: RuntimeApi) -> tuple[bool, str | None, dict[str, object]]:
            time.sleep(0.05)
            expected = keras_backend_value(backend) if api is RuntimeApi.KERAS else backend.value
            return True, None, {
                "backend": expected,
                "available": True,
                "accelerator": "cpu",
                "device_name": "Test CPU",
                "devices": [
                    {
                        "device_id": "cpu",
                        "lease_key": "cpu",
                        "kind": "cpu",
                        "name": "Test CPU",
                        "accelerator": "cpu",
                        "supported_memory_policies": ["default"],
                    }
                ],
            }

        monkeypatch.setattr(runtime, "_probe_backend_api", fake_probe)
        first = runtime.request_discovery(ExecutionBackend.PYTORCH, RuntimeApi.KERAS)
        second = runtime.request_discovery(ExecutionBackend.PYTORCH, RuntimeApi.KERAS)
        assert first.state is RuntimeDiscoveryState.CHECKING
        assert second.state is RuntimeDiscoveryState.CHECKING

        result = runtime.discovery_status(ExecutionBackend.PYTORCH, RuntimeApi.KERAS)
        for _ in range(100):
            result = runtime.discovery_status(ExecutionBackend.PYTORCH, RuntimeApi.KERAS)
            if result.state is not RuntimeDiscoveryState.CHECKING:
                break
            time.sleep(0.01)
        assert result.state is RuntimeDiscoveryState.VERIFIED
        assert result.available is True
        assert result.source == "live"
    finally:
        runtime.shutdown()

###############################################################################
def test_force_refresh_bypasses_matching_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = RuntimeController(Settings(ml_job_root=tmp_path / "jobs", runtime_discovery_path=tmp_path / "discovery.json"))
    probes: list[tuple[ExecutionBackend, RuntimeApi]] = []
    try:
        monkeypatch.setattr(runtime, "_environment_fingerprint", lambda: ("test-fingerprint", {}))
        monkeypatch.setattr(
            runtime,
            "_load_discovery_snapshot",
            lambda _fingerprint: {
                ExecutionBackend.PYTORCH: ExecutionBackendStatus(
                    id=ExecutionBackend.PYTORCH,
                    installed=True,
                    available=True,
                    runtime_apis=[
                        RuntimeApiStatus(
                            id=RuntimeApi.KERAS,
                            available=True,
                            state=RuntimeDiscoveryState.VERIFIED,
                            source="snapshot",
                        )
                    ],
                )
            },
        )

        def fake_probe(backend: ExecutionBackend, api: RuntimeApi) -> tuple[bool, str | None, dict[str, object]]:
            probes.append((backend, api))
            expected = keras_backend_value(backend) if api is RuntimeApi.KERAS else backend.value
            return True, None, {
                "backend": expected,
                "available": True,
                "devices": [
                    RuntimeDevice(
                        device_id="cpu",
                        lease_key="cpu",
                        kind="cpu",
                        name="CPU",
                        accelerator="cpu",
                        supported_memory_policies=[MemoryPolicyKind.DEFAULT],
                    ).model_dump(mode="json")
                ],
            }

        monkeypatch.setattr(runtime, "_probe_backend_api", fake_probe)
        runtime.request_discovery(ExecutionBackend.PYTORCH, RuntimeApi.KERAS, force_refresh=True)
        result = runtime.discovery_status(ExecutionBackend.PYTORCH, RuntimeApi.KERAS)
        for _ in range(100):
            result = runtime.discovery_status(ExecutionBackend.PYTORCH, RuntimeApi.KERAS)
            if result.state is not RuntimeDiscoveryState.CHECKING:
                break
            time.sleep(0.01)
        assert probes == [(ExecutionBackend.PYTORCH, RuntimeApi.KERAS)]
        assert result.state is RuntimeDiscoveryState.VERIFIED
        assert result.source == "live"
    finally:
        runtime.shutdown()

###############################################################################
def test_health_and_metadata_do_not_probe_optional_frameworks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    app_factory: Callable[..., FastAPI],
) -> None:
    def fail_probe(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("optional framework probe ran during a health or metadata read")

    monkeypatch.setattr(RuntimeController, "_run_probe", fail_probe)
    with TestClient(app_factory(database_path=tmp_path / "database.db")) as client:
        assert client.get("/health").json() == {"status": "ok"}
        metadata = client.get("/api/metadata")
        assert metadata.status_code == 200
        assert all(item["state"] in {"unchecked", "verified", "unavailable", "failed"} for item in metadata.json()["runtime_apis"])

###############################################################################
def test_probe_timeout_is_reported_and_shutdown_joins_discovery_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RuntimeController(Settings(ml_job_root=tmp_path / "jobs", runtime_discovery_path=tmp_path / "discovery.json"))
    try:
        monkeypatch.setattr(runtime, "_environment_fingerprint", lambda: ("test-fingerprint", {}))
        monkeypatch.setattr(runtime, "_load_discovery_snapshot", lambda _fingerprint: None)
        monkeypatch.setattr(runtime, "_probe_backend_api", lambda _backend, _api: (False, "probe_timeout", {"message": "deadline"}))
        runtime.request_discovery(ExecutionBackend.PYTORCH, RuntimeApi.KERAS)
        result = runtime.discovery_status(ExecutionBackend.PYTORCH, RuntimeApi.KERAS)
        for _ in range(100):
            result = runtime.discovery_status(ExecutionBackend.PYTORCH, RuntimeApi.KERAS)
            if result.state is not RuntimeDiscoveryState.CHECKING:
                break
            time.sleep(0.01)
        assert result.state is RuntimeDiscoveryState.UNAVAILABLE
        assert result.reason_code == "probe_timeout"
        future = runtime._discovery_futures.get((ExecutionBackend.PYTORCH, RuntimeApi.KERAS))  # pyright: ignore[reportPrivateUsage]
        assert future is None or future.done()
    finally:
        runtime.shutdown()

###############################################################################
def test_job_submission_starts_discovery_and_returns_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = RuntimeController(Settings(ml_job_root=tmp_path / "jobs", runtime_discovery_path=tmp_path / "discovery.json"))
    try:
        monkeypatch.setattr(runtime, "_environment_fingerprint", lambda: ("test-fingerprint", {}))
        monkeypatch.setattr(runtime, "_load_discovery_snapshot", lambda _fingerprint: None)
        monkeypatch.setattr(runtime, "_probe_backend_api", lambda _backend, _api: (False, "dependency_missing", {}))
        request = JobSubmitRequest(
            operation=SmokeInferenceOperation(project_id=uuid4(), graph=ModelGraph()),
            worker=WorkerConfiguration(backend=ExecutionBackend.PYTORCH, device_id="cpu"),
        )
        with pytest.raises(RuntimeControllerError) as error:
            runtime.submit(request, graph=ModelGraph())
        assert error.value.code == "runtime_discovery_pending"
        for _ in range(100):
            if runtime.discovery_status(ExecutionBackend.PYTORCH, RuntimeApi.KERAS).state is not RuntimeDiscoveryState.CHECKING:
                break
            time.sleep(0.01)
        assert runtime.discovery_status(ExecutionBackend.PYTORCH, RuntimeApi.KERAS).reason_code == "dependency_missing"
    finally:
        runtime.shutdown()
