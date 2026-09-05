# pyright: reportUnknownLambdaType=false, reportUnknownArgumentType=false
from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import create_app
from configurations.settings import Settings
from domain.enums import ExecutionBackend, keras_backend_value
from domain.graph import ModelGraph
from domain.jobs import JobSubmitRequest, SmokeInferenceOperation
from domain.metadata import ExecutionBackendStatus
from domain.runtime import (
    RuntimeApi,
    RuntimeApiStatus,
    RuntimeDevice,
    RuntimeDiscoveryState,
    WorkerConfiguration,
)
from services.runtime_controller import RuntimeController, RuntimeControllerError


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
                        supported_memory_policies=["default"],
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


def test_health_and_metadata_do_not_probe_optional_frameworks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_probe(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("optional framework probe ran during a health or metadata read")

    monkeypatch.setattr(RuntimeController, "_run_probe", fail_probe)
    monkeypatch.setenv("DATABASE_SQLITE_PATH", str(tmp_path / "database.db"))
    monkeypatch.setenv("NEURO_WEAVE_ML_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("NEURO_WEAVE_EXPORT_ROOT", str(tmp_path / "exports"))
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        metadata = client.get("/api/metadata")
        assert metadata.status_code == 200
        assert all(item["state"] in {"unchecked", "verified", "unavailable", "failed"} for item in metadata.json()["runtime_apis"])


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


def test_preview_server_serves_spa_root_without_crashing() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    node = repository_root / "runtimes" / "node" / "node.exe"
    preview = repository_root / "app" / "client" / "tools" / "preview-server.mjs"
    index = repository_root / "app" / "tests" / "cache" / "frontend-dist" / "index.html"
    if not node.is_file() or not preview.is_file() or not index.is_file():
        pytest.skip("official frontend preview prerequisites are not installed")

    import urllib.error
    import urllib.request

    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        port = probe_socket.getsockname()[1]
    process = subprocess.Popen(
        [str(node), str(preview), "--host", "127.0.0.1", "--port", str(port)],
        cwd=repository_root / "app" / "client",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(30):
            if process.poll() is not None:
                diagnostics = process.stderr.read() if process.stderr else ""
                raise AssertionError(diagnostics)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
                    assert response.status == 200
                break
            except (ConnectionError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            raise AssertionError("preview server did not become ready")
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/missing.js", timeout=1)
        assert missing.value.code == 404
        missing.value.close()
        assert process.poll() is None
    finally:
        try:
            process.terminate()
            process.wait(timeout=3)
        finally:
            if process.stderr is not None:
                process.stderr.close()
