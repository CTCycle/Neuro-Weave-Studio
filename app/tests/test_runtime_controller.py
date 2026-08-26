from __future__ import annotations

import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from uuid import uuid4

import pytest

from configurations.settings import Settings
from domain.enums import ExecutionBackend
from domain.graph import ModelGraph
from domain.jobs import JobResponse, JobSubmitRequest, SmokeInferenceOperation
from domain.metadata import ExecutionBackendStatus
from domain.runtime import MemoryPolicy, MemoryPolicyKind, RuntimeDevice, WorkerConfiguration
from services.runtime_controller import RuntimeController, RuntimeControllerError


###############################################################################
def _status(*devices: RuntimeDevice) -> ExecutionBackendStatus:
    return ExecutionBackendStatus(
        id=ExecutionBackend.PYTORCH,
        installed=True,
        version="test",
        available=True,
        accelerator="cpu",
        devices=list(devices),
    )

###############################################################################
def _fake_start(job_dir: Path, sleep_seconds: float) -> subprocess.Popen[bytes]:
    code = (
        "import json, pathlib, sys, time; "
        "d=pathlib.Path(sys.argv[1]); "
        "(d/'status.json').write_text(json.dumps({'state':'running','phase':'inference','message':'fake worker','percent':50,'pid':__import__('os').getpid()})); "
        f"time.sleep({sleep_seconds}); "
        "(d/'status.json').write_text(json.dumps({'state':'succeeded','phase':'finished','message':'fake complete','percent':100,'pid':__import__('os').getpid(),'result':{'ok':True}}))"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code, str(job_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

###############################################################################
def _controller(
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
    controller = RuntimeController(settings)
    cpu = RuntimeDevice(device_id="cpu", lease_key="cpu", kind="cpu", name="CPU", accelerator="cpu", supported_memory_policies=[MemoryPolicyKind.DEFAULT])
    gpu = RuntimeDevice(device_id="cuda:0", lease_key="cuda:0", kind="cuda", name="Test GPU", accelerator="cuda", supported_memory_policies=[MemoryPolicyKind.DEFAULT])
    controller._backend_statuses = {ExecutionBackend.PYTORCH: _status(cpu, gpu)}  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(
        controller,
        "_start_process",
        partial(_fake_start, sleep_seconds=sleep_seconds),
    )
    return controller

###############################################################################
def _request(device_id: str = "cpu") -> JobSubmitRequest:
    return JobSubmitRequest(
        operation=SmokeInferenceOperation(project_id=uuid4(), graph=ModelGraph()),
        worker=WorkerConfiguration(backend=ExecutionBackend.PYTORCH, device_id=device_id),
    )

###############################################################################
def _wait(controller: RuntimeController, job_id: str) -> JobResponse:
    for _ in range(320):
        job = controller.get(job_id)
        if job.state.value in {"succeeded", "terminated", "failed"}:
            return job
        time.sleep(0.02)
    pytest.fail(f"job {job_id} did not finish")

###############################################################################
def test_worker_configuration_is_frozen_and_policy_parameters_are_strict() -> None:
    configuration = WorkerConfiguration(backend=ExecutionBackend.PYTORCH, device_id="cpu")
    with pytest.raises((TypeError, ValueError)):
        configuration.device_id = "cuda:0"  # type: ignore[misc]
    with pytest.raises(ValueError):
        MemoryPolicy(kind=MemoryPolicyKind.DEFAULT, fraction=0.5)
    with pytest.raises(ValueError):
        MemoryPolicy(kind=MemoryPolicyKind.TENSORFLOW_LIMIT_MB)

###############################################################################
def test_leases_are_exclusive_per_device_but_different_devices_can_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _controller(tmp_path, monkeypatch, sleep_seconds=0.35)
    first = controller.submit(_request("cpu"), graph=ModelGraph())
    with pytest.raises(RuntimeControllerError, match="already has an active"):
        controller.submit(_request("cpu"), graph=ModelGraph())
    second = controller.submit(_request("cuda:0"), graph=ModelGraph())
    assert second.device_lease.device_id == "cuda:0"
    assert _wait(controller, first.id).state.value == "succeeded"
    assert _wait(controller, second.id).state.value == "succeeded"
    replacement = controller.submit(_request("cpu"), graph=ModelGraph())
    assert replacement.restart_reason == "replacement_job"
    _wait(controller, replacement.id)
    controller.shutdown()

###############################################################################
def test_stop_is_idempotent_and_reaps_before_lease_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _controller(tmp_path, monkeypatch, sleep_seconds=10)
    job = controller.submit(_request(), graph=ModelGraph())
    first = controller.stop(job.id)
    second = controller.stop(job.id)
    assert first.cancellation_requested is True
    assert second.cancellation_requested is True
    terminal = _wait(controller, job.id)
    assert terminal.state.value == "terminated"
    assert terminal.device_lease.state == "idle"
    replacement = controller.submit(_request(), graph=ModelGraph())
    controller.stop(replacement.id)
    _wait(controller, replacement.id)
    controller.shutdown()

###############################################################################
def test_terminal_history_is_bounded_and_orphan_workspace_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _controller(tmp_path, monkeypatch, sleep_seconds=0.01, history_limit=10)
    for _ in range(11):
        job = controller.submit(_request(), graph=ModelGraph())
        _wait(controller, job.id)
    records = controller.list()
    assert len(records) == 10
    for _ in range(50):
        if not any(path.is_dir() for path in (tmp_path / "jobs").iterdir()):
            break
        time.sleep(0.02)
    assert not any(path.is_dir() for path in (tmp_path / "jobs").iterdir())
    controller.shutdown()
