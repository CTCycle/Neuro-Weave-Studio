from __future__ import annotations

import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from common.process_control import process_start_options
from configurations.settings import Settings
from domain.enums import ExecutionBackend
from domain.graph import ModelGraph
from domain.jobs import JobResponse, JobSubmitRequest, SmokeInferenceOperation
from domain.metadata import ExecutionBackendStatus
from domain.runtime import (
    MemoryPolicyKind,
    RuntimeApi,
    RuntimeApiStatus,
    RuntimeDevice,
    WorkerConfiguration,
)
from services.runtime_controller import RuntimeController, RuntimeControllerError


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
