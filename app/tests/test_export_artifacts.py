# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import services.runtime_controller as runtime_controller_module
from configurations.settings import Settings
from domain.enums import ExecutionBackend
from domain.exports import (
    ExportArtifactFile,
    ExportArtifactManifest,
    ExportConversion,
    ExportFormat,
    ExportLayout,
    ExportPayload,
)
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from domain.jobs import JobResponse, JobSubmitRequest, ModelExportOperation
from domain.runtime import MemoryPolicy, RuntimeApi, WorkerConfiguration
from services.runtime_controller import RuntimeController


###############################################################################
@pytest.fixture(scope="module")
def export_controller(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RuntimeController]:
    runtime_root = tmp_path_factory.mktemp("export-controller")
    controller = RuntimeController(
        Settings(
            ml_job_root=runtime_root / "jobs",
            export_root=runtime_root / "exports",
            ml_cancel_grace_seconds=1,
            ml_terminate_grace_seconds=1,
        )
    )
    controller.probe_statuses()
    yield controller
    controller.shutdown()

###############################################################################
@pytest.mark.parametrize(
    ("backend", "api", "export_format"),
    [
        (ExecutionBackend.TENSORFLOW, RuntimeApi.KERAS, ExportFormat.KERAS),
        (ExecutionBackend.TENSORFLOW, RuntimeApi.KERAS, ExportFormat.KERAS_WEIGHTS),
        (ExecutionBackend.TENSORFLOW, RuntimeApi.KERAS, ExportFormat.TENSORFLOW_SAVED_MODEL),
        (ExecutionBackend.TENSORFLOW, RuntimeApi.KERAS, ExportFormat.ONNX),
        (ExecutionBackend.TENSORFLOW, RuntimeApi.KERAS, ExportFormat.SAFETENSORS),
        (ExecutionBackend.TENSORFLOW, RuntimeApi.NATIVE, ExportFormat.TENSORFLOW_SAVED_MODEL),
        (ExecutionBackend.TENSORFLOW, RuntimeApi.NATIVE, ExportFormat.ONNX),
        (ExecutionBackend.TENSORFLOW, RuntimeApi.NATIVE, ExportFormat.SAFETENSORS),
        (ExecutionBackend.TENSORFLOW, RuntimeApi.NATIVE, ExportFormat.TENSORFLOW_CHECKPOINT),
        (ExecutionBackend.PYTORCH, RuntimeApi.KERAS, ExportFormat.KERAS),
        (ExecutionBackend.PYTORCH, RuntimeApi.KERAS, ExportFormat.KERAS_WEIGHTS),
        (ExecutionBackend.PYTORCH, RuntimeApi.KERAS, ExportFormat.ONNX),
        (ExecutionBackend.PYTORCH, RuntimeApi.KERAS, ExportFormat.SAFETENSORS),
        (ExecutionBackend.PYTORCH, RuntimeApi.NATIVE, ExportFormat.PYTORCH_EXPORTED_PROGRAM),
        (ExecutionBackend.PYTORCH, RuntimeApi.NATIVE, ExportFormat.ONNX),
        (ExecutionBackend.PYTORCH, RuntimeApi.NATIVE, ExportFormat.SAFETENSORS),
        (ExecutionBackend.JAX, RuntimeApi.KERAS, ExportFormat.KERAS),
        (ExecutionBackend.JAX, RuntimeApi.KERAS, ExportFormat.KERAS_WEIGHTS),
        (ExecutionBackend.JAX, RuntimeApi.KERAS, ExportFormat.ONNX),
        (ExecutionBackend.JAX, RuntimeApi.KERAS, ExportFormat.SAFETENSORS),
        (ExecutionBackend.JAX, RuntimeApi.NATIVE, ExportFormat.FLAX_CHECKPOINT),
        (ExecutionBackend.JAX, RuntimeApi.NATIVE, ExportFormat.SAFETENSORS),
    ],
)
def test_export_artifact_matrix(
    export_controller: RuntimeController,
    tmp_path: Path,
    backend: ExecutionBackend,
    api: RuntimeApi,
    export_format: ExportFormat,
) -> None:
    worker = worker_for(export_controller, backend, api)
    if worker is None:
        pytest.skip(f"{api.value} {backend.value} runtime is unavailable")
    destination = tmp_path / f"artifact{extension_for(export_format)}"
    project_id = uuid4()
    job = export_controller.submit(
        JobSubmitRequest(
            operation=ModelExportOperation(
                project_id=project_id,
                format=export_format,
                filename_stem="matrix-artifact",
            ),
            worker=worker,
        ),
        graph=dense_graph(),
        artifact_destination=destination,
        artifact_filename=destination.name,
        project_revision=1,
    )
    result = wait_for_job(export_controller, job.id)
    assert result.state.value == "succeeded", result
    assert result.artifact is not None
    assert destination.is_file()
    assert result.artifact.format is export_format
    assert result.artifact.execution_backend is backend
    assert result.artifact.runtime_api is api
    assert result.artifact.project_id == project_id
    assert result.artifact.project_revision == 1
    assert result.artifact.verification.structural
    assert result.artifact.verification.reload
    assert result.artifact.verification.inference
    assert result.artifact.compilation["native"]["pytorch"]["compilation"]["optimizer"]["id"] == "adam"

    file_hashes = verify_published_artifact(
        destination,
        result.artifact,
        backend=backend,
        api=api,
        project_id=project_id,
        revision=1,
    )
    record_matrix_evidence(
        {
            "backend": backend.value,
            "api": api.value,
            "device_id": worker.device_id,
            "precision": worker.precision,
            "format": export_format.value,
            "project_id": str(project_id),
            "project_revision": result.artifact.project_revision,
            "filename": destination.name,
            "published_size_bytes": destination.stat().st_size,
            "published_sha256": sha256_bytes(destination.read_bytes()),
            "files": file_hashes,
            "manifest": result.artifact.model_dump(mode="json"),
            "verification": result.artifact.verification.model_dump(mode="json"),
        }
    )

###############################################################################
def test_conversion_failure_preserves_published_artifact_and_cleans_job(
    export_controller: RuntimeController,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ExecutionBackend.TENSORFLOW
    api = RuntimeApi.KERAS
    worker = worker_for(export_controller, backend, api)
    if worker is None:
        pytest.skip(f"{api.value} {backend.value} runtime is unavailable")
    destination = tmp_path / "existing.keras"
    original = b"previously published artifact"
    destination.write_bytes(original)
    job_dirs: list[Path] = []

    def start_failed_worker(job_dir: Path) -> _FinishedWorker:
        job_dirs.append(job_dir)
        (job_dir / "worker.log").write_text("simulated exporter conversion failure", encoding="utf-8")
        (job_dir / "status.json").write_text(
            json.dumps(
                {
                    "state": "failed",
                    "phase": "conversion",
                    "message": "Export conversion failed",
                    "error": {"code": "export_conversion_failed", "message": "simulated exporter conversion failure"},
                }
            ),
            encoding="utf-8",
        )
        return _FinishedWorker(returncode=1)

    monkeypatch.setattr(export_controller, "_start_process", start_failed_worker)
    job = export_controller.submit(
        _export_request(backend, api),
        graph=dense_graph(),
        artifact_destination=destination,
        artifact_filename=destination.name,
        project_revision=1,
    )
    result = wait_for_job(export_controller, job.id)

    assert result.state.value == "failed"
    assert result.error is not None and result.error.code == "export_conversion_failed"
    assert destination.read_bytes() == original
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))
    assert job_dirs and not job_dirs[0].exists()
    assert export_controller.list(active_only=True) == []

###############################################################################
def test_atomic_publication_failure_preserves_prior_artifact_and_cleans_temp(
    export_controller: RuntimeController,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ExecutionBackend.TENSORFLOW
    api = RuntimeApi.KERAS
    worker = worker_for(export_controller, backend, api)
    if worker is None:
        pytest.skip(f"{api.value} {backend.value} runtime is unavailable")
    project_id = uuid4()
    destination = tmp_path / "existing.keras"
    original = b"previously published artifact"
    destination.write_bytes(original)
    job_dirs: list[Path] = []

    def start_successful_worker(job_dir: Path) -> _FinishedWorker:
        job_dirs.append(job_dir)
        source = job_dir / "staged.keras"
        payload = b"verified staged artifact bytes"
        source.write_bytes(payload)
        manifest = ExportArtifactManifest(
            format=ExportFormat.KERAS,
            filename=destination.name,
            layout=ExportLayout.FILE,
            payload=ExportPayload.FULL_MODEL,
            conversion=ExportConversion.KERAS,
            execution_backend=backend,
            runtime_api=api,
            project_id=project_id,
            project_revision=1,
            files=[
                ExportArtifactFile(
                    path=source.name,
                    size_bytes=len(payload),
                    sha256=sha256_bytes(payload),
                )
            ],
        )
        (job_dir / "status.json").write_text(
            json.dumps(
                {
                    "state": "succeeded",
                    "phase": "completed",
                    "message": "Export conversion completed",
                    "result": {
                        "artifact_path": source.name,
                        "artifact_manifest": manifest.model_dump(mode="json"),
                    },
                }
            ),
            encoding="utf-8",
        )
        return _FinishedWorker(returncode=0)

    real_replace = os.replace

    def fail_atomic_publish(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        if Path(target) == destination:
            raise OSError("simulated atomic publication failure")
        real_replace(source, target)

    monkeypatch.setattr(export_controller, "_start_process", start_successful_worker)
    monkeypatch.setattr(runtime_controller_module.os, "replace", fail_atomic_publish)
    job = export_controller.submit(
        _export_request(backend, api, project_id=project_id),
        graph=dense_graph(),
        artifact_destination=destination,
        artifact_filename=destination.name,
        project_revision=1,
    )
    result = wait_for_job(export_controller, job.id)

    assert result.state.value == "failed"
    assert result.error is not None and result.error.code == "artifact_publish_failed"
    assert destination.read_bytes() == original
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))
    assert job_dirs and not job_dirs[0].exists()
    assert export_controller.list(active_only=True) == []

###############################################################################
def test_cancelled_export_preserves_published_artifact_and_cleans_worker(
    export_controller: RuntimeController,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ExecutionBackend.TENSORFLOW
    api = RuntimeApi.KERAS
    worker = worker_for(export_controller, backend, api)
    if worker is None:
        pytest.skip(f"{api.value} {backend.value} runtime is unavailable")
    destination = tmp_path / "existing.keras"
    original = b"previously published artifact"
    destination.write_bytes(original)
    job_dirs: list[Path] = []
    running_worker = _FinishedWorker(returncode=None)

    def start_running_worker(job_dir: Path) -> _FinishedWorker:
        job_dirs.append(job_dir)
        (job_dir / "status.json").write_text(
            json.dumps({"state": "running", "phase": "conversion", "message": "Exporting"}),
            encoding="utf-8",
        )
        return running_worker

    def terminate_fake_worker(_pid: int, *, force: bool) -> None:
        assert not force
        running_worker.returncode = 20

    monkeypatch.setattr(export_controller, "_start_process", start_running_worker)
    monkeypatch.setattr(runtime_controller_module, "terminate_process_tree", terminate_fake_worker)
    job = export_controller.submit(
        _export_request(backend, api),
        graph=dense_graph(),
        artifact_destination=destination,
        artifact_filename=destination.name,
        project_revision=1,
    )
    export_controller.stop(job.id)
    result = wait_for_job(export_controller, job.id)

    assert result.state.value == "terminated"
    assert result.error is not None and result.error.code == "cancelled"
    wait_for_workspace_removed(job_dirs[0])
    assert destination.read_bytes() == original
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))
    assert job_dirs and not job_dirs[0].exists()
    assert export_controller.list(active_only=True) == []

###############################################################################
class _FinishedWorker:
    def __init__(self, *, returncode: int | None) -> None:
        self.pid: int | None = None
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9

###############################################################################
def _export_request(
    backend: ExecutionBackend,
    api: RuntimeApi,
    *,
    project_id: UUID | None = None,
) -> JobSubmitRequest:
    return JobSubmitRequest(
        operation=ModelExportOperation(
            project_id=project_id or uuid4(),
            format=ExportFormat.KERAS,
        ),
        worker=WorkerConfiguration(
            api=api,
            backend=backend,
            device_id="cpu",
            memory_policy=MemoryPolicy(),
            precision="float32",
        ),
    )

###############################################################################
def verify_published_artifact(
    destination: Path,
    artifact: ExportArtifactManifest,
    *,
    backend: ExecutionBackend,
    api: RuntimeApi,
    project_id: UUID,
    revision: int,
) -> list[dict[str, object]]:
    assert artifact.filename == destination.name
    assert artifact.execution_backend is backend
    assert artifact.runtime_api is api
    assert artifact.project_id == project_id
    assert artifact.project_revision == revision
    file_hashes: list[dict[str, object]] = []
    if artifact.layout is ExportLayout.ARCHIVE:
        with zipfile.ZipFile(destination) as archive:
            names = set(archive.namelist())
            expected_names = {"manifest.json", *(item.path for item in artifact.files)}
            assert names == expected_names
            archive_manifest = json.loads(archive.read("manifest.json"))
            packaged_artifact = archive_manifest["artifact"]
            assert packaged_artifact == artifact.model_dump(mode="json")
            assert packaged_artifact["format"] == artifact.format.value
            assert packaged_artifact["execution_backend"] == backend.value
            assert packaged_artifact["runtime_api"] == api.value
            assert packaged_artifact["project_id"] == str(project_id)
            assert packaged_artifact["project_revision"] == revision
            for declared in artifact.files:
                assert declared.path in names
                payload = archive.read(declared.path)
                digest = sha256_bytes(payload)
                assert len(payload) == declared.size_bytes
                assert digest == declared.sha256
                file_hashes.append(
                    {"path": declared.path, "size_bytes": len(payload), "sha256": digest}
                )
    else:
        assert len(artifact.files) == 1
        declared = artifact.files[0]
        assert declared.path == destination.name
        payload = destination.read_bytes()
        digest = sha256_bytes(payload)
        assert len(payload) == declared.size_bytes
        assert digest == declared.sha256
        file_hashes.append(
            {"path": declared.path, "size_bytes": len(payload), "sha256": digest}
        )
    return file_hashes

###############################################################################
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

###############################################################################
def wait_for_workspace_removed(path: Path) -> None:
    deadline = time.monotonic() + 5
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not path.exists()

###############################################################################
def record_matrix_evidence(record: dict[str, object]) -> None:
    evidence_file = os.environ.get("NW_EXPORT_MATRIX_EVIDENCE_PATH")
    if evidence_file:
        target = Path(evidence_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

###############################################################################
def worker_for(
    controller: RuntimeController,
    backend: ExecutionBackend,
    api: RuntimeApi,
) -> WorkerConfiguration | None:
    status = controller.backend_statuses.get(backend)
    if status is None or not status.available:
        return None
    api_status = next((item for item in status.runtime_apis if item.id is api), None)
    if api_status is None or not api_status.available:
        return None
    device = next((item for item in status.devices if item.available), None)
    if device is None:
        return None
    return WorkerConfiguration(
        api=api,
        backend=backend,
        device_id=device.device_id,
        memory_policy=MemoryPolicy(),
        precision="float32",
    )

###############################################################################
def wait_for_job(controller: RuntimeController, job_id: str) -> JobResponse:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        result = controller.get(job_id)
        if result.state.value in {"succeeded", "terminated", "failed"}:
            return result
        time.sleep(0.05)
    controller.stop(job_id)
    raise TimeoutError(f"Export job {job_id} timed out")

###############################################################################
def extension_for(export_format: ExportFormat) -> str:
    return {
        ExportFormat.KERAS: ".keras",
        ExportFormat.KERAS_WEIGHTS: ".weights.h5",
        ExportFormat.TENSORFLOW_SAVED_MODEL: ".savedmodel.zip",
        ExportFormat.ONNX: ".onnx",
        ExportFormat.SAFETENSORS: ".safetensors.zip",
        ExportFormat.PYTORCH_EXPORTED_PROGRAM: ".pt2",
        ExportFormat.TENSORFLOW_CHECKPOINT: ".tf-checkpoint.zip",
        ExportFormat.FLAX_CHECKPOINT: ".flax-checkpoint.zip",
    }[export_format]

###############################################################################
def dense_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            NodeInstance(
                id="input",
                definition_id="input.tabular",
                definition_version=1,
                position=Position(x=0, y=0),
                parameters={"shape": [4], "dtype": "float32"},
            ),
            NodeInstance(
                id="dense",
                definition_id="layer.dense",
                definition_version=1,
                position=Position(x=300, y=0),
                parameters={"units": 3, "activation": "softmax"},
            ),
            NodeInstance(
                id="output",
                definition_id="output.classification",
                definition_version=1,
                position=Position(x=600, y=0),
                parameters={},
            ),
        ],
        edges=[
            EdgeInstance(
                id="input-dense",
                source_node_id="input",
                source_port_id="output",
                target_node_id="dense",
                target_port_id="input",
            ),
            EdgeInstance(
                id="dense-output",
                source_node_id="dense",
                source_port_id="output",
                target_node_id="output",
                target_port_id="input",
            ),
        ],
    )
