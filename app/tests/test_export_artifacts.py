from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from uuid import uuid4

import pytest

from configurations.settings import Settings
from domain.enums import ExecutionBackend
from domain.exports import ExportFormat
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from domain.jobs import JobResponse, JobSubmitRequest, ModelExportOperation
from domain.runtime import RuntimeApi, WorkerConfiguration
from services.runtime_controller import RuntimeController


###############################################################################
@pytest.fixture(scope="module")
def export_controller(tmp_path_factory: pytest.TempPathFactory) -> RuntimeController:
    runtime_root = tmp_path_factory.mktemp("export-controller")
    controller = RuntimeController(
        Settings(
            ml_job_root=runtime_root / "jobs",
            export_root=runtime_root / "exports",
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
        (ExecutionBackend.PYTORCH, RuntimeApi.NATIVE, ExportFormat.TORCHSCRIPT),
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
    job = export_controller.submit(
        JobSubmitRequest(
            operation=ModelExportOperation(
                project_id=uuid4(),
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
    assert result.artifact.project_revision == 1
    assert result.artifact.verification.structural
    assert result.artifact.verification.reload
    assert result.artifact.verification.inference

    if result.artifact.layout.value == "archive":
        with zipfile.ZipFile(destination) as archive:
            names = set(archive.namelist())
            assert "manifest.json" in names
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["artifact"]["format"] == export_format.value
            assert all(item.path in names for item in result.artifact.files)
    else:
        assert len(result.artifact.files) == 1
        assert result.artifact.files[0].path == destination.name


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
        memory_policy={"kind": "default"},
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
        ExportFormat.TORCHSCRIPT: ".pt",
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
                parameters={"shape": [None, 4], "dtype": "float32"},
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
