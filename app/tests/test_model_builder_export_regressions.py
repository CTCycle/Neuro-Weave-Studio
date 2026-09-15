from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import pytest

from configurations.settings import Settings
from domain.enums import ExecutionBackend
from domain.exports import ExportConversion, ExportFormat, ExportOptions
from domain.graph import (
    EdgeInstance,
    JaxNativeConfiguration,
    ModelConfiguration,
    ModelGraph,
    NativeCompilationConfiguration,
    NodeInstance,
    Position,
    TensorFlowNativeConfiguration,
)
from domain.jobs import (
    JobResponse,
    JobSubmitRequest,
    ModelExportOperation,
    SmokeInferenceOperation,
)
from domain.runtime import MemoryPolicy, RuntimeApi, WorkerConfiguration
from services.runtime_controller import RuntimeController


###############################################################################
def _node(node_id: str, definition_id: str, parameters: dict[str, object]) -> NodeInstance:
    return NodeInstance(
        id=node_id,
        definition_id=definition_id,
        position=Position(x=0, y=0),
        parameters=parameters,
    )


###############################################################################
def _edge(edge_id: str, source: str, target: str) -> EdgeInstance:
    return EdgeInstance(
        id=edge_id,
        source_node_id=source,
        source_port_id="output",
        target_node_id=target,
        target_port_id="input",
    )


###############################################################################
def _dense_graph(*, jit_compile: bool = False) -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("input", "input.tabular", {"shape": [4], "dtype": "float32"}),
            _node("dense", "layer.dense", {"units": 3, "activation": "softmax"}),
            _node("output", "output.classification", {}),
        ],
        edges=[_edge("input-dense", "input", "dense"), _edge("dense-output", "dense", "output")],
        model_configuration=ModelConfiguration(
            native=NativeCompilationConfiguration(
                tensorflow=TensorFlowNativeConfiguration(jit_compile=jit_compile),
            ),
        ),
    )


###############################################################################
def _jax_transformer_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("tokens", "input.token", {"shape": [8], "dtype": "int32"}),
            _node(
                "embedding",
                "layer.embedding",
                {"vocabulary_size": 32, "embedding_dim": 4},
            ),
            _node(
                "position",
                "transformer.positional_encoding",
                {"mode": "sinusoidal"},
            ),
            _node(
                "encoder",
                "transformer.encoder_block",
                {
                    "num_heads": 2,
                    "key_dim": 2,
                    "ffn_multiplier": 2,
                    "layers": 1,
                    "activation": "gelu",
                },
            ),
            _node("output", "output.sequence", {}),
        ],
        edges=[
            _edge("tokens-embedding", "tokens", "embedding"),
            _edge("embedding-position", "embedding", "position"),
            _edge("position-encoder", "position", "encoder"),
            _edge("encoder-output", "encoder", "output"),
        ],
    )


###############################################################################
def _native_jax_multi_input_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("left", "input.tabular", {"shape": [4], "dtype": "float32"}),
            _node("right", "input.tabular", {"shape": [4], "dtype": "float32"}),
            _node("concat", "merge.concatenate", {"axis": -1}),
            _node("dense", "layer.dense", {"units": 3, "activation": "softmax"}),
            _node("output", "output.classification", {}),
        ],
        edges=[
            EdgeInstance(
                id="left-concat",
                source_node_id="left",
                source_port_id="output",
                target_node_id="concat",
                target_port_id="input_0",
            ),
            EdgeInstance(
                id="right-concat",
                source_node_id="right",
                source_port_id="output",
                target_node_id="concat",
                target_port_id="input_1",
            ),
            _edge("concat-dense", "concat", "dense"),
            _edge("dense-output", "dense", "output"),
        ],
        model_configuration=ModelConfiguration(
            native=NativeCompilationConfiguration(
                jax=JaxNativeConfiguration(jit_enabled=True),
            ),
        ),
    )


###############################################################################
def _worker(controller: RuntimeController, backend: ExecutionBackend, api: RuntimeApi) -> WorkerConfiguration:
    status = controller.backend_statuses.get(backend)
    if status is None or not status.available:
        pytest.skip(f"{backend.value} runtime is unavailable")
    device = next((item for item in status.devices if item.available), None)
    if device is None:
        pytest.skip(f"{backend.value} has no available device")
    api_status = next((item for item in status.runtime_apis if item.id is api), None)
    if api_status is None or not api_status.available:
        pytest.skip(f"{api.value}/{backend.value} runtime API is unavailable")
    return WorkerConfiguration(
        api=api,
        backend=backend,
        device_id=device.device_id,
        memory_policy=MemoryPolicy(),
        precision="float32",
    )


###############################################################################
def _wait(controller: RuntimeController, job_id: str) -> JobResponse:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        result = controller.get(job_id)
        if result.state.value in {"succeeded", "failed", "terminated"}:
            return result
        time.sleep(0.05)
    controller.stop(job_id)
    raise TimeoutError(f"ML worker job {job_id} timed out")


###############################################################################
def _export(
    controller: RuntimeController,
    graph: ModelGraph,
    backend: ExecutionBackend,
    api: RuntimeApi,
    export_format: ExportFormat,
    destination: Path,
) -> JobResponse:
    worker = _worker(controller, backend, api)
    job = controller.submit(
        JobSubmitRequest(
            operation=ModelExportOperation(
                project_id=uuid4(),
                format=export_format,
                filename_stem=destination.stem,
                options=ExportOptions(opset_version=18, dynamic_batch=True),
            ),
            worker=worker,
        ),
        graph=graph,
        artifact_destination=destination,
        artifact_filename=destination.name,
        project_revision=1,
    )
    result = _wait(controller, job.id)
    assert result.state.value == "succeeded", (
        result.error.message if result.error is not None else result
    )
    assert result.artifact is not None
    assert result.artifact.verification.structural
    assert result.artifact.verification.reload
    assert result.artifact.verification.inference
    assert destination.is_file()
    return result


###############################################################################
@pytest.fixture
def controller(tmp_path: Path):
    runtime = RuntimeController(
        Settings(
            ml_job_root=tmp_path / "jobs",
            export_root=tmp_path / "exports",
        )
    )
    runtime.probe_statuses(force_refresh=True)
    yield runtime
    runtime.shutdown()


###############################################################################
def test_native_tensorflow_onnx_export_inlines_jit_call_graph(
    controller: RuntimeController,
    tmp_path: Path,
) -> None:
    _export(
        controller,
        _dense_graph(jit_compile=True),
        ExecutionBackend.TENSORFLOW,
        RuntimeApi.NATIVE,
        ExportFormat.ONNX,
        tmp_path / "native_tensorflow_jit.onnx",
    )


###############################################################################
def test_native_smoke_reports_training_not_run(
    controller: RuntimeController,
) -> None:
    graph = _dense_graph()
    worker = _worker(controller, ExecutionBackend.TENSORFLOW, RuntimeApi.NATIVE)
    job = controller.submit(
        JobSubmitRequest(
            operation=SmokeInferenceOperation(project_id=uuid4(), graph=graph),
            worker=worker,
        ),
        graph=graph,
        project_revision=1,
    )
    result = _wait(controller, job.id)
    assert result.state.value == "succeeded", (
        result.error.message if result.error is not None else result
    )
    assert result.result is not None
    assert result.result["training"] == "not_run"


###############################################################################
def test_keras_jax_positional_encoding_onnx_export_uses_static_sequence_shape(
    controller: RuntimeController,
    tmp_path: Path,
) -> None:
    _export(
        controller,
        _jax_transformer_graph(),
        ExecutionBackend.JAX,
        RuntimeApi.KERAS,
        ExportFormat.ONNX,
        tmp_path / "keras_jax_positional.onnx",
    )


###############################################################################
def test_native_jax_onnx_export_uses_direct_jax2onnx_bridge(
    controller: RuntimeController,
    tmp_path: Path,
) -> None:
    result = _export(
        controller,
        _native_jax_multi_input_graph(),
        ExecutionBackend.JAX,
        RuntimeApi.NATIVE,
        ExportFormat.ONNX,
        tmp_path / "native_jax_direct.onnx",
    )
    assert result.artifact is not None
    assert result.artifact.conversion is ExportConversion.NATIVE


###############################################################################
def _missing_executable(_name: str) -> None:
    return None


###############################################################################
def test_pytorch_compile_capability_requires_msvc_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    import models.pytorch_worker as pytorch_worker_module

    monkeypatch.setattr(pytorch_worker_module.sys, "platform", "win32")
    monkeypatch.setattr(pytorch_worker_module.shutil, "which", _missing_executable)

    assert pytorch_worker_module.PyTorchWorker().native_compilation_capabilities() == {
        "torch_compile": False
    }
