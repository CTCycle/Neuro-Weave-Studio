from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pytest

from common.catalogs.loader import get_node_catalog
from compilation.compiler import compile_to_ir
from configurations.settings import get_settings
from domain.enums import ExecutionBackend, keras_backend_value
from domain.exports import ExportFormat
from domain.graph import (
    EdgeInstance,
    KerasCompilationConfiguration,
    ModelConfiguration,
    ModelGraph,
    NodeInstance,
    Position,
)
from domain.jobs import (
    JobResponse,
    JobSubmitRequest,
    ModelExportOperation,
    SmokeInferenceOperation,
)
from domain.runtime import WorkerConfiguration
from exporters.keras_exporter import KerasSourceExporter
from services.runtime_controller import RuntimeController
from services.validation_service import ValidationService

###############################################################################
def node(
    node_id: str,
    definition_id: str,
    parameters: dict[str, Any] | None = None,
    *,
    label: str | None = None,
) -> NodeInstance:
    return NodeInstance(
        id=node_id,
        definition_id=definition_id,
        label=label,
        position=Position(x=0, y=0),
        parameters=parameters or {},
    )

###############################################################################
def edge(
    edge_id: str,
    source: str,
    target: str,
    target_port: str = "input",
) -> EdgeInstance:
    return EdgeInstance(
        id=edge_id,
        source_node_id=source,
        source_port_id="output",
        target_node_id=target,
        target_port_id=target_port,
    )

###############################################################################
def validate(graph: ModelGraph):
    return ValidationService(get_node_catalog()).validate(
        graph,
        ExecutionBackend.PYTORCH,
        True,
    )

###############################################################################
def runtime_controller() -> RuntimeController:
    controller = RuntimeController(get_settings())
    controller.probe_statuses()
    return controller

###############################################################################
def runtime_worker(
    controller: RuntimeController,
    backend: ExecutionBackend,
    *,
    precision: Literal[
        "float32",
        "mixed_float16",
        "mixed_bfloat16",
    ] = "float32",
) -> WorkerConfiguration | None:
    status = controller.backend_statuses.get(backend)
    if status is None or not status.available or not status.devices:
        return None
    device = next((item for item in status.devices if item.available), None)
    if device is None:
        return None
    return WorkerConfiguration(
        backend=backend,
        device_id=device.device_id,
        precision=precision,
    )

###############################################################################
def available_backend() -> ExecutionBackend | None:
    controller = runtime_controller()
    try:
        return next(
            (
                backend
                for backend, status in controller.backend_statuses.items()
                if status.available
            ),
            None,
        )
    finally:
        controller.shutdown()

###############################################################################
def run_smoke(
    graph: ModelGraph,
    backend: ExecutionBackend = ExecutionBackend.PYTORCH,
) -> dict[str, Any] | None:
    controller = runtime_controller()
    try:
        worker = runtime_worker(controller, backend)
        if worker is None:
            return None
        job = controller.submit(
            JobSubmitRequest(
                operation=SmokeInferenceOperation(
                    project_id=uuid4(),
                    graph=graph,
                ),
                worker=worker,
            ),
            graph=graph,
        )
        result = wait_for_runtime_job(controller, job.id)
        assert result.state.value == "succeeded", result
        return result.result or {}
    finally:
        controller.shutdown()

###############################################################################
def export_job(
    graph: ModelGraph,
    destination: Path,
    backend: ExecutionBackend = ExecutionBackend.PYTORCH,
) -> Path | None:
    controller = runtime_controller()
    try:
        worker = runtime_worker(controller, backend)
        if worker is None:
            return None
        job = controller.submit(
            JobSubmitRequest(
                operation=ModelExportOperation(
                    project_id=uuid4(),
                    format=ExportFormat.KERAS,
                ),
                worker=worker,
            ),
            graph=graph,
            artifact_destination=destination,
            artifact_filename=destination.name,
        )
        result = wait_for_runtime_job(controller, job.id)
        assert result.state.value == "succeeded", result
        assert result.artifact is not None
        return destination
    finally:
        controller.shutdown()

###############################################################################
def wait_for_runtime_job(
    controller: RuntimeController,
    job_id: str,
) -> JobResponse:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        result = controller.get(job_id)
        if result.state.value in {"succeeded", "terminated", "failed"}:
            return result
        time.sleep(0.05)
    controller.stop(job_id)
    raise TimeoutError(f"ML worker job {job_id} timed out")

###############################################################################
def test_attention_shape_and_parameter_contract_matches_runtime() -> None:
    graph = ModelGraph(
        nodes=[
            node(
                "query",
                "input.sequence",
                {"shape": [5, 10], "dtype": "float32"},
            ),
            node(
                "context",
                "input.sequence",
                {"shape": [7, 12], "dtype": "float32"},
            ),
            node(
                "attention",
                "attention.multi_head",
                {"num_heads": 3, "key_dim": 4, "output_shape": [6]},
            ),
            node("output", "output.sequence"),
        ],
        edges=[
            edge("context-first", "context", "attention", "context"),
            edge("query-second", "query", "attention", "query"),
            edge("e3", "attention", "output"),
        ],
    )

    result = validate(graph)
    assert result.valid
    attention = next(item for item in result.shape_summaries if item.node_id == "attention")
    assert attention.outputs["output"].shape == [None, 5, 6]
    assert attention.parameter_count == 522

    runtime = run_smoke(graph)
    if runtime is None:
        pytest.skip("Operational PyTorch Keras backend is unavailable")
    assert runtime["outputs"] == [
        {
            "node_id": "output",
            "shape": [1, 5, 6],
            "dtype": "float32",
        }
    ]
    assert runtime["parameter_count"] == 522

###############################################################################
def test_generated_keras_source_executes_and_compiles(tmp_path: Path) -> None:
    backend = available_backend()
    if backend is None:
        pytest.skip("No operational Keras backend is installed")

    graph = ModelGraph(
        nodes=[
            node(
                "input",
                "input.tabular",
                {"shape": [4], "dtype": "float32"},
            ),
            node(
                "dense",
                "layer.dense",
                {"units": 2, "activation": "softmax"},
            ),
            node("output", "output.classification"),
        ],
        edges=[
            edge("e1", "input", "dense"),
            edge("e2", "dense", "output"),
        ],
        model_configuration=ModelConfiguration(
            keras=KerasCompilationConfiguration(
                losses={"__default__": "categorical_crossentropy"},
                metrics={"__default__": ["categorical_accuracy"]},
            ),
        ),
    )
    assert validate(graph).valid

    source = KerasSourceExporter().generate(
        graph,
        compile_to_ir(
            graph,
            ExecutionBackend.PYTORCH,
            True,
            get_node_catalog(),
        ),
    )
    assert "loss=keras.losses.CategoricalCrossentropy()" in source
    source_path = tmp_path / "generated_model.py"
    source_path.write_text(source, encoding="utf-8")

    environment = os.environ.copy()
    environment["KERAS_BACKEND"] = keras_backend_value(backend)
    command = [
        sys.executable,
        "-c",
        (
            "import importlib.util; "
            f"spec=importlib.util.spec_from_file_location('generated_model', r'{source_path}'); "
            "module=importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            "model=module.build_model(); "
            "assert model.output_shape == (None, 2); "
            "assert model.optimizer is not None"
        ),
    ]
    completed = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr

###############################################################################
def test_generated_keras_source_preserves_recurrent_state_ports(tmp_path: Path) -> None:
    backend = available_backend()
    if backend is None:
        pytest.skip("No operational Keras backend is installed")

    graph = ModelGraph(
        nodes=[
            node(
                "input",
                "input.sequence",
                {"shape": [5, 3], "dtype": "float32"},
            ),
            node(
                "lstm",
                "recurrent.lstm",
                {"units": 4, "return_state": True},
            ),
        ],
        edges=[edge("e1", "input", "lstm")],
    )
    result = validate(graph)
    assert result.valid
    assert {
        (item.node_id, item.port_id)
        for item in result.effective_outputs
    } == {("lstm", "output"), ("lstm", "hidden_state"), ("lstm", "cell_state")}

    source = KerasSourceExporter().generate(
        graph,
        compile_to_ir(graph, ExecutionBackend.PYTORCH, True, get_node_catalog()),
    )
    source_path = tmp_path / "generated_recurrent.py"
    source_path.write_text(source, encoding="utf-8")
    environment = os.environ.copy()
    environment["KERAS_BACKEND"] = keras_backend_value(backend)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util; "
                f"spec=importlib.util.spec_from_file_location('generated_recurrent', r'{source_path}'); "
                "module=importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(module); "
                "model=module.build_model(); "
                "assert model.output_shape == [(None, 4), (None, 4), (None, 4)]"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

###############################################################################
def test_multi_output_compile_selectors_generate_executable_transformer_source(
    tmp_path: Path,
) -> None:
    backend = available_backend()
    if backend is None:
        pytest.skip("No operational Keras backend is installed")

    graph = ModelGraph(
        nodes=[
            node(
                "input",
                "input.sequence",
                {"shape": [5, 128], "dtype": "float32"},
            ),
            node(
                "encoder",
                "transformer.encoder_block",
                {
                    "num_heads": 8,
                    "key_dim": 16,
                    "ffn_multiplier": 2,
                    "layers": 2,
                },
            ),
            node("output_a", "output.sequence", label="tokens"),
            node("output_b", "output.sequence", label="features"),
        ],
        edges=[
            edge("e1", "input", "encoder", "query"),
            edge("e2", "encoder", "output_a"),
            edge("e3", "encoder", "output_b"),
        ],
        model_configuration=ModelConfiguration(
            keras=KerasCompilationConfiguration(
                optimizer="adamw",
                losses={
                    "tokens": "mean_squared_error",
                    "output_b": "mean_absolute_error",
                },
                metrics={"__default__": ["mean_squared_error"]},
            ),
        ),
    )
    assert validate(graph).valid

    source = KerasSourceExporter().generate(
        graph,
        compile_to_ir(
            graph,
            ExecutionBackend.PYTORCH,
            True,
            get_node_catalog(),
        ),
    )
    source_path = tmp_path / "generated_transformer.py"
    source_path.write_text(source, encoding="utf-8")

    environment = os.environ.copy()
    environment["KERAS_BACKEND"] = keras_backend_value(backend)
    command = [
        sys.executable,
        "-c",
        (
            "import importlib.util; "
            f"spec=importlib.util.spec_from_file_location('generated_transformer', r'{source_path}'); "
            "module=importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            "model=module.build_model(); "
            "assert len(model.outputs) == 2; "
            "assert model.optimizer.__class__.__name__ == 'AdamW'"
        ),
    ]
    completed = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr

###############################################################################
def test_keras_export_reloads_custom_layer_in_clean_process(
    tmp_path: Path,
) -> None:
    graph = ModelGraph(
        nodes=[
            node(
                "input",
                "input.sequence",
                {"shape": [5, 8], "dtype": "float32"},
            ),
            node(
                "position",
                "transformer.positional_encoding",
                {"mode": "sinusoidal"},
            ),
            node("output", "output.sequence"),
        ],
        edges=[
            edge("e1", "input", "position"),
            edge("e2", "position", "output"),
        ],
        model_configuration=ModelConfiguration(
            keras=KerasCompilationConfiguration(
                optimizer="adamw",
                losses={"__default__": "mean_squared_error"},
                metrics={"__default__": ["mean_squared_error"]},
                random_seed=123,
            ),
        ),
    )
    assert validate(graph).valid

    destination = export_job(graph, tmp_path / "positional.keras")
    if destination is None:
        pytest.skip("Operational PyTorch Keras backend is unavailable")

    environment = os.environ.copy()
    environment["KERAS_BACKEND"] = "torch"
    server_root = Path(__file__).resolve().parents[1] / "server"
    command = [
        sys.executable,
        "-c",
        (
            "import keras; "
            "from models.keras_layers import SinusoidalPositionalEncoding; "
            f"model=keras.saving.load_model(r'{destination}', compile=True); "
            "output=model(keras.ops.zeros((1, 5, 8)), training=False); "
            "assert tuple(output.shape) == (1, 5, 8); "
            "assert model.optimizer.__class__.__name__ == 'AdamW'"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=server_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
