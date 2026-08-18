from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from common.catalogs.loader import get_node_catalog
from domain.enums import CodeTarget, ExecutionBackend, keras_backend_value
from domain.graph import EdgeInstance, ModelConfiguration, ModelGraph, NodeInstance, Position
from exporters.keras_exporter import KerasSourceExporter
from models.keras_backend_runner import KerasBackendRunner
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
    return ValidationService(get_node_catalog()).validate(graph, CodeTarget.KERAS)


###############################################################################
def test_pool_defaults_and_grouped_convolution_follow_keras_contract() -> None:
    pool_graph = ModelGraph(
        nodes=[
            node("image", "input.image", {"shape": [None, 32, 32, 3], "dtype": "float32"}),
            node("pool", "conv.max_pool2d", {"pool_size": [2, 2], "strides": None}),
            node("output", "output.custom"),
        ],
        edges=[edge("e1", "image", "pool"), edge("e2", "pool", "output")],
    )
    pool_result = validate(pool_graph)
    assert pool_result.valid
    assert next(item for item in pool_result.shape_summaries if item.node_id == "pool").outputs[
        "output"
    ].shape == [None, 16, 16, 3]

    grouped_graph = ModelGraph(
        nodes=[
            node("image", "input.image", {"shape": [None, 8, 8, 3], "dtype": "float32"}),
            node("conv", "conv.conv2d", {"filters": 4, "groups": 2, "padding": "same"}),
            node("output", "output.custom"),
        ],
        edges=[edge("e1", "image", "conv"), edge("e2", "conv", "output")],
    )
    grouped_result = validate(grouped_graph)
    assert not grouped_result.valid
    assert any("divisible by groups" in item.message for item in grouped_result.diagnostics)

    valid_grouped_graph = ModelGraph(
        nodes=[
            node("image", "input.image", {"shape": [None, 8, 8, 4], "dtype": "float32"}),
            node("conv", "conv.conv2d", {"filters": 8, "groups": 2, "padding": "same"}),
            node("output", "output.custom"),
        ],
        edges=[edge("e1", "image", "conv"), edge("e2", "conv", "output")],
    )
    valid_grouped_result = validate(valid_grouped_graph)
    assert valid_grouped_result.valid
    assert next(
        item for item in valid_grouped_result.shape_summaries if item.node_id == "conv"
    ).parameter_count == 152

    runner = KerasBackendRunner()
    if runner.probe(ExecutionBackend.PYTORCH):
        runtime = runner.smoke_test(pool_graph, ExecutionBackend.PYTORCH)
        assert runtime["outputs"] == [
            {"node_id": "output", "shape": [1, 16, 16, 3], "dtype": "float32"}
        ]


###############################################################################
def test_validation_rejects_invalid_shape_and_dtype_contracts() -> None:
    invalid_axis_graph = ModelGraph(
        nodes=[
            node("left", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            node("right", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            node("merge", "merge.concatenate", {"axis": 2}),
            node("output", "output.custom"),
        ],
        edges=[
            edge("e1", "left", "merge"),
            edge("e2", "right", "merge"),
            edge("e3", "merge", "output"),
        ],
    )
    assert not validate(invalid_axis_graph).valid

    invalid_batch_graph = ModelGraph(
        nodes=[
            node("input", "input.tabular", {"shape": [4], "dtype": "float32"}),
            node("output", "output.custom"),
        ],
        edges=[edge("e1", "input", "output")],
    )
    assert not validate(invalid_batch_graph).valid

    invalid_embedding_graph = ModelGraph(
        nodes=[
            node("input", "input.sequence", {"shape": [None, 3], "dtype": "float32"}),
            node("embedding", "layer.embedding"),
            node("output", "output.sequence"),
        ],
        edges=[edge("e1", "input", "embedding"), edge("e2", "embedding", "output")],
    )
    assert not validate(invalid_embedding_graph).valid

    learned_position_graph = ModelGraph(
        nodes=[
            node("input", "input.sequence", {"shape": [None, 3, 8], "dtype": "float32"}),
            node("position", "transformer.positional_encoding", {"mode": "learned"}),
            node("output", "output.sequence"),
        ],
        edges=[edge("e1", "input", "position"), edge("e2", "position", "output")],
    )
    assert not validate(learned_position_graph).valid


###############################################################################
def test_attention_shape_and_port_order_match_runtime() -> None:
    graph = ModelGraph(
        nodes=[
            node("query", "input.sequence", {"shape": [None, 5, 10], "dtype": "float32"}),
            node("context", "input.sequence", {"shape": [None, 7, 12], "dtype": "float32"}),
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

    runner = KerasBackendRunner()
    if not runner.probe(ExecutionBackend.PYTORCH):
        pytest.skip("CPU PyTorch backend is unavailable")
    runtime = runner.smoke_test(graph, ExecutionBackend.PYTORCH)
    assert runtime["outputs"] == [{"node_id": "output", "shape": [1, 5, 6], "dtype": "float32"}]
    assert runtime["parameter_count"] == 522


###############################################################################
def test_transformer_encoder_preserves_residual_width() -> None:
    graph = ModelGraph(
        nodes=[
            node("input", "input.sequence", {"shape": [None, 5, 128], "dtype": "float32"}),
            node(
                "encoder",
                "transformer.encoder_block",
                {"num_heads": 8, "key_dim": 64, "ffn_multiplier": 4, "layers": 2},
            ),
            node("output", "output.sequence"),
        ],
        edges=[edge("e1", "input", "encoder", "query"), edge("e2", "encoder", "output")],
    )
    result = validate(graph)
    assert result.valid
    assert next(item for item in result.shape_summaries if item.node_id == "encoder").outputs[
        "output"
    ].shape == [None, 5, 128]

    runner = KerasBackendRunner()
    if not runner.probe(ExecutionBackend.PYTORCH):
        pytest.skip("CPU PyTorch backend is unavailable")
    runtime = runner.smoke_test(graph, ExecutionBackend.PYTORCH)
    assert runtime["outputs"] == [{"node_id": "output", "shape": [1, 5, 128], "dtype": "float32"}]


###############################################################################
def test_invalid_state_and_unreachable_components_are_blocked() -> None:
    state_graph = ModelGraph(
        nodes=[
            node("input", "input.sequence", {"shape": [None, 3, 4], "dtype": "float32"}),
            node("lstm", "recurrent.lstm", {"units": 2, "return_state": True}),
            node("output", "output.sequence"),
        ],
        edges=[edge("e1", "input", "lstm"), edge("e2", "lstm", "output")],
    )
    state_result = validate(state_graph)
    assert not state_result.valid
    assert any("return_state" in item.message for item in state_result.diagnostics)

    unreachable_graph = ModelGraph(
        nodes=[
            node("used", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            node("unused", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            node("dense", "layer.dense", {"units": 2}),
            node("output", "output.custom"),
        ],
        edges=[edge("e1", "used", "dense"), edge("e2", "dense", "output")],
    )
    unreachable_result = validate(unreachable_graph)
    assert not unreachable_result.valid
    assert any(item.code == "unreachable_model_input" for item in unreachable_result.diagnostics)


###############################################################################
def test_generated_keras_source_executes_and_compiles(tmp_path: Path) -> None:
    runner = KerasBackendRunner()
    backend = next(iter(runner.available_backends()), None)
    if backend is None:
        pytest.skip("No operational Keras backend is installed")
    graph = ModelGraph(
        nodes=[
            node("input", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            node("dense", "layer.dense", {"units": 2, "activation": "softmax"}),
            node("output", "output.classification"),
        ],
        edges=[edge("e1", "input", "dense"), edge("e2", "dense", "output")],
        model_configuration=ModelConfiguration(
            losses={"__default__": "mean_squared_error"},
            metrics={"__default__": ["mse"]},
        ),
    )
    result = validate(graph)
    assert result.valid
    from compilation.compiler import compile_to_ir

    source = KerasSourceExporter().generate(
        graph,
        compile_to_ir(graph, CodeTarget.KERAS, get_node_catalog()),
    )
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
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "model=module.build_model(); assert model.output_shape == (None, 2); "
            "assert model.optimizer is not None"
        ),
    ]
    completed = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


###############################################################################
@pytest.mark.parametrize("optimizer", ["adam", "adamw", "sgd", "rmsprop"])
def test_supported_optimizers_compile_on_cpu_backend(optimizer: str) -> None:
    graph = ModelGraph(
        nodes=[
            node("input", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            node("dense", "layer.dense", {"units": 2}),
            node("output", "output.regression"),
        ],
        edges=[edge("e1", "input", "dense"), edge("e2", "dense", "output")],
        model_configuration=ModelConfiguration(
            optimizer=optimizer,
            optimizer_parameters={"learning_rate": 0.002},
            losses={"__default__": "mean_squared_error"},
            metrics={"__default__": ["mae"]},
            random_seed=7,
        ),
    )
    result = validate(graph)
    assert result.valid
    runner = KerasBackendRunner()
    if not runner.probe(ExecutionBackend.PYTORCH):
        pytest.skip("CPU PyTorch backend is unavailable")
    runtime = runner.smoke_test(graph, ExecutionBackend.PYTORCH)
    assert runtime["outputs"] == [
        {"node_id": "output", "shape": [1, 2], "dtype": "float32"}
    ]


###############################################################################
def test_mixed_precision_policy_is_applied_before_cpu_model_build() -> None:
    graph = ModelGraph(
        nodes=[
            node("input", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            node("dense", "layer.dense", {"units": 2}),
            node("output", "output.regression"),
        ],
        edges=[edge("e1", "input", "dense"), edge("e2", "dense", "output")],
        model_configuration=ModelConfiguration(
            losses={"__default__": "mean_squared_error"},
            mixed_precision=True,
            random_seed=11,
        ),
    )
    assert validate(graph).valid
    runner = KerasBackendRunner()
    if not runner.probe(ExecutionBackend.PYTORCH):
        pytest.skip("CPU PyTorch backend is unavailable")
    runtime = runner.smoke_test(graph, ExecutionBackend.PYTORCH)
    assert runtime["outputs"] == [
        {"node_id": "output", "shape": [1, 2], "dtype": "float16"}
    ]


###############################################################################
def test_multi_output_compile_selectors_and_transformer_source(tmp_path: Path) -> None:
    runner = KerasBackendRunner()
    backend = next(iter(runner.available_backends()), None)
    if backend is None:
        pytest.skip("No operational Keras backend is installed")
    graph = ModelGraph(
        nodes=[
            node("input", "input.sequence", {"shape": [None, 5, 128], "dtype": "float32"}),
            node(
                "encoder",
                "transformer.encoder_block",
                {"num_heads": 8, "key_dim": 16, "ffn_multiplier": 2, "layers": 2},
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
            optimizer="adamw",
            losses={"tokens": "mean_squared_error", "output_b": "mean_absolute_error"},
            metrics={"__default__": ["mse"]},
        ),
    )
    result = validate(graph)
    assert result.valid
    from compilation.compiler import compile_to_ir

    source = KerasSourceExporter().generate(
        graph,
        compile_to_ir(graph, CodeTarget.KERAS, get_node_catalog()),
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
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "model=module.build_model(); assert len(model.outputs) == 2; "
            "assert model.optimizer.__class__.__name__ == 'AdamW'"
        ),
    ]
    completed = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr

    runner = KerasBackendRunner()
    if runner.probe(ExecutionBackend.PYTORCH):
        runtime = runner.smoke_test(graph, ExecutionBackend.PYTORCH)
        assert runtime["outputs"] == [
            {"node_id": "output_a", "shape": [1, 5, 128], "dtype": "float32"},
            {"node_id": "output_b", "shape": [1, 5, 128], "dtype": "float32"},
        ]


###############################################################################
def test_binary_artifact_reloads_in_a_clean_process(tmp_path: Path) -> None:
    graph = ModelGraph(
        nodes=[
            node("input", "input.sequence", {"shape": [None, 5, 8], "dtype": "float32"}),
            node("position", "transformer.positional_encoding", {"mode": "sinusoidal"}),
            node("output", "output.sequence"),
        ],
        edges=[edge("e1", "input", "position"), edge("e2", "position", "output")],
        model_configuration=ModelConfiguration(
            optimizer="adamw",
            losses={"__default__": "mean_squared_error"},
            metrics={"__default__": ["mse"]},
            random_seed=123,
        ),
    )
    assert validate(graph).valid
    runner = KerasBackendRunner()
    if not runner.probe(ExecutionBackend.PYTORCH):
        pytest.skip("CPU PyTorch backend is unavailable")
    destination = runner.export(graph, tmp_path / "positional.keras", ExecutionBackend.PYTORCH)
    environment = os.environ.copy()
    environment["KERAS_BACKEND"] = "torch"
    server_root = Path(__file__).resolve().parents[1] / "server"
    command = [
        sys.executable,
        "-c",
        (
            "import keras; from models.keras_layers import SinusoidalPositionalEncoding; "
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
