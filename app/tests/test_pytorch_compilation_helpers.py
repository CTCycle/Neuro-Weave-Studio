from __future__ import annotations

from contextlib import nullcontext
from itertools import pairwise
from typing import Any

import pytest
from common.catalogs.loader import get_node_catalog
from domain.compilation_catalog import get_compilation_catalog
from domain.enums import ExecutionBackend
from domain.graph import (
    EdgeInstance,
    KerasCompilationConfiguration,
    ModelConfiguration,
    ModelGraph,
    NativeCompilationConfiguration,
    NodeInstance,
    Position,
    PyTorchNativeConfiguration,
)
from models.worker_backend import run_keras_smoke, run_native_smoke
from services.validation_service import ValidationService
from validation.compile_configuration import validate_model_configuration


###############################################################################
def _node(
    node_id: str, definition_id: str, parameters: dict[str, Any] | None = None
) -> NodeInstance:
    return NodeInstance(
        id=node_id,
        definition_id=definition_id,
        position=Position(x=0, y=0),
        parameters=parameters or {},
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
def _workflow_graphs() -> dict[str, tuple[ModelGraph, tuple[int, ...], str]]:
    def graph(
        nodes: list[NodeInstance], links: list[str], expected: tuple[int, ...], output: str
    ) -> tuple[ModelGraph, tuple[int, ...], str]:
        return (
            ModelGraph(
                nodes=nodes,
                edges=[
                    _edge(f"edge-{index}", source, target)
                    for index, (source, target) in enumerate(pairwise(links), start=1)
                ],
            ),
            expected,
            output,
        )

    return {
        "dense_regression": graph(
            [
                _node("input", "input.tabular", {"shape": [None, 8], "dtype": "float32"}),
                _node("hidden", "layer.dense", {"units": 16, "activation": "relu"}),
                _node("head", "layer.dense", {"units": 1, "activation": "linear"}),
                _node("output", "output.regression"),
            ],
            ["input", "hidden", "head", "output"],
            (1, 1),
            "regression",
        ),
        "binary_classification": graph(
            [
                _node("input", "input.tabular", {"shape": [None, 8], "dtype": "float32"}),
                _node("head", "layer.dense", {"units": 1, "activation": "sigmoid"}),
                _node("output", "output.classification"),
            ],
            ["input", "head", "output"],
            (1, 1),
            "binary",
        ),
        "multiclass_classification": graph(
            [
                _node("input", "input.tabular", {"shape": [None, 8], "dtype": "float32"}),
                _node("head", "layer.dense", {"units": 3, "activation": "softmax"}),
                _node("output", "output.classification"),
            ],
            ["input", "head", "output"],
            (1, 3),
            "multiclass",
        ),
        "recurrent": graph(
            [
                _node("input", "input.sequence", {"shape": [None, 6, 4], "dtype": "float32"}),
                _node("recurrent", "recurrent.gru", {"units": 4}),
                _node("head", "layer.dense", {"units": 1, "activation": "sigmoid"}),
                _node("output", "output.classification"),
            ],
            ["input", "recurrent", "head", "output"],
            (1, 1),
            "binary",
        ),
        "lstm": graph(
            [
                _node("input", "input.sequence", {"shape": [None, 6, 4], "dtype": "float32"}),
                _node("lstm", "recurrent.lstm", {"units": 4, "return_sequences": True}),
                _node("flatten", "layer.flatten"),
                _node("head", "layer.dense", {"units": 3, "activation": "softmax"}),
                _node("output", "output.classification"),
            ],
            ["input", "lstm", "flatten", "head", "output"],
            (1, 3),
            "multiclass",
        ),
        "cnn": graph(
            [
                _node("input", "input.image", {"shape": [None, 16, 16, 1], "dtype": "float32"}),
                _node(
                    "conv",
                    "conv.conv2d",
                    {
                        "filters": 4,
                        "kernel_size": [3, 3],
                        "padding": "same",
                        "activation": "linear",
                    },
                ),
                _node("activation", "layer.activation", {"activation": "relu"}),
                _node(
                    "pool",
                    "conv.max_pool2d",
                    {"pool_size": [2, 2], "strides": [2, 2], "padding": "valid"},
                ),
                _node("flatten", "layer.flatten"),
                _node("head", "layer.dense", {"units": 3, "activation": "softmax"}),
                _node("output", "output.classification"),
            ],
            ["input", "conv", "activation", "pool", "flatten", "head", "output"],
            (1, 3),
            "multiclass",
        ),
    }


###############################################################################
def _configured_graph(graph: ModelGraph, api: str, task: str) -> ModelGraph:
    if task == "regression":
        loss = "mean_squared_error"
        metrics = ["mean_absolute_error", "mean_squared_error", "root_mean_squared_error"]
    elif task == "binary":
        loss = "binary_crossentropy"
        metrics = ["binary_accuracy", "precision", "recall", "auc"]
    else:
        loss = "categorical_crossentropy"
        metrics = [
            "accuracy",
            "categorical_accuracy",
            "sparse_categorical_accuracy",
            "precision",
            "recall",
            "auc",
        ]
    if api == "native":
        configuration = ModelConfiguration(
            native=NativeCompilationConfiguration(
                pytorch=PyTorchNativeConfiguration(
                    optimizer="adamw",
                    optimizer_parameters={"learning_rate": 0.001, "weight_decay": 0.01},
                    losses={"__default__": loss},
                    metrics={"__default__": metrics},
                )
            )
        )
    else:
        configuration = ModelConfiguration(
            keras=KerasCompilationConfiguration(
                losses={"__default__": loss},
                metrics={"__default__": metrics},
            )
        )
    return graph.model_copy(update={"model_configuration": configuration})


###############################################################################
@pytest.mark.parametrize("workflow", list(_workflow_graphs()))
def test_native_pytorch_workflows_verify_forward_and_compilation_helpers(workflow: str) -> None:
    graph, expected_shape, task = _workflow_graphs()[workflow]
    graph = _configured_graph(graph, "native", task)
    result = run_native_smoke(
        graph,
        backend=ExecutionBackend.PYTORCH,
        capabilities={"accelerator": "cpu", "device_name": "CPU"},
        scope=nullcontext(),
        synchronize=lambda values: None,
        checkpoint=lambda: None,
    )

    assert result["status"] == "succeeded"
    assert result["compilation"]["verified"] is True
    assert result["compilation"]["optimizer"] == {"id": "adamw", "instantiated": True}
    assert [tuple(item["shape"]) for item in result["outputs"]] == [expected_shape]


###############################################################################
@pytest.mark.parametrize("workflow", list(_workflow_graphs()))
def test_keras_pytorch_backend_workflows_compile_and_forward(workflow: str) -> None:
    pytest.importorskip("keras")
    import keras
    from models.keras_runtime import compile_model
    from models.keras_worker import build_model

    graph, expected_shape, task = _workflow_graphs()[workflow]
    graph = _configured_graph(graph, "keras", task)
    result = run_keras_smoke(
        graph,
        keras_module=keras,
        build_model=build_model,
        compile_model=compile_model,
        capabilities={"accelerator": "cpu", "device_name": "CPU"},
        scope=nullcontext(),
        synchronize=lambda values: None,
        checkpoint=lambda: None,
    )

    assert result["status"] == "succeeded"
    assert result["compilation"]["verified"] is True
    assert [tuple(item["shape"]) for item in result["outputs"]] == [expected_shape]


###############################################################################
def test_catalog_contains_complete_keras_and_native_pytorch_mappings() -> None:
    catalog = get_compilation_catalog()
    assert all(
        option.implementations["keras"].supported
        for option in catalog.optimizers + catalog.losses + catalog.metrics
    )
    assert all(
        option.implementations["pytorch"].supported
        for option in catalog.optimizers + catalog.losses + catalog.metrics
    )
    assert all(option.formula for option in catalog.losses + catalog.metrics)
    assert {option.id for option in catalog.losses} == {
        option.id for option in get_compilation_catalog().losses
    }


###############################################################################
def test_native_configuration_rejects_incompatible_loss_metric_and_activation() -> None:
    graph, _, _ = _workflow_graphs()["binary_classification"]
    graph = graph.model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                    pytorch=PyTorchNativeConfiguration(
                        losses={"__default__": "categorical_crossentropy"},
                        metrics={"__default__": ["mean_squared_error"]},
                    )
                )
            )
        }
    )
    diagnostics = validate_model_configuration(graph, ExecutionBackend.PYTORCH, False)
    codes = {item.code for item in diagnostics}
    assert "incompatible_compile_loss" in codes
    assert "incompatible_compile_metric" in codes

    graph = graph.model_copy(
        update={
            "nodes": [
                node.model_copy(update={"parameters": {**node.parameters, "activation": "softmax"}})
                if node.id == "head"
                else node
                for node in graph.nodes
            ]
        }
    )
    assert any(
        item.code == "invalid_output_activation"
        for item in validate_model_configuration(graph, ExecutionBackend.PYTORCH, False)
    )


###############################################################################
def test_native_configured_graph_passes_full_static_validation() -> None:
    graph, _, task = _workflow_graphs()["multiclass_classification"]
    graph = _configured_graph(graph, "native", task)
    result = ValidationService(get_node_catalog()).validate(graph, ExecutionBackend.PYTORCH, False)
    assert result.valid
    assert result.shape_summaries


###############################################################################
@pytest.mark.parametrize(
    ("optimizer", "parameters"),
    [
        ("adam", {"learning_rate": 0.001}),
        ("adamw", {"learning_rate": 0.001, "weight_decay": 0.01}),
        ("sgd", {"learning_rate": 0.01, "momentum": 0.1, "nesterov": False}),
        ("rmsprop", {"learning_rate": 0.001, "rho": 0.9, "momentum": 0.0, "centered": False}),
    ],
)
def test_native_optimizer_helpers_cover_catalog_variants(
    optimizer: str, parameters: dict[str, Any]
) -> None:
    graph, _, _ = _workflow_graphs()["dense_regression"]
    graph = graph.model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                    pytorch=PyTorchNativeConfiguration(
                        optimizer=optimizer,
                        optimizer_parameters=parameters,
                        losses={"__default__": "mean_squared_error"},
                    )
                )
            )
        }
    )
    result = run_native_smoke(
        graph,
        backend=ExecutionBackend.PYTORCH,
        capabilities={"accelerator": "cpu", "device_name": "CPU"},
        scope=nullcontext(),
        synchronize=lambda values: None,
        checkpoint=lambda: None,
    )
    assert result["compilation"]["optimizer"] == {"id": optimizer, "instantiated": True}
    assert result["compilation"]["losses"]["output"]["id"] == "mean_squared_error"


###############################################################################
@pytest.mark.parametrize(
    ("workflow", "loss_id"),
    [
        ("dense_regression", "mean_absolute_error"),
        ("dense_regression", "huber"),
        ("multiclass_classification", "sparse_categorical_crossentropy"),
        ("multiclass_classification", "kullback_leibler_divergence"),
    ],
)
def test_native_loss_helpers_cover_remaining_catalog_variants(workflow: str, loss_id: str) -> None:
    graph, _, _ = _workflow_graphs()[workflow]
    graph = graph.model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                    pytorch=PyTorchNativeConfiguration(losses={"__default__": loss_id})
                )
            )
        }
    )
    result = run_native_smoke(
        graph,
        backend=ExecutionBackend.PYTORCH,
        capabilities={"accelerator": "cpu", "device_name": "CPU"},
        scope=nullcontext(),
        synchronize=lambda values: None,
        checkpoint=lambda: None,
    )
    assert result["compilation"]["losses"]["output"]["id"] == loss_id
