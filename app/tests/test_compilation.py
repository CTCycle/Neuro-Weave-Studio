# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from itertools import pairwise
from math import isfinite
from typing import Any, cast

import pytest

from common.catalogs.loader import get_node_catalog
from compilation.compiler import compile_to_ir
from domain.compilation_capabilities import get_compilation_capabilities
from domain.compilation_catalog import get_compilation_catalog
from domain.compilation_plan import CompilationObject, CompilationPlan, CompilerObject
from domain.enums import ExecutionBackend
from domain.graph import (
    EdgeInstance,
    JaxNativeConfiguration,
    KerasProfileConfiguration,
    ModelConfiguration,
    ModelGraph,
    NativeCompilationConfiguration,
    NodeInstance,
    Position,
    PyTorchNativeConfiguration,
    TensorFlowNativeConfiguration,
)
from domain.metadata import ExecutionBackendStatus
from models.native_runtime import build_native_model
from models.worker_backend import (
    _verify_native_compilation,  # pyright: ignore[reportPrivateUsage]
    run_keras_smoke,
    run_native_smoke,
)
from services.compilation_service import CompilationService
from services.validation_service import ValidationService
from validation.compile_configuration import validate_model_configuration

# Consolidated from test_compilation_capabilities.py

###############################################################################
def _tensorflow_native(**payload: object) -> TensorFlowNativeConfiguration:
    return TensorFlowNativeConfiguration.model_validate(payload)


###############################################################################
def _pytorch_native(**payload: object) -> PyTorchNativeConfiguration:
    return PyTorchNativeConfiguration.model_validate(payload)


###############################################################################
def _jax_native(**payload: object) -> JaxNativeConfiguration:
    return JaxNativeConfiguration.model_validate(payload)


###############################################################################
def _approx(expected: object) -> object:
    approx = cast(Callable[..., object], pytest.approx)  # pyright: ignore[reportUnknownMemberType]
    return approx(expected)

###############################################################################
def _native_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            NodeInstance(
                id="input",
                definition_id="input.tabular",
                position=Position(x=0, y=0),
                parameters={"shape": [4], "dtype": "float32"},
            ),
            NodeInstance(
                id="dense",
                definition_id="layer.dense",
                position=Position(x=1, y=0),
                parameters={"units": 2, "activation": "softmax"},
            ),
            NodeInstance(
                id="output",
                definition_id="output.classification",
                position=Position(x=2, y=0),
            ),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="input",
                source_port_id="output",
                target_node_id="dense",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e2",
                source_node_id="dense",
                source_port_id="output",
                target_node_id="output",
                target_port_id="input",
            ),
        ],
        model_configuration=ModelConfiguration(
            native=NativeCompilationConfiguration(
                tensorflow=_tensorflow_native(
                    compiler=CompilerObject(
                        id="tensorflow_function", parameters={"jit_compile": True}
                    )
                ),
                pytorch=_pytorch_native(
                    compiler=CompilerObject(
                        id="pytorch_compile",
                        parameters={
                            "enabled": True,
                            "mode": "max-autotune",
                            "fullgraph": True,
                            "dynamic": True,
                        },
                    )
                ),
                jax=_jax_native(
                    compiler=CompilerObject(id="jax_jit", parameters={"enabled": True})
                ),
            )
        ),
    )

###############################################################################
def test_metadata_exposes_one_capability_tree_with_explicit_runtime_reasons() -> None:
    statuses = {
        ExecutionBackend.TENSORFLOW: ExecutionBackendStatus(
            id=ExecutionBackend.TENSORFLOW,
            available=True,
            native_compilers={"tensorflow_function": True},
        ),
        ExecutionBackend.PYTORCH: ExecutionBackendStatus(
            id=ExecutionBackend.PYTORCH,
            available=True,
            native_compilers={"pytorch_compile": False},
        ),
        ExecutionBackend.JAX: ExecutionBackendStatus(
            id=ExecutionBackend.JAX,
            available=False,
            native_compilers={},
        ),
    }

    capabilities = get_compilation_capabilities(get_compilation_catalog(), statuses)

    assert set(capabilities.native) == set(ExecutionBackend)
    assert capabilities.native[ExecutionBackend.TENSORFLOW].available is True
    assert capabilities.native[ExecutionBackend.PYTORCH].reason_code == "compiler_unavailable"
    assert capabilities.native[ExecutionBackend.JAX].reason_code == "compiler_unavailable"
    assert capabilities.native[ExecutionBackend.PYTORCH].parameters[-1].type == "boolean"
    assert capabilities.native[ExecutionBackend.PYTORCH].catalog.optimizers
    assert capabilities.native[ExecutionBackend.PYTORCH].catalog.losses
    assert capabilities.native[ExecutionBackend.PYTORCH].catalog.metrics
    assert all(
        "pytorch" in item.implementations
        for item in capabilities.native[ExecutionBackend.PYTORCH].catalog.losses
    )

###############################################################################
def test_native_configuration_round_trips_and_invalid_options_are_diagnostic() -> None:
    graph = _native_graph()
    round_tripped = ModelGraph.model_validate(graph.model_dump(mode="json"))

    assert round_tripped.model_configuration == graph.model_configuration
    assert (
        round_tripped.model_configuration.native.pytorch.compiler.parameters["mode"]
        == "max-autotune"
    )
    invalid = graph.model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                    pytorch=_pytorch_native(
                        compiler=CompilerObject(
                            id="pytorch_compile",
                            parameters={
                                "enabled": False,
                                "mode": "not-a-torch-mode",
                                "fullgraph": False,
                                "dynamic": False,
                            },
                        )
                    )
                )
            )
        }
    )

    diagnostics = validate_model_configuration(invalid, ExecutionBackend.PYTORCH, False)

    assert [item.code for item in diagnostics] == ["invalid_native_compilation"]

###############################################################################
def test_native_compilation_source_uses_each_backend_native_compiler() -> None:
    graph = _native_graph()
    catalog = get_node_catalog()
    service = CompilationService(catalog, ValidationService(catalog))

    tensorflow = service.compile(graph, ExecutionBackend.TENSORFLOW, False)
    pytorch = service.compile(graph, ExecutionBackend.PYTORCH, False)
    jax = service.compile(graph, ExecutionBackend.JAX, False)

    assert tensorflow.valid is True
    assert pytorch.valid is True
    assert jax.valid is True
    assert tensorflow.code is not None
    assert pytorch.code is not None
    assert jax.code is not None
    assert "@tf.function(jit_compile=True)" in tensorflow.code
    assert "torch.compile(" in pytorch.code
    assert "mode='max-autotune'" in pytorch.code
    assert "def build_optimizer" in pytorch.code
    assert "def configured_losses" in pytorch.code
    assert "def configured_metrics" in pytorch.code
    assert "return jax.jit(model.apply)" in jax.code
    assert "keras.Model" not in tensorflow.code
    assert "keras.Model" not in pytorch.code
    assert "keras.Model" not in jax.code

###############################################################################
def test_native_tensorflow_optimizer_uses_normalized_plan_parameters() -> None:
    tensorflow = pytest.importorskip("tensorflow")
    graph = _native_graph().model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                tensorflow=_tensorflow_native(
                        compilation=cast(dict[str, object], {
                            "optimizer": {
                                "id": "adamw",
                                "parameters": {
                                    "learning_rate": 0.007,
                                    "weight_decay": 0.021,
                                    "beta_1": 0.81,
                                    "beta_2": 0.97,
                                    "epsilon": 1e-6,
                                },
                            }
                        }),
                        compiler={
                            "id": "tensorflow_function",
                            "parameters": {"jit_compile": False},
                        },
                    )
                )
            )
        }
    )

    handle = build_native_model(graph, ExecutionBackend.TENSORFLOW, zero_parameters=True)

    assert handle.compilation is not None
    optimizer = handle.compilation["optimizer_runtime"]
    configuration = optimizer.get_config()
    assert optimizer.__class__.__name__ == "AdamW"
    assert configuration["learning_rate"] == _approx(0.007)
    assert configuration["weight_decay"] == _approx(0.021)
    assert configuration["beta_1"] == _approx(0.81)
    assert configuration["beta_2"] == _approx(0.97)
    assert configuration["epsilon"] == _approx(1e-6)
    parameter = tensorflow.Variable(1.0)
    with tensorflow.GradientTape() as tape:
        loss = parameter * parameter
    gradient = tape.gradient(loss, parameter)
    optimizer.apply_gradients([(gradient, parameter)])
    assert float(parameter.numpy()) != _approx(1.0)
    assert bool(tensorflow.math.is_finite(parameter))

###############################################################################
@pytest.mark.parametrize(
    ("optimizer_id", "parameters"),
    [
        (
            "adam",
            {
                "learning_rate": 0.007,
                "weight_decay": 0.021,
                "beta_1": 0.81,
                "beta_2": 0.97,
                "epsilon": 1e-6,
            },
        ),
        (
            "adamw",
            {
                "learning_rate": 0.007,
                "weight_decay": 0.021,
                "beta_1": 0.81,
                "beta_2": 0.97,
                "epsilon": 1e-6,
            },
        ),
        (
            "sgd",
            {
                "learning_rate": 0.007,
                "momentum": 0.2,
                "nesterov": True,
                "weight_decay": 0.03,
            },
        ),
        (
            "rmsprop",
            {
                "learning_rate": 0.007,
                "rho": 0.8,
                "momentum": 0.1,
                "epsilon": 1e-6,
                "centered": True,
                "weight_decay": 0.03,
            },
        ),
        (
            "adagrad",
            {
                "learning_rate": 0.007,
                "initial_accumulator_value": 0.2,
                "epsilon": 1e-6,
                "weight_decay": 0.03,
            },
        ),
        (
            "adadelta",
            {
                "learning_rate": 0.007,
                "rho": 0.8,
                "epsilon": 1e-6,
                "weight_decay": 0.03,
            },
        ),
        (
            "adamax",
            {
                "learning_rate": 0.007,
                "beta_1": 0.81,
                "beta_2": 0.97,
                "epsilon": 1e-6,
                "weight_decay": 0.03,
            },
        ),
        (
            "nadam",
            {
                "learning_rate": 0.007,
                "beta_1": 0.81,
                "beta_2": 0.97,
                "epsilon": 1e-6,
                "weight_decay": 0.03,
            },
        ),
    ],
)
def test_native_tensorflow_optimizer_catalog_variants_update(
    optimizer_id: str,
    parameters: dict[str, object],
) -> None:
    tensorflow = pytest.importorskip("tensorflow")
    graph = _native_graph().model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                tensorflow=_tensorflow_native(
                        compilation=cast(dict[str, object], {
                            "optimizer": {
                                "id": optimizer_id,
                                "parameters": parameters,
                            }
                        }),
                        compiler={
                            "id": "tensorflow_function",
                            "parameters": {"jit_compile": False},
                        },
                    )
                )
            )
        }
    )

    handle = build_native_model(graph, ExecutionBackend.TENSORFLOW, zero_parameters=True)

    assert handle.compilation is not None
    optimizer = handle.compilation["optimizer_runtime"]
    configuration = optimizer.get_config()
    assert optimizer.__class__.__name__.lower() == optimizer_id.replace("_", "")
    assert configuration["learning_rate"] == _approx(0.007)
    for parameter_id in ("weight_decay", "beta_1", "beta_2", "epsilon", "rho", "momentum"):
        if parameter_id in parameters:
            assert configuration[parameter_id] == _approx(parameters[parameter_id])
    for parameter_id in ("nesterov", "centered"):
        if parameter_id in parameters:
            assert configuration[parameter_id] is parameters[parameter_id]
    parameter = tensorflow.Variable(1.0)
    with tensorflow.GradientTape() as tape:
        loss = parameter * parameter
    gradient = tape.gradient(loss, parameter)
    optimizer.apply_gradients([(gradient, parameter)])
    assert float(parameter.numpy()) != _approx(1.0)
    assert bool(tensorflow.math.is_finite(parameter))

###############################################################################
def test_native_tensorflow_compilation_builds_raw_loss_and_metric_functions() -> None:
    tensorflow = pytest.importorskip("tensorflow")
    graph = _native_graph().model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                    tensorflow=_tensorflow_native(
                        compilation=cast(dict[str, object], {
                            "optimizer": {"id": "adam", "parameters": {"learning_rate": 0.002}},
                            "losses": {
                                "__default__": {
                                    "id": "huber",
                                    "parameters": {"delta": 0.7},
                                }
                            },
                            "metrics": {
                                "__default__": [
                                    {
                                        "id": "precision",
                                        "parameters": {
                                            "threshold": 0.35,
                                            "top_k": None,
                                            "class_id": None,
                                        },
                                    },
                                    {
                                        "id": "prediction_mean",
                                        "parameters": {},
                                    },
                                ]
                            },
                        }),
                        compiler={
                            "id": "tensorflow_function",
                            "parameters": {"jit_compile": False},
                        },
                    )
                )
            )
        }
    )

    handle = build_native_model(graph, ExecutionBackend.TENSORFLOW, zero_parameters=True)

    assert handle.compilation is not None
    prediction = handle.outputs[0]
    target = tensorflow.zeros_like(prediction)
    loss_value = handle.compilation["loss_functions"]["output"](prediction, target)
    metric_values = {
        metric_id: function(prediction, target)
        for metric_id, function in handle.compilation["metric_functions"]["output"].items()
    }
    assert float(loss_value.numpy()) >= 0.0
    assert all(bool(tensorflow.math.is_finite(value)) for value in metric_values.values())
    assert handle.compilation["losses"]["output"]["parameters"]["delta"] == _approx(0.7)
    assert handle.compilation["metrics"]["output"][0]["id"] == "precision"

###############################################################################
def test_native_jax_optimizer_uses_optax_and_updates_finite_values() -> None:
    jax = pytest.importorskip("jax")
    optax = pytest.importorskip("optax")
    graph = _native_graph().model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                jax=_jax_native(
                        compilation={
                            "optimizer": {
                                "id": "adamw",
                                "parameters": {
                                    "learning_rate": 0.007,
                                    "weight_decay": 0.021,
                                    "beta_1": 0.81,
                                    "beta_2": 0.97,
                                    "epsilon": 1e-6,
                                },
                            }
                        },
                        compiler={"id": "jax_jit", "parameters": {"enabled": False}},
                    )
                )
            )
        }
    )

    handle = build_native_model(graph, ExecutionBackend.JAX, zero_parameters=True)

    assert handle.compilation is not None
    optimizer = handle.compilation["optimizer_runtime"]
    assert handle.compilation["optimizer"]["id"] == "adamw"
    assert handle.compilation["optimizer"]["parameters"]["learning_rate"] == _approx(0.007)
    parameter = jax.numpy.asarray(1.0)
    state = optimizer.init(parameter)
    updates, _ = optimizer.update(jax.numpy.asarray(2.0), state, parameter)
    updated = optax.apply_updates(parameter, updates)
    assert float(updated) != _approx(1.0)
    assert bool(jax.numpy.isfinite(updated))

###############################################################################
@pytest.mark.parametrize(
    ("optimizer_id", "parameters"),
    [
        (
            "sgd",
            {
                "learning_rate": 0.007,
                "momentum": 0.2,
                "nesterov": True,
                "weight_decay": 0.03,
            },
        ),
        (
            "rmsprop",
            {
                "learning_rate": 0.007,
                "rho": 0.8,
                "momentum": 0.1,
                "epsilon": 1e-6,
                "centered": True,
                "weight_decay": 0.03,
            },
        ),
        (
            "adagrad",
            {
                "learning_rate": 0.007,
                "initial_accumulator_value": 0.2,
                "epsilon": 1e-6,
                "weight_decay": 0.03,
            },
        ),
        (
            "adadelta",
            {
                "learning_rate": 0.007,
                "rho": 0.8,
                "epsilon": 1e-6,
                "weight_decay": 0.03,
            },
        ),
        (
            "adamax",
            {
                "learning_rate": 0.007,
                "beta_1": 0.81,
                "beta_2": 0.97,
                "epsilon": 1e-6,
                "weight_decay": 0.03,
            },
        ),
        (
            "nadam",
            {
                "learning_rate": 0.007,
                "beta_1": 0.81,
                "beta_2": 0.97,
                "epsilon": 1e-6,
                "weight_decay": 0.03,
            },
        ),
    ],
)
def test_native_jax_optimizer_catalog_variants_update(
    optimizer_id: str,
    parameters: dict[str, object],
) -> None:
    jax = pytest.importorskip("jax")
    optax = pytest.importorskip("optax")
    graph = _native_graph().model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                jax=_jax_native(
                        compilation={
                            "optimizer": {
                                "id": optimizer_id,
                                "parameters": parameters,
                            }
                        },
                        compiler={"id": "jax_jit", "parameters": {"enabled": False}},
                    )
                )
            )
        }
    )

    handle = build_native_model(graph, ExecutionBackend.JAX, zero_parameters=True)

    assert handle.compilation is not None
    optimizer = handle.compilation["optimizer_runtime"]
    assert handle.compilation["optimizer"]["id"] == optimizer_id
    parameter = jax.numpy.asarray(1.0)
    state = optimizer.init(parameter)
    updates, _ = optimizer.update(jax.numpy.asarray(2.0), state, parameter)
    updated = optax.apply_updates(parameter, updates)
    assert float(updated) != _approx(1.0)
    assert bool(jax.numpy.isfinite(updated))

###############################################################################
def test_native_jax_compilation_builds_raw_loss_and_metric_functions() -> None:
    jax = pytest.importorskip("jax")
    graph = _native_graph().model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                jax=_jax_native(
                        compilation=cast(dict[str, object], {
                            "optimizer": {"id": "adam", "parameters": {"learning_rate": 0.002}},
                            "losses": {
                                "__default__": {
                                    "id": "huber",
                                    "parameters": {"delta": 0.7},
                                }
                            },
                            "metrics": {
                                "__default__": [
                                    {
                                        "id": "precision",
                                        "parameters": {
                                            "threshold": 0.35,
                                            "top_k": None,
                                            "class_id": None,
                                        },
                                    },
                                    {"id": "prediction_mean", "parameters": {}},
                                ]
                            },
                        }),
                        compiler={"id": "jax_jit", "parameters": {"enabled": False}},
                    )
                )
            )
        }
    )

    handle = build_native_model(graph, ExecutionBackend.JAX, zero_parameters=True)

    assert handle.compilation is not None
    prediction = handle.outputs[0]
    target = jax.numpy.zeros_like(prediction)
    loss_value = handle.compilation["loss_functions"]["output"](prediction, target)
    metric_values = {
        metric_id: function(prediction, target)
        for metric_id, function in handle.compilation["metric_functions"]["output"].items()
    }
    assert float(loss_value) >= 0.0
    assert all(bool(jax.numpy.isfinite(value)) for value in metric_values.values())
    assert handle.compilation["losses"]["output"]["parameters"]["delta"] == _approx(0.7)
    assert handle.compilation["metrics"]["output"][0]["id"] == "precision"

###############################################################################
@pytest.mark.parametrize("backend", [ExecutionBackend.TENSORFLOW, ExecutionBackend.JAX])
def test_native_compilation_verifier_runs_raw_optimizer_update(
    backend: ExecutionBackend,
) -> None:
    pytest.importorskip("tensorflow" if backend is ExecutionBackend.TENSORFLOW else "jax")
    graph = _native_graph().model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                tensorflow=_tensorflow_native(
                        compiler={
                            "id": "tensorflow_function",
                            "parameters": {"jit_compile": False},
                        }
                    ),
                jax=_jax_native(
                        compiler={"id": "jax_jit", "parameters": {"enabled": False}}
                    ),
                )
            )
        }
    )
    handle = build_native_model(graph, backend, zero_parameters=True)

    result = _verify_native_compilation(graph, handle, lambda: None)

    assert result is not None and result["verified"] is True
    optimizer = result.get("optimizer")
    assert isinstance(optimizer, dict)
    optimizer = cast(dict[str, object], optimizer)
    assert optimizer.get("instantiated") is True
    assert optimizer.get("updated") is True

###############################################################################
@pytest.mark.parametrize("backend", [ExecutionBackend.TENSORFLOW, ExecutionBackend.JAX])
def test_native_crossentropy_parameters_execute_without_keras_fallback(
    backend: ExecutionBackend,
) -> None:
    framework = pytest.importorskip(
        "tensorflow" if backend is ExecutionBackend.TENSORFLOW else "jax"
    )
    compilation: dict[str, object] = {
        "optimizer": {"id": "adam", "parameters": {"learning_rate": 0.002}},
        "losses": {
            "__default__": {
                "id": "categorical_crossentropy",
                "parameters": {
                    "from_logits": True,
                    "label_smoothing": 0.1,
                    "axis": -1,
                },
            }
        },
    }
    graph = _native_graph().model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                tensorflow=_tensorflow_native(
                        compilation=compilation,
                        compiler={
                            "id": "tensorflow_function",
                            "parameters": {"jit_compile": False},
                        },
                    ),
                jax=_jax_native(
                        compilation=compilation,
                        compiler={"id": "jax_jit", "parameters": {"enabled": False}},
                    ),
                )
            )
        }
    )
    handle = build_native_model(graph, backend, zero_parameters=True)
    assert handle.compilation is not None
    compilation_result = cast(dict[str, object], handle.compilation)
    prediction = handle.outputs[0]
    target = (
        framework.one_hot([0], depth=2)
        if backend is ExecutionBackend.TENSORFLOW
        else framework.nn.one_hot(framework.numpy.asarray([0]), 2)
    )

    loss_functions = cast(dict[str, Callable[..., object]], compilation_result["loss_functions"])
    value = loss_functions["output"](prediction, target)

    finite = (
        bool(framework.math.is_finite(value))
        if backend is ExecutionBackend.TENSORFLOW
        else bool(framework.numpy.isfinite(value))
    )
    assert finite
    losses = cast(dict[str, dict[str, object]], compilation_result["losses"])
    parameters = cast(dict[str, object], losses["output"]["parameters"])
    assert parameters["from_logits"] is True
    assert parameters["label_smoothing"] == _approx(0.1)

###############################################################################
def test_compilation_rejects_a_native_parameter_without_falling_back_to_keras() -> None:
    graph = _native_graph().model_copy(deep=True)
    dense = next(node for node in graph.nodes if node.id == "dense")
    dense.parameters["kernel_initializer"] = "he_normal"
    catalog = get_node_catalog()
    service = CompilationService(catalog, ValidationService(catalog))

    result = service.compile(graph, ExecutionBackend.PYTORCH, False)

    assert result.valid is False
    assert result.code is None
    assert any(
        item.code == "unsupported_parameter_value"
        and item.parameter_id == "kernel_initializer"
        for item in result.diagnostics
    )

# Consolidated from test_pytorch_compilation_helpers.py

# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportOptionalCall=false, reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportUntypedFunctionDecorator=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false



pytest.importorskip("torch", reason="PyTorch backend is not installed")



###############################################################################
def _keras_configuration(**payload: object) -> KerasProfileConfiguration:
    return KerasProfileConfiguration.model_validate(payload)


###############################################################################
def _test_pytorch_compilation_helpers__pytorch_native(**payload: object) -> PyTorchNativeConfiguration:
    return PyTorchNativeConfiguration.model_validate(payload)

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
                _node("input", "input.tabular", {"shape": [8], "dtype": "float32"}),
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
                _node("input", "input.tabular", {"shape": [8], "dtype": "float32"}),
                _node("head", "layer.dense", {"units": 1, "activation": "sigmoid"}),
                _node("output", "output.classification"),
            ],
            ["input", "head", "output"],
            (1, 1),
            "binary",
        ),
        "multiclass_classification": graph(
            [
                _node("input", "input.tabular", {"shape": [8], "dtype": "float32"}),
                _node("head", "layer.dense", {"units": 3, "activation": "softmax"}),
                _node("output", "output.classification"),
            ],
            ["input", "head", "output"],
            (1, 3),
            "multiclass",
        ),
        "recurrent": graph(
            [
                _node("input", "input.sequence", {"shape": [6, 4], "dtype": "float32"}),
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
                _node("input", "input.sequence", {"shape": [6, 4], "dtype": "float32"}),
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
                _node("input", "input.image", {"shape": [16, 16, 1], "dtype": "float32"}),
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
                pytorch=_test_pytorch_compilation_helpers__pytorch_native(
                    compilation=CompilationPlan(
                        optimizer=CompilationObject(
                            id="adamw", parameters={"learning_rate": 0.001, "weight_decay": 0.01}
                        ),
                        losses={"__default__": CompilationObject(id=loss)},
                        metrics={"__default__": [CompilationObject(id=item) for item in metrics]},
                    )
                )
            )
        )
    else:
        configuration = ModelConfiguration(
            keras=_keras_configuration(
                compilation=CompilationPlan(
                    losses={"__default__": CompilationObject(id=loss)},
                    metrics={"__default__": [CompilationObject(id=item) for item in metrics]},
                )
            )
        )
    return graph.model_copy(update={"model_configuration": configuration})

###############################################################################
def _multi_output_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("left_input", "input.tabular", {"shape": [8], "dtype": "float32"}),
            _node("right_input", "input.tabular", {"shape": [8], "dtype": "float32"}),
            _node("left_head", "layer.dense", {"units": 1, "activation": "sigmoid"}).model_copy(
                update={"label": "prediction"}
            ),
            _node("right_head", "layer.dense", {"units": 2, "activation": "softmax"}).model_copy(
                update={"label": "score"}
            ),
        ],
        edges=[
            _edge("left", "left_input", "left_head"),
            _edge("right", "right_input", "right_head"),
        ],
    )

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
    outputs = cast(list[dict[str, Any]], result["outputs"])
    assert [tuple(item["shape"]) for item in outputs] == [expected_shape]

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
    outputs = cast(list[dict[str, Any]], result["outputs"])
    assert [tuple(item["shape"]) for item in outputs] == [expected_shape]

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
                pytorch=_test_pytorch_compilation_helpers__pytorch_native(
                        compilation=CompilationPlan(
                            losses={"__default__": CompilationObject(id="categorical_crossentropy")},
                            metrics={"__default__": [CompilationObject(id="mean_squared_error")]},
                        )
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
def test_multi_output_configuration_resolves_all_selector_forms() -> None:
    graph = _multi_output_graph()
    graph = graph.model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                pytorch=_test_pytorch_compilation_helpers__pytorch_native(
                        compilation=CompilationPlan(
                            losses={"left_head:output": CompilationObject(id="binary_crossentropy")},
                            metrics={"prediction": [CompilationObject(id="binary_accuracy")]},
                        )
                    )
                )
            ),
        }
    )

    diagnostics = validate_model_configuration(graph, ExecutionBackend.PYTORCH, False)

    assert diagnostics == []

###############################################################################
def test_unknown_output_selector_reports_available_output_specific_selectors() -> None:
    graph = _multi_output_graph()
    graph = graph.model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                pytorch=_test_pytorch_compilation_helpers__pytorch_native(
                        compilation=CompilationPlan(
                            losses={"missing-output": CompilationObject(id="binary_crossentropy")},
                        )
                    )
                )
            )
        }
    )

    diagnostic = next(
        item
        for item in validate_model_configuration(graph, ExecutionBackend.PYTORCH, False)
        if item.code == "unknown_compile_output"
    )

    assert diagnostic.parameter_id == "losses"
    assert diagnostic.details["output_selector"] == "missing-output"
    assert "left_head:output" in diagnostic.details["available_selectors"]

###############################################################################
def test_unknown_classification_width_does_not_assume_multiclass_semantics() -> None:
    graph = ModelGraph(
        nodes=[
            _node("input", "input.tabular", {"shape": [8], "dtype": "float32"}),
            _node("output", "output.classification"),
        ],
        edges=[_edge("input-output", "input", "output")],
    )

    diagnostics = validate_model_configuration(graph, ExecutionBackend.PYTORCH, True)

    assert not {
        "invalid_output_activation",
        "invalid_binary_output_shape",
        "invalid_multiclass_output_shape",
    } & {item.code for item in diagnostics}

###############################################################################
def test_native_non_pytorch_validation_ignores_pytorch_loss_and_metric_settings() -> None:
    graph, _, _ = _workflow_graphs()["binary_classification"]
    graph = graph.model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                    tensorflow=TensorFlowNativeConfiguration(),
                pytorch=_test_pytorch_compilation_helpers__pytorch_native(
                        compilation=CompilationPlan(
                            optimizer=CompilationObject(id="not-an-optimizer"),
                            losses={"missing-output": CompilationObject(id="not-a-loss")},
                            metrics={"missing-output": [CompilationObject(id="not-a-metric")]},
                        ),
                    ),
                )
            )
        }
    )

    assert validate_model_configuration(graph, ExecutionBackend.TENSORFLOW, False) == []

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
        (
            "adagrad",
            {
                "learning_rate": 0.01,
                "initial_accumulator_value": 0.2,
                "epsilon": 1e-7,
                "weight_decay": 0.01,
            },
        ),
        (
            "adadelta",
            {"learning_rate": 0.01, "rho": 0.8, "epsilon": 1e-6, "weight_decay": 0.01},
        ),
        (
            "adamax",
            {
                "learning_rate": 0.01,
                "beta_1": 0.8,
                "beta_2": 0.99,
                "epsilon": 1e-7,
                "weight_decay": 0.01,
            },
        ),
        (
            "nadam",
            {
                "learning_rate": 0.01,
                "beta_1": 0.8,
                "beta_2": 0.99,
                "epsilon": 1e-7,
                "weight_decay": 0.01,
            },
        ),
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
                pytorch=_test_pytorch_compilation_helpers__pytorch_native(
                        compilation=CompilationPlan(
                            optimizer=CompilationObject(id=optimizer, parameters=parameters),
                            losses={"__default__": CompilationObject(id="mean_squared_error")},
                        )
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
    ("workflow", "loss_id", "parameters"),
    [
        ("dense_regression", "mean_absolute_error", {}),
        ("dense_regression", "huber", {"delta": 0.5}),
        (
            "binary_classification",
            "binary_crossentropy",
            {"from_logits": True, "label_smoothing": 0.1, "axis": -1},
        ),
        (
            "multiclass_classification",
            "categorical_crossentropy",
            {"from_logits": True, "label_smoothing": 0.1, "axis": -1},
        ),
        (
            "multiclass_classification",
            "sparse_categorical_crossentropy",
            {"from_logits": True, "label_smoothing": 0.1, "axis": -1, "ignore_class": None},
        ),
        ("multiclass_classification", "kullback_leibler_divergence", {}),
        ("dense_regression", "log_cosh", {}),
        ("dense_regression", "mean_absolute_percentage_error", {}),
        ("dense_regression", "mean_squared_logarithmic_error", {}),
        ("binary_classification", "hinge", {}),
        ("binary_classification", "squared_hinge", {}),
        ("multiclass_classification", "categorical_hinge", {}),
        ("dense_regression", "poisson", {}),
        ("multiclass_classification", "dice", {"axis": -1}),
        (
            "multiclass_classification",
            "tversky",
            {"alpha": 0.3, "beta": 0.7, "axis": -1},
        ),
    ],
)
def test_native_loss_helpers_cover_remaining_catalog_variants(
    workflow: str, loss_id: str, parameters: dict[str, Any]
) -> None:
    graph, _, _ = _workflow_graphs()[workflow]
    graph = graph.model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                pytorch=_test_pytorch_compilation_helpers__pytorch_native(
                        compilation=CompilationPlan(
                            losses={"__default__": CompilationObject(id=loss_id, parameters=parameters)}
                        )
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
    assert result["compilation"]["losses"]["output"]["id"] == loss_id

###############################################################################
@pytest.mark.parametrize(
    ("workflow", "metric_id", "parameters"),
    [
        ("multiclass_classification", "top_k_categorical_accuracy", {"k": 2}),
        ("multiclass_classification", "sparse_top_k_categorical_accuracy", {"k": 2}),
        (
            "binary_classification",
            "precision",
            {"threshold": 0.35, "top_k": None, "class_id": None},
        ),
        (
            "binary_classification",
            "recall",
            {"threshold": 0.65, "top_k": None, "class_id": None},
        ),
        (
            "binary_classification",
            "auc",
            {
                "num_thresholds": 11,
                "curve": "PR",
                "summation_method": "majoring",
                "thresholds": [0.2, 0.5, 0.8],
                "multi_label": False,
                "num_labels": None,
                "label_weights": None,
                "from_logits": False,
            },
        ),
        ("binary_classification", "true_positives", {"threshold": 0.4}),
        ("binary_classification", "true_negatives", {"threshold": 0.4}),
        ("binary_classification", "false_positives", {"threshold": 0.4}),
        ("binary_classification", "false_negatives", {"threshold": 0.4}),
        ("dense_regression", "mean_absolute_percentage_error", {}),
        ("dense_regression", "mean_squared_logarithmic_error", {}),
        ("dense_regression", "cosine_similarity", {"axis": -1}),
        ("dense_regression", "log_cosh_error", {}),
        ("dense_regression", "prediction_mean", {}),
        ("dense_regression", "prediction_sum", {}),
    ],
)
def test_native_metric_helpers_cover_remaining_catalog_variants(
    workflow: str, metric_id: str, parameters: dict[str, Any]
) -> None:
    graph, _, _ = _workflow_graphs()[workflow]
    graph = graph.model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                pytorch=_test_pytorch_compilation_helpers__pytorch_native(
                        compilation=CompilationPlan(
                            metrics={
                                "__default__": [
                                    CompilationObject(id=metric_id, parameters=parameters)
                                ]
                            }
                        )
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
    value = result["compilation"]["metrics"]["output"][metric_id]
    assert isinstance(value, float)
    assert isfinite(value)

# Consolidated from test_compiler.py

###############################################################################
def test_compiler_generates_unique_keyword_safe_variable_names() -> None:
    graph = ModelGraph(
        nodes=[
            NodeInstance(
                id="input",
                definition_id="input.tabular",
                label="class",
                position=Position(x=0, y=0),
                parameters={"shape": [4], "dtype": "float32"},
            ),
            NodeInstance(
                id="dense-a",
                definition_id="layer.dense",
                label="hidden",
                position=Position(x=1, y=0),
                parameters={"units": 4},
            ),
            NodeInstance(
                id="dense-b",
                definition_id="layer.dense",
                label="hidden",
                position=Position(x=2, y=0),
                parameters={"units": 2},
            ),
            NodeInstance(
                id="output",
                definition_id="output.classification",
                label="result",
                position=Position(x=3, y=0),
                parameters={},
            ),
        ],
        edges=[
            EdgeInstance(
                id="e1",
                source_node_id="input",
                source_port_id="output",
                target_node_id="dense-a",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e2",
                source_node_id="dense-a",
                source_port_id="output",
                target_node_id="dense-b",
                target_port_id="input",
            ),
            EdgeInstance(
                id="e3",
                source_node_id="dense-b",
                source_port_id="output",
                target_node_id="output",
                target_port_id="input",
            ),
        ],
    )
    ir = compile_to_ir(graph, ExecutionBackend.PYTORCH, True, get_node_catalog())
    names = [node.variable_name for node in ir.ordered_nodes]
    assert names == ["node_class", "hidden", "hidden_2", "result"]
    dense_binding = ir.ordered_nodes[1].input_bindings[0]
    assert dense_binding.source_node_id == "input"
    assert dense_binding.source_port_id == "output"
    assert dense_binding.target_node_id == "dense-a"
    assert dense_binding.target_port_id == "input"
