from __future__ import annotations

import pytest

from common.catalogs.loader import get_node_catalog
from domain.compilation_capabilities import get_compilation_capabilities
from domain.compilation_catalog import get_compilation_catalog
from domain.enums import ExecutionBackend
from domain.graph import (
    EdgeInstance,
    JaxNativeConfiguration,
    ModelConfiguration,
    ModelGraph,
    NativeCompilationConfiguration,
    NodeInstance,
    Position,
    PyTorchNativeConfiguration,
    TensorFlowNativeConfiguration,
)
from domain.metadata import ExecutionBackendStatus
from services.compilation_service import CompilationService
from services.validation_service import ValidationService
from validation.compile_configuration import validate_model_configuration
from models.native_runtime import build_native_model
from models.worker_backend import _verify_native_compilation


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
                tensorflow=TensorFlowNativeConfiguration(jit_compile=True),
                pytorch=PyTorchNativeConfiguration(
                    compile_enabled=True,
                    mode="max-autotune",
                    fullgraph=True,
                    dynamic=True,
                ),
                jax=JaxNativeConfiguration(jit_enabled=True),
            )
        ),
    )

###############################################################################
def test_metadata_exposes_one_capability_tree_with_explicit_runtime_reasons() -> None:
    statuses = {
        ExecutionBackend.TENSORFLOW: ExecutionBackendStatus(
            id=ExecutionBackend.TENSORFLOW,
            available=True,
            native_compilers={"tensorflow_jit": True},
        ),
        ExecutionBackend.PYTORCH: ExecutionBackendStatus(
            id=ExecutionBackend.PYTORCH,
            available=True,
            native_compilers={"torch_compile": False},
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
    assert round_tripped.model_configuration.native.pytorch.mode == "max-autotune"
    invalid = graph.model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                    pytorch=PyTorchNativeConfiguration(mode="not-a-torch-mode")
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
                    tensorflow=TensorFlowNativeConfiguration(
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
    assert configuration["learning_rate"] == pytest.approx(0.007)
    assert configuration["weight_decay"] == pytest.approx(0.021)
    assert configuration["beta_1"] == pytest.approx(0.81)
    assert configuration["beta_2"] == pytest.approx(0.97)
    assert configuration["epsilon"] == pytest.approx(1e-6)
    parameter = tensorflow.Variable(1.0)
    with tensorflow.GradientTape() as tape:
        loss = parameter * parameter
    gradient = tape.gradient(loss, parameter)
    optimizer.apply_gradients([(gradient, parameter)])
    assert float(parameter.numpy()) != pytest.approx(1.0)
    assert bool(tensorflow.math.is_finite(parameter))


###############################################################################
def test_native_tensorflow_compilation_builds_raw_loss_and_metric_functions() -> None:
    tensorflow = pytest.importorskip("tensorflow")
    graph = _native_graph().model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                    tensorflow=TensorFlowNativeConfiguration(
                        compilation={
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
                        },
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
    assert handle.compilation["losses"]["output"]["parameters"]["delta"] == pytest.approx(0.7)
    assert handle.compilation["metrics"]["output"][0]["id"] == "precision"


###############################################################################
def test_native_jax_optimizer_uses_optax_and_updates_finite_values() -> None:
    jax = pytest.importorskip("jax")
    optax = pytest.importorskip("optax")
    graph = _native_graph().model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                    jax=JaxNativeConfiguration(
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
    assert handle.compilation["optimizer"]["parameters"]["learning_rate"] == pytest.approx(0.007)
    parameter = jax.numpy.asarray(1.0)
    state = optimizer.init(parameter)
    updates, _ = optimizer.update(jax.numpy.asarray(2.0), state, parameter)
    updated = optax.apply_updates(parameter, updates)
    assert float(updated) != pytest.approx(1.0)
    assert bool(jax.numpy.isfinite(updated))


###############################################################################
def test_native_jax_compilation_builds_raw_loss_and_metric_functions() -> None:
    jax = pytest.importorskip("jax")
    graph = _native_graph().model_copy(
        update={
            "model_configuration": ModelConfiguration(
                native=NativeCompilationConfiguration(
                    jax=JaxNativeConfiguration(
                        compilation={
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
                        },
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
    assert handle.compilation["losses"]["output"]["parameters"]["delta"] == pytest.approx(0.7)
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
                    tensorflow=TensorFlowNativeConfiguration(
                        compiler={
                            "id": "tensorflow_function",
                            "parameters": {"jit_compile": False},
                        }
                    ),
                    jax=JaxNativeConfiguration(
                        compiler={"id": "jax_jit", "parameters": {"enabled": False}}
                    ),
                )
            )
        }
    )
    handle = build_native_model(graph, backend, zero_parameters=True)

    result = _verify_native_compilation(graph, handle, lambda: None)

    assert result["verified"] is True
    assert result["optimizer"]["instantiated"] is True
    assert result["optimizer"]["updated"] is True

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
