# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import ast
from typing import Any, cast

import numpy as np
import pytest

from common.catalogs.loader import get_node_catalog
from common.catalogs.operation_emitter import emit_operation_source
from compilation.compiler import compile_to_ir
from domain.compilation_catalog import (
    get_compilation_catalog,
    get_compilation_catalog_provider,
)
from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from domain.manifests import (
    ImplementationMapping,
    NodeManifest,
    OperationMetadata,
    ParameterDefinition,
    PortDefinition,
    ShapeRule,
)
from exporters.keras_exporter import KerasSourceExporter
from models.native_runtime import build_native_model, forward_native
from services.validation_service import ValidationService
from validation.safe_expression import SafeExpressionError, validate_safe_expression
from validation.shape_inference import infer_shapes

###############################################################################
def _node(
    node_id: str,
    definition_id: str,
    parameters: dict[str, Any] | None = None,
) -> NodeInstance:
    return NodeInstance(
        id=node_id,
        definition_id=definition_id,
        definition_version=1,
        position=Position(x=0, y=0),
        parameters=parameters or {},
    )

###############################################################################
def _edge(edge_id: str, source: str, target: str, target_port: str = "input") -> EdgeInstance:
    return EdgeInstance(
        id=edge_id,
        source_node_id=source,
        source_port_id="output",
        target_node_id=target,
        target_port_id=target_port,
    )

###############################################################################
def _conv1d_graph() -> ModelGraph:
    return ModelGraph(
        nodes=[
            _node("input", "input.sequence", {"shape": [8, 3], "dtype": "float32"}),
            _node(
                "conv",
                "conv.conv1d",
                {
                    "filters": 4,
                    "kernel_size": [3],
                    "strides": [2],
                    "padding": "same",
                    "activation": "relu",
                    "data_format": "channels_last",
                    "dilation_rate": [1],
                    "groups": 1,
                    "use_bias": True,
                },
            ),
            _node("output", "output.sequence"),
        ],
        edges=[_edge("e1", "input", "conv"), _edge("e2", "conv", "output")],
    )

###############################################################################
def _operation_graph(
    definition_id: str,
    input_shapes: list[list[Any]],
    parameters: dict[str, Any] | None = None,
    input_dtypes: list[str] | None = None,
) -> ModelGraph:
    input_ids = [f"input_{index}" for index in range(len(input_shapes))]
    dtypes = input_dtypes or ["float32"] * len(input_shapes)
    if len(dtypes) != len(input_shapes):
        raise ValueError("input_dtypes must match input_shapes")
    nodes = [
        _node(node_id, "input.tensor", {"shape": shape, "dtype": dtype})
        for node_id, shape, dtype in zip(input_ids, input_shapes, dtypes, strict=True)
    ]
    nodes.extend(
        [
            _node("operation", definition_id, parameters),
            _node("output", "output.custom"),
        ]
    )
    target_port = "inputs" if len(input_ids) > 1 else "input"
    edges = [
        _edge(f"input_edge_{index}", input_id, "operation", target_port)
        for index, input_id in enumerate(input_ids)
    ]
    edges.append(_edge("output_edge", "operation", "output"))
    return ModelGraph(nodes=nodes, edges=edges)


_UNARY_OPERATION_CASES: list[tuple[str, dict[str, int], dict[str, str]]] = [
    ("math.ceil", {}, {"keras": "keras.ops.ceil", "tensorflow": "tf.math.ceil", "pytorch": "torch.ceil", "jax": "jnp.ceil"}),
    ("math.floor", {}, {"keras": "keras.ops.floor", "tensorflow": "tf.math.floor", "pytorch": "torch.floor", "jax": "jnp.floor"}),
    ("math.round", {}, {"keras": "keras.ops.round", "tensorflow": "tf.math.round", "pytorch": "torch.round", "jax": "jnp.round"}),
    ("math.sin", {}, {"keras": "keras.ops.sin", "tensorflow": "tf.math.sin", "pytorch": "torch.sin", "jax": "jnp.sin"}),
    ("math.cos", {}, {"keras": "keras.ops.cos", "tensorflow": "tf.math.cos", "pytorch": "torch.cos", "jax": "jnp.cos"}),
    ("math.tan", {}, {"keras": "keras.ops.tan", "tensorflow": "tf.math.tan", "pytorch": "torch.tan", "jax": "jnp.tan"}),
    ("math.exp", {}, {"keras": "keras.ops.exp", "tensorflow": "tf.math.exp", "pytorch": "torch.exp", "jax": "jnp.exp"}),
    ("math.log", {}, {"keras": "keras.ops.log", "tensorflow": "tf.math.log", "pytorch": "torch.log", "jax": "jnp.log"}),
    ("math.log1p", {}, {"keras": "keras.ops.log1p", "tensorflow": "tf.math.log1p", "pytorch": "torch.log1p", "jax": "jnp.log1p"}),
    ("math.erf", {}, {"keras": "keras.ops.erf", "tensorflow": "tf.math.erf", "pytorch": "torch.erf", "jax": "jax.scipy.special.erf"}),
    ("math.sqrt", {}, {"keras": "keras.ops.sqrt", "tensorflow": "tf.math.sqrt", "pytorch": "torch.sqrt", "jax": "jnp.sqrt"}),
    ("math.rsqrt", {}, {"keras": "keras.ops.rsqrt", "tensorflow": "tf.math.rsqrt", "pytorch": "torch.rsqrt", "jax": "jax.lax.rsqrt"}),
    ("math.square", {}, {"keras": "keras.ops.square", "tensorflow": "tf.math.square", "pytorch": "torch.square", "jax": "jnp.square"}),
    ("math.negative", {}, {"keras": "keras.ops.negative", "tensorflow": "tf.math.negative", "pytorch": "torch.neg", "jax": "jnp.negative"}),
    ("math.reciprocal", {}, {"keras": "keras.ops.reciprocal", "tensorflow": "tf.math.reciprocal", "pytorch": "torch.reciprocal", "jax": "jnp.reciprocal"}),
    ("math.sign", {}, {"keras": "keras.ops.sign", "tensorflow": "tf.math.sign", "pytorch": "torch.sign", "jax": "jnp.sign"}),
    ("math.clip", {"min_value": -1, "max_value": 1}, {"keras": "keras.ops.clip", "tensorflow": "tf.clip_by_value", "pytorch": "torch.clamp", "jax": "jnp.clip"}),
    ("math.sinh", {}, {"keras": "keras.ops.sinh", "tensorflow": "tf.math.sinh", "pytorch": "torch.sinh", "jax": "jnp.sinh"}),
    ("math.cosh", {}, {"keras": "keras.ops.cosh", "tensorflow": "tf.math.cosh", "pytorch": "torch.cosh", "jax": "jnp.cosh"}),
    ("math.tanh", {}, {"keras": "keras.ops.tanh", "tensorflow": "tf.math.tanh", "pytorch": "torch.tanh", "jax": "jnp.tanh"}),
    ("math.relu", {}, {"keras": "keras.ops.relu", "tensorflow": "tf.nn.relu", "pytorch": "torch.relu", "jax": "jax.nn.relu"}),
    ("math.gelu", {}, {"keras": "keras.ops.gelu", "tensorflow": "tf.nn.gelu", "pytorch": "torch.nn.functional.gelu", "jax": "jax.nn.gelu"}),
    ("math.sigmoid", {}, {"keras": "keras.ops.sigmoid", "tensorflow": "tf.nn.sigmoid", "pytorch": "torch.sigmoid", "jax": "jax.nn.sigmoid"}),
    ("math.softplus", {}, {"keras": "keras.ops.softplus", "tensorflow": "tf.nn.softplus", "pytorch": "torch.nn.functional.softplus", "jax": "jax.nn.softplus"}),
]


###############################################################################
def _native_float_input(backend: ExecutionBackend) -> Any:
    values = np.asarray([[0.25, 0.5, 1.25, 2.0]], dtype=np.float32)
    if backend is ExecutionBackend.TENSORFLOW:
        import tensorflow as tf

        return tf.convert_to_tensor(values)
    if backend is ExecutionBackend.PYTORCH:
        import torch

        torch_module = cast(Any, torch)
        return torch_module.from_numpy(values)
    import jax.numpy as jnp

    jnp_module = cast(Any, jnp)
    return jnp_module.asarray(values)


###############################################################################
def _native_values(backend: ExecutionBackend, values: Any, dtype: str) -> Any:
    array = np.asarray(values, dtype=np.bool_ if dtype == "bool" else np.float32)
    if backend is ExecutionBackend.TENSORFLOW:
        import tensorflow as tf

        return tf.convert_to_tensor(array)
    if backend is ExecutionBackend.PYTORCH:
        import torch

        torch_module = cast(Any, torch)
        return torch_module.from_numpy(array)
    import jax.numpy as jnp

    jnp_module = cast(Any, jnp)
    return jnp_module.asarray(array)


_ELEMENTWISE_BOOLEAN_CASES: list[tuple[str, dict[str, str]]] = [
    ("math.equal", {"keras": "keras.ops.equal", "tensorflow": "tf.equal", "pytorch": "torch.eq", "jax": "jnp.equal"}),
    ("math.not_equal", {"keras": "keras.ops.not_equal", "tensorflow": "tf.not_equal", "pytorch": "torch.ne", "jax": "jnp.not_equal"}),
    ("math.greater", {"keras": "keras.ops.greater", "tensorflow": "tf.greater", "pytorch": "torch.gt", "jax": "jnp.greater"}),
    ("math.greater_equal", {"keras": "keras.ops.greater_equal", "tensorflow": "tf.greater_equal", "pytorch": "torch.ge", "jax": "jnp.greater_equal"}),
    ("math.less", {"keras": "keras.ops.less", "tensorflow": "tf.less", "pytorch": "torch.lt", "jax": "jnp.less"}),
    ("math.less_equal", {"keras": "keras.ops.less_equal", "tensorflow": "tf.less_equal", "pytorch": "torch.le", "jax": "jnp.less_equal"}),
    ("math.logical_and", {"keras": "keras.ops.logical_and", "tensorflow": "tf.logical_and", "pytorch": "torch.logical_and", "jax": "jnp.logical_and"}),
    ("math.logical_or", {"keras": "keras.ops.logical_or", "tensorflow": "tf.logical_or", "pytorch": "torch.logical_or", "jax": "jnp.logical_or"}),
    ("math.logical_xor", {"keras": "keras.ops.logical_xor", "tensorflow": "tf.math.logical_xor", "pytorch": "torch.logical_xor", "jax": "jnp.logical_xor"}),
]


_REDUCTION_CASES: list[tuple[str, dict[str, Any], dict[str, str], str, list[int | None], str]] = [
    ("math.argmax", {"axis": 1}, {"keras": "keras.ops.argmax", "tensorflow": "tf.argmax", "pytorch": "torch.argmax", "jax": "jnp.argmax"}, "float32", [None, 3], "int64"),
    ("math.sum", {"axis": 1, "keepdims": True}, {"keras": "keras.ops.sum", "tensorflow": "tf.reduce_sum", "pytorch": "torch.sum", "jax": "jnp.sum"}, "float32", [None, 1, 3], "float32"),
    ("math.mean", {"axis": 1, "keepdims": True}, {"keras": "keras.ops.mean", "tensorflow": "tf.reduce_mean", "pytorch": "torch.mean", "jax": "jnp.mean"}, "float32", [None, 1, 3], "float32"),
    ("math.min", {"axis": 1, "keepdims": True}, {"keras": "keras.ops.min", "tensorflow": "tf.reduce_min", "pytorch": "torch.amin", "jax": "jnp.min"}, "float32", [None, 1, 3], "float32"),
    ("math.max", {"axis": 1, "keepdims": True}, {"keras": "keras.ops.max", "tensorflow": "tf.reduce_max", "pytorch": "torch.amax", "jax": "jnp.max"}, "float32", [None, 1, 3], "float32"),
    ("math.prod", {"axis": 1, "keepdims": True}, {"keras": "keras.ops.prod", "tensorflow": "tf.reduce_prod", "pytorch": "torch.prod", "jax": "jnp.prod"}, "float32", [None, 1, 3], "float32"),
    ("math.all", {"axis": 1, "keepdims": True}, {"keras": "keras.ops.all", "tensorflow": "tf.reduce_all", "pytorch": "torch.all", "jax": "jnp.all"}, "bool", [None, 1, 3], "bool"),
    ("math.any", {"axis": 1, "keepdims": True}, {"keras": "keras.ops.any", "tensorflow": "tf.reduce_any", "pytorch": "torch.any", "jax": "jnp.any"}, "bool", [None, 1, 3], "bool"),
    ("math.variance", {"axis": 1, "keepdims": True}, {"keras": "keras.ops.var", "tensorflow": "tf.math.reduce_variance", "pytorch": "torch.var", "jax": "jnp.var"}, "float32", [None, 1, 3], "float32"),
    ("math.stddev", {"axis": 1, "keepdims": True}, {"keras": "keras.ops.std", "tensorflow": "tf.math.reduce_std", "pytorch": "torch.std", "jax": "jnp.std"}, "float32", [None, 1, 3], "float32"),
    ("math.logsumexp", {"axis": 1, "keepdims": True}, {"keras": "keras.ops.logsumexp", "tensorflow": "tf.math.reduce_logsumexp", "pytorch": "torch.logsumexp", "jax": "jax.scipy.special.logsumexp"}, "float32", [None, 1, 3], "float32"),
    ("math.reduce", {"operation": "sum", "axis": 1, "keepdims": True}, {"keras": "keras.ops.sum", "tensorflow": "tf.reduce_sum", "pytorch": "torch.sum", "jax": "jnp.sum"}, "float32", [None, 1, 3], "float32"),
]

###############################################################################
def test_conv1d_manifest_exposes_capabilities_and_exact_footer_shape() -> None:
    catalog = get_node_catalog()
    manifest = catalog.get("conv.conv1d")

    assert {
        "filters",
        "kernel_size",
        "strides",
        "dilation_rate",
        "groups",
        "padding",
        "activation",
        "data_format",
        "use_bias",
    } <= {parameter.id for parameter in manifest.parameters}
    assert manifest.implementations["keras"].execution_backends == {
        ExecutionBackend.TENSORFLOW: True,
        ExecutionBackend.PYTORCH: True,
        ExecutionBackend.JAX: True,
    }
    assert manifest.implementations["tensorflow"].adapter == "conv1d"
    assert manifest.implementations["pytorch"].adapter == "conv1d"
    assert manifest.implementations["jax"].adapter == "conv1d"

    summary = next(
        item for item in infer_shapes(_conv1d_graph(), catalog) if item.node_id == "conv"
    )
    assert summary.outputs["output"].shape == [None, 4, 4]
    assert summary.outputs["output"].dtype.value == "float32"

###############################################################################
@pytest.mark.parametrize("backend", list(ExecutionBackend))
@pytest.mark.parametrize("keras_enabled", [True, False])
def test_conv1d_validation_accepts_all_supported_profiles(
    backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    result = ValidationService(get_node_catalog()).validate(_conv1d_graph(), backend, keras_enabled)

    assert result.valid, result.diagnostics

###############################################################################
def test_generated_keras_conv1d_source_is_parseable_and_capability_driven() -> None:
    graph = _conv1d_graph()
    catalog = get_node_catalog()
    ir = compile_to_ir(graph, ExecutionBackend.PYTORCH, True, catalog)
    source = KerasSourceExporter().generate(graph, ir, catalog)

    ast.parse(source)
    assert "layers.Conv1D" in source
    assert "strides=[2]" in source
    assert "dilation_rate=[1]" in source
    assert "groups=1" in source
    assert "data_format='channels_last'" in source

###############################################################################
def test_keras_conv1d_executes_for_the_selected_keras_backend() -> None:
    keras = pytest.importorskip("keras")
    from models.keras_worker import build_model

    if keras.backend.backend() not in {"tensorflow", "torch", "jax"}:
        pytest.skip("The selected Keras backend is not part of the supported profile set")

    model: Any = cast(Any, build_model(_conv1d_graph()))
    value: Any = model({"input": np.zeros((1, 8, 3), dtype=np.float32)}, training=False)
    detached: Any = (
        value.detach().cpu().numpy()
        if callable(getattr(value, "detach", None))
        else np.asarray(value)
    )
    assert detached.shape == (1, 4, 4)

###############################################################################
def test_keras_slice_executes_without_dynamic_code() -> None:
    keras = pytest.importorskip("keras")
    from models.keras_worker import build_model

    if keras.backend.backend() not in {"tensorflow", "torch", "jax"}:
        pytest.skip("The selected Keras backend is not part of the supported profile set")

    model: Any = cast(
        Any,
        build_model(_operation_graph("math.slice", [[4, 3]], {"expression": "x[:, 1:3]"})),
    )
    value: Any = model({"input_0": np.zeros((1, 4, 3), dtype=np.float32)}, training=False)
    detached: Any = (
        value.detach().cpu().numpy()
        if callable(getattr(value, "detach", None))
        else np.asarray(value)
    )
    assert detached.shape == (1, 2, 3)


###############################################################################
@pytest.mark.parametrize(("definition_id", "parameters", "symbols"), _UNARY_OPERATION_CASES)
def test_unary_operation_catalog_source_and_static_contract(
    definition_id: str,
    parameters: dict[str, Any],
    symbols: dict[str, str],
) -> None:
    catalog = get_node_catalog()
    manifest = catalog.get(definition_id)
    assert manifest.category == "mathematical"
    assert manifest.operation is not None
    assert manifest.operation.family == "unary"
    assert manifest.operation.shape_behavior == "preserve"
    assert manifest.operation.arity == 1
    assert manifest.operation.output_dtype == "same"

    for implementation, symbol in symbols.items():
        mapping = manifest.implementations[implementation]
        assert mapping.supported
        assert mapping.status == "verified"
        assert mapping.symbol == symbol
        assert symbol in emit_operation_source(manifest, implementation, ["value"], parameters)

    summary = next(
        item
        for item in infer_shapes(_operation_graph(definition_id, [[4]], parameters), catalog)
        if item.node_id == "operation"
    )
    assert summary.outputs["output"].shape == [None, 4]
    assert summary.outputs["output"].dtype.value == "float32"


###############################################################################
@pytest.mark.parametrize(("definition_id", "parameters", "_symbols"), _UNARY_OPERATION_CASES)
@pytest.mark.parametrize("backend", list(ExecutionBackend))
def test_unary_operations_execute_with_finite_nonzero_inputs_across_native_backends(
    definition_id: str,
    parameters: dict[str, Any],
    _symbols: dict[str, str],
    backend: ExecutionBackend,
) -> None:
    pytest.importorskip(
        {
            ExecutionBackend.TENSORFLOW: "tensorflow",
            ExecutionBackend.PYTORCH: "torch",
            ExecutionBackend.JAX: "jax",
        }[backend]
    )

    handle = build_native_model(
        _operation_graph(definition_id, [[4]], parameters),
        backend,
        input_values=[_native_float_input(backend)],
    )
    outputs = forward_native(handle)
    value = (
        outputs[0].detach().cpu().numpy()
        if backend is ExecutionBackend.PYTORCH
        else np.asarray(outputs[0])
    )
    assert value.shape == (1, 4)
    assert value.dtype == np.dtype("float32")
    assert np.isfinite(value).all()


###############################################################################
@pytest.mark.parametrize(("definition_id", "symbols"), _ELEMENTWISE_BOOLEAN_CASES)
def test_boolean_elementwise_catalog_and_static_contract(
    definition_id: str,
    symbols: dict[str, str],
) -> None:
    catalog = get_node_catalog()
    manifest = catalog.get(definition_id)
    assert manifest.operation is not None
    assert manifest.operation.family == "elementwise"
    assert manifest.operation.arity == 2
    assert manifest.operation.variadic
    assert manifest.operation.output_dtype == "bool"
    for implementation, symbol in symbols.items():
        mapping = manifest.implementations[implementation]
        assert mapping.supported
        assert mapping.status == "verified"
        assert mapping.symbol == symbol
        assert symbol in emit_operation_source(manifest, implementation, ["left", "right"], {})

    input_dtype = "bool" if definition_id.startswith("math.logical_") else "float32"
    summary = next(
        item
        for item in infer_shapes(
            _operation_graph(
                definition_id,
                [[4], [4]],
                input_dtypes=[input_dtype, input_dtype],
            ),
            catalog,
        )
        if item.node_id == "operation"
    )
    assert summary.outputs["output"].shape == [None, 4]
    assert summary.outputs["output"].dtype.value == "bool"


###############################################################################
@pytest.mark.parametrize("backend", list(ExecutionBackend))
@pytest.mark.parametrize(("definition_id", "_symbols"), _ELEMENTWISE_BOOLEAN_CASES)
def test_boolean_elementwise_operations_execute_across_native_backends(
    definition_id: str,
    _symbols: dict[str, str],
    backend: ExecutionBackend,
) -> None:
    pytest.importorskip(
        {
            ExecutionBackend.TENSORFLOW: "tensorflow",
            ExecutionBackend.PYTORCH: "torch",
            ExecutionBackend.JAX: "jax",
        }[backend]
    )

    input_dtype = "bool" if definition_id.startswith("math.logical_") else "float32"
    if input_dtype == "bool":
        left = [True, True, False, False]
        right = [True, False, True, False]
        expected_by_id = {
            "math.logical_and": [True, False, False, False],
            "math.logical_or": [True, True, True, False],
            "math.logical_xor": [False, True, True, False],
        }
    else:
        left = [0.0, 1.0, 2.0, 3.0]
        right = [0.0, 2.0, 1.0, 4.0]
        expected_by_id = {
            "math.equal": [True, False, False, False],
            "math.not_equal": [False, True, True, True],
            "math.greater": [False, False, True, False],
            "math.greater_equal": [True, False, True, False],
            "math.less": [False, True, False, True],
            "math.less_equal": [True, True, False, True],
        }

    handle = build_native_model(
        _operation_graph(
            definition_id,
            [[4], [4]],
            input_dtypes=[input_dtype, input_dtype],
        ),
        backend,
        input_values=[
            _native_values(backend, [left], input_dtype),
            _native_values(backend, [right], input_dtype),
        ],
    )
    outputs = forward_native(handle)
    value = (
        outputs[0].detach().cpu().numpy()
        if backend is ExecutionBackend.PYTORCH
        else np.asarray(outputs[0])
    )
    assert value.shape == (1, 4)
    assert value.dtype == np.dtype("bool")
    np.testing.assert_array_equal(value, np.asarray([expected_by_id[definition_id]], dtype=bool))


###############################################################################
@pytest.mark.parametrize(
    ("definition_id", "parameters", "symbols", "input_dtype", "expected_shape", "expected_dtype"),
    _REDUCTION_CASES,
)
def test_reduction_catalog_source_and_static_contract(
    definition_id: str,
    parameters: dict[str, Any],
    symbols: dict[str, str],
    input_dtype: str,
    expected_shape: list[int | None],
    expected_dtype: str,
) -> None:
    catalog = get_node_catalog()
    manifest = catalog.get(definition_id)
    assert manifest.operation is not None
    assert manifest.operation.family == "reduction"
    assert manifest.operation.shape_behavior == "reduce"
    assert manifest.operation.arity == 1
    for implementation, symbol in symbols.items():
        mapping = manifest.implementations[implementation]
        assert mapping.supported
        assert mapping.status == "verified"
        assert mapping.symbol == symbol
        assert symbol in emit_operation_source(manifest, implementation, ["value"], parameters)

    summary = next(
        item
        for item in infer_shapes(
            _operation_graph(definition_id, [[2, 3]], parameters, [input_dtype]),
            catalog,
        )
        if item.node_id == "operation"
    )
    assert summary.outputs["output"].shape == expected_shape
    assert summary.outputs["output"].dtype.value == expected_dtype


###############################################################################
@pytest.mark.parametrize("backend", list(ExecutionBackend))
@pytest.mark.parametrize(
    ("definition_id", "parameters", "input_dtype"),
    [(case[0], case[1], case[3]) for case in _REDUCTION_CASES],
)
def test_reductions_execute_with_expected_values_across_native_backends(
    definition_id: str,
    parameters: dict[str, Any],
    input_dtype: str,
    backend: ExecutionBackend,
) -> None:
    pytest.importorskip(
        {
            ExecutionBackend.TENSORFLOW: "tensorflow",
            ExecutionBackend.PYTORCH: "torch",
            ExecutionBackend.JAX: "jax",
        }[backend]
    )

    source_values = (
        [[[True, False, True], [True, True, False]]]
        if input_dtype == "bool"
        else [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    )
    source = np.asarray(source_values, dtype=np.bool_ if input_dtype == "bool" else np.float32)
    if definition_id == "math.argmax":
        expected = np.asarray([[1, 1, 1]], dtype=np.int64)
    elif definition_id in {"math.all", "math.any"}:
        expected = np.asarray(
            [[[True, False, False]] if definition_id == "math.all" else [[True, True, True]]],
            dtype=bool,
        )
    elif definition_id in {"math.sum", "math.reduce"}:
        expected = np.sum(source, axis=1, keepdims=True)
    elif definition_id == "math.mean":
        expected = np.mean(source, axis=1, keepdims=True)
    elif definition_id == "math.min":
        expected = np.min(source, axis=1, keepdims=True)
    elif definition_id == "math.max":
        expected = np.max(source, axis=1, keepdims=True)
    elif definition_id == "math.prod":
        expected = np.prod(source, axis=1, keepdims=True)
    elif definition_id == "math.variance":
        expected = np.var(source, axis=1, keepdims=True)
    elif definition_id == "math.stddev":
        expected = np.std(source, axis=1, keepdims=True)
    else:
        expected = np.log(np.exp(source).sum(axis=1, keepdims=True))

    handle = build_native_model(
        _operation_graph(definition_id, [[2, 3]], parameters, [input_dtype]),
        backend,
        input_values=[_native_values(backend, source_values, input_dtype)],
    )
    outputs = forward_native(handle)
    value = (
        outputs[0].detach().cpu().numpy()
        if backend is ExecutionBackend.PYTORCH
        else np.asarray(outputs[0])
    )
    assert value.shape == expected.shape
    if expected.dtype == np.dtype("bool") or expected.dtype == np.dtype("int64"):
        np.testing.assert_array_equal(value, expected)
    else:
        np.testing.assert_allclose(value, expected, rtol=1e-5, atol=1e-5)


###############################################################################
@pytest.mark.parametrize("backend", list(ExecutionBackend))
def test_native_log_functional_dummy_input_stays_in_a_finite_domain(backend: ExecutionBackend) -> None:
    pytest.importorskip(
        {
            ExecutionBackend.TENSORFLOW: "tensorflow",
            ExecutionBackend.PYTORCH: "torch",
            ExecutionBackend.JAX: "jax",
        }[backend]
    )

    handle = build_native_model(_operation_graph("math.log", [[4]]), backend)
    outputs = forward_native(handle)
    value = (
        outputs[0].detach().cpu().numpy()
        if backend is ExecutionBackend.PYTORCH
        else np.asarray(outputs[0])
    )
    assert np.isfinite(value).all()

###############################################################################
@pytest.mark.parametrize("backend", list(ExecutionBackend))
def test_native_conv1d_executes_with_exact_output_shape(backend: ExecutionBackend) -> None:
    pytest.importorskip(
        {
            ExecutionBackend.TENSORFLOW: "tensorflow",
            ExecutionBackend.PYTORCH: "torch",
            ExecutionBackend.JAX: "jax",
        }[backend]
    )

    handle = build_native_model(_conv1d_graph(), backend)
    outputs = forward_native(handle)

    assert len(outputs) == 1
    value = (
        outputs[0].detach().cpu().numpy()
        if backend is ExecutionBackend.PYTORCH
        else np.asarray(outputs[0])
    )
    assert value.shape == (1, 4, 4)

###############################################################################
@pytest.mark.parametrize(
    ("definition_id", "input_shapes", "parameters", "expected_shape"),
    [
        ("math.sin", [[4]], {}, (1, 4)),
        ("math.abs", [[4]], {}, (1, 4)),
        ("math.power", [[4]], {"exponent": 2}, (1, 4)),
        ("math.softmax", [[2, 3]], {"axis": -1}, (1, 2, 3)),
        ("math.add", [[4], [4]], {}, (1, 4)),
        ("math.argmin", [[2, 3]], {"axis": -1}, (1, 2)),
        ("math.reduce", [[2, 3]], {"operation": "mean", "axis": 1, "keepdims": True}, (1, 1, 3)),
        ("math.cast", [[4]], {"dtype": "int32"}, (1, 4)),
        ("math.transpose", [[2, 3]], {"axes": [0, 2, 1]}, (1, 3, 2)),
        ("math.expand_dims", [[2, 3]], {"axis": 1}, (1, 1, 2, 3)),
        ("math.expression", [[4]], {"expression": "x * 2 + 1"}, (1, 4)),
        ("math.sum", [[4]], {"axis": -1}, (1,)),
        ("math.reshape", [[2, 2]], {"target_shape": [4]}, (1, 4)),
        ("math.concatenate", [[2], [2]], {"axis": -1}, (1, 4)),
        ("math.matmul", [[3, 4], [4, 2]], {}, (1, 3, 2)),
        ("math.slice", [[4, 3]], {"expression": "x[:, 1:3]"}, (1, 2, 3)),
        ("math.index", [[4, 3]], {"expression": "x[:, 1:3]"}, (1, 2, 3)),
    ],
)
@pytest.mark.parametrize("backend", list(ExecutionBackend))
def test_representative_operations_execute_across_native_backends(
    definition_id: str,
    input_shapes: list[list[Any]],
    parameters: dict[str, Any],
    expected_shape: tuple[int, ...],
    backend: ExecutionBackend,
) -> None:
    pytest.importorskip(
        {
            ExecutionBackend.TENSORFLOW: "tensorflow",
            ExecutionBackend.PYTORCH: "torch",
            ExecutionBackend.JAX: "jax",
        }[backend]
    )

    handle = build_native_model(_operation_graph(definition_id, input_shapes, parameters), backend)
    outputs = forward_native(handle)
    value = (
        outputs[0].detach().cpu().numpy()
        if backend is ExecutionBackend.PYTORCH
        else np.asarray(outputs[0])
    )
    assert value.shape == expected_shape

###############################################################################
@pytest.mark.parametrize("backend", list(ExecutionBackend))
def test_argmin_native_runtime_preserves_int64_output_dtype(backend: ExecutionBackend) -> None:
    pytest.importorskip(
        {
            ExecutionBackend.TENSORFLOW: "tensorflow",
            ExecutionBackend.PYTORCH: "torch",
            ExecutionBackend.JAX: "jax",
        }[backend]
    )

    handle = build_native_model(
        _operation_graph("math.argmin", [[2, 3]], {"axis": -1}), backend
    )
    outputs = forward_native(handle)
    value = (
        outputs[0].detach().cpu().numpy()
        if backend is ExecutionBackend.PYTORCH
        else np.asarray(outputs[0])
    )
    assert value.dtype == np.dtype("int64")

###############################################################################
@pytest.mark.parametrize(
    ("definition_id", "input_shapes", "parameters", "expected_shape", "expected_dtype"),
    [
        ("math.argmin", [[2, 3]], {"axis": -1}, [None, 2], "int64"),
        ("math.reduce", [[2, 3]], {"operation": "mean", "axis": 1, "keepdims": True}, [None, 1, 3], "float32"),
        ("math.cast", [[4]], {"dtype": "int32"}, [None, 4], "int32"),
        ("math.transpose", [[2, 3]], {"axes": [0, 2, 1]}, [None, 3, 2], "float32"),
        ("math.expand_dims", [[2, 3]], {"axis": 1}, [None, 1, 2, 3], "float32"),
        ("math.index", [[4, 3]], {"expression": "x[:, 1:3]"}, [None, 2, 3], "float32"),
    ],
)
def test_selected_math_operations_infer_static_contract(
    definition_id: str,
    input_shapes: list[list[Any]],
    parameters: dict[str, Any],
    expected_shape: list[int | None],
    expected_dtype: str,
) -> None:
    summary = next(
        item
        for item in infer_shapes(_operation_graph(definition_id, input_shapes, parameters), get_node_catalog())
        if item.node_id == "operation"
    )

    output = summary.outputs["output"]
    assert output.shape == expected_shape
    assert output.dtype.value == expected_dtype

###############################################################################
def test_operation_catalog_covers_typed_families_and_all_backend_symbols() -> None:
    catalog = get_node_catalog()
    operations = [manifest for manifest in catalog.manifests if manifest.operation is not None]

    operation_families = {
        operation.family
        for manifest in operations
        for operation in [manifest.operation]
        if operation is not None
    }
    assert operation_families == {
        "unary",
        "binary",
        "elementwise",
        "reduction",
        "tensor",
    }
    assert len(operations) >= 70
    for manifest in operations:
        for implementation in ("keras", "tensorflow", "pytorch", "jax"):
            mapping = manifest.implementations[implementation]
            assert mapping.supported
            assert mapping.symbol or mapping.adapter

###############################################################################
def test_operation_emitter_uses_manifest_metadata_and_rejects_unsafe_expression() -> None:
    catalog = get_node_catalog()
    assert (
        emit_operation_source(catalog.get("math.exp"), "pytorch", ["value"], {})
        == "torch.exp(value)"
    )
    assert (
        emit_operation_source(catalog.get("math.add"), "tensorflow", ["left", "right"], {})
        == "tf.add(left, right)"
    )
    assert "keras.ops.cast(keras.ops.argmin" in emit_operation_source(
        catalog.get("math.argmin"), "keras", ["value"], {"axis": -1}
    )
    assert "torch.logsumexp" in emit_operation_source(
        catalog.get("math.logsumexp"), "pytorch", ["value"], {"axis": -1, "keepdims": True}
    )
    assert (
        emit_operation_source(
            catalog.get("math.slice"), "pytorch", ["value"], {"expression": "x[:, 1:3]"}
        )
        == "value[:, 1:3]"
    )

    with pytest.raises(SafeExpressionError):
        validate_safe_expression("__import__('os').system('whoami')")
    with pytest.raises(SafeExpressionError):
        validate_safe_expression("x[__import__('os')]")
    with pytest.raises(SafeExpressionError):
        emit_operation_source(
            catalog.get("math.expression"),
            "pytorch",
            ["value"],
            {"expression": "__import__('os')"},
        )

###############################################################################
def test_operation_manifest_validates_call_parameters_and_backend_symbols() -> None:
    with pytest.raises(ValueError, match="unknown parameters"):
        NodeManifest(
            id="test.operation",
            display_name="Test operation",
            category="mathematical",
            description="Test operation.",
            inputs=[PortDefinition(id="input", display_name="Input")],
            outputs=[PortDefinition(id="output", display_name="Output")],
            parameters=[],
            shape_rule=ShapeRule(kind="operation"),
            operation=OperationMetadata(family="unary", arguments=["missing"]),
            implementations={"tensorflow": ImplementationMapping(symbol="tf.math.sin")},
            documentation_ref="test",
        )

    with pytest.raises(ValueError, match="validated backend symbol"):
        NodeManifest(
            id="test.operation",
            display_name="Test operation",
            category="mathematical",
            description="Test operation.",
            inputs=[PortDefinition(id="input", display_name="Input")],
            outputs=[PortDefinition(id="output", display_name="Output")],
            parameters=[ParameterDefinition(id="axis", display_name="Axis", type="axis")],
            shape_rule=ShapeRule(kind="operation"),
            operation=OperationMetadata(family="unary", keywords=["axis"]),
            implementations={"tensorflow": ImplementationMapping(symbol="torch.sin")},
            documentation_ref="test",
        )

###############################################################################
def test_compilation_catalog_provider_is_the_shared_metrics_boundary() -> None:
    provider = get_compilation_catalog_provider()
    catalog = get_compilation_catalog()

    assert provider.get() is catalog
    assert {option.id for option in catalog.metrics}
    assert {option.id for option in catalog.losses}
