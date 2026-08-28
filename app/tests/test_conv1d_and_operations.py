from __future__ import annotations

import ast
from typing import Any, cast

import numpy as np
import pytest

from common.catalogs.loader import get_node_catalog
from common.catalogs.operation_emitter import emit_operation_source
from compilation.compiler import compile_to_ir
from domain.compilation_catalog import get_compilation_catalog, get_compilation_catalog_provider
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
            _node("input", "input.sequence", {"shape": [None, 8, 3], "dtype": "float32"}),
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
) -> ModelGraph:
    input_ids = [f"input_{index}" for index in range(len(input_shapes))]
    nodes = [
        _node(node_id, "input.tensor", {"shape": shape, "dtype": "float32"})
        for node_id, shape in zip(input_ids, input_shapes, strict=True)
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

    summary = next(item for item in infer_shapes(_conv1d_graph(), catalog) if item.node_id == "conv")
    assert summary.outputs["output"].shape == [None, 4, 4]
    assert summary.outputs["output"].dtype.value == "float32"

###############################################################################
@pytest.mark.parametrize("backend", list(ExecutionBackend))
@pytest.mark.parametrize("keras_enabled", [True, False])
def test_conv1d_validation_accepts_all_supported_profiles(
    backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        _conv1d_graph(), backend, keras_enabled
    )

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
    value: Any = model(
        np.zeros((1, 8, 3), dtype=np.float32), training=False
    )
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

    model: Any = cast(Any, build_model(
        _operation_graph("math.slice", [[None, 4, 3]], {"expression": "x[:, 1:3]"})
    ))
    value: Any = model(np.zeros((1, 4, 3), dtype=np.float32), training=False)
    detached: Any = (
        value.detach().cpu().numpy()
        if callable(getattr(value, "detach", None))
        else np.asarray(value)
    )
    assert detached.shape == (1, 2, 3)

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
    value = outputs[0].detach().cpu().numpy() if backend is ExecutionBackend.PYTORCH else np.asarray(outputs[0])
    assert value.shape == (1, 4, 4)

###############################################################################
@pytest.mark.parametrize(
    ("definition_id", "input_shapes", "parameters", "expected_shape"),
    [
        ("math.sin", [[None, 4]], {}, (1, 4)),
        ("math.add", [[None, 4], [None, 4]], {}, (1, 4)),
        ("math.sum", [[None, 4]], {"axis": -1}, (1,)),
        ("math.reshape", [[None, 2, 2]], {"target_shape": [4]}, (1, 4)),
        ("math.concatenate", [[None, 2], [None, 2]], {"axis": -1}, (1, 4)),
        ("math.slice", [[None, 4, 3]], {"expression": "x[:, 1:3]"}, (1, 2, 3)),
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

    handle = build_native_model(
        _operation_graph(definition_id, input_shapes, parameters), backend
    )
    outputs = forward_native(handle)
    value = outputs[0].detach().cpu().numpy() if backend is ExecutionBackend.PYTORCH else np.asarray(outputs[0])
    assert value.shape == expected_shape

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
    assert emit_operation_source(
        catalog.get("math.exp"), "pytorch", ["value"], {}
    ) == "torch.exp(value)"
    assert emit_operation_source(
        catalog.get("math.add"), "tensorflow", ["left", "right"], {}
    ) == "tf.add(left, right)"
    assert "torch.logsumexp" in emit_operation_source(
        catalog.get("math.logsumexp"), "pytorch", ["value"], {"axis": -1, "keepdims": True}
    )
    assert emit_operation_source(
        catalog.get("math.slice"), "pytorch", ["value"], {"expression": "x[:, 1:3]"}
    ) == "value[:, 1:3]"

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
