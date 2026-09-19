from __future__ import annotations

from common.catalogs.loader import NodeCatalog, get_node_catalog
from common.catalogs.operation_emitter import emit_operation_source
from compilation.compiler import compile_to_ir
from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from exporters.keras_exporter import KerasSourceExporter

NEW_NODE_IDS = {
    "layer.rms_norm",
    "layer.prelu",
    "layer.unit_norm",
    "math.stop_gradient",
}


def _graph_for_node(definition_id: str, catalog: NodeCatalog) -> ModelGraph:
    manifest = catalog.get(definition_id)
    parameters = {
        parameter.id: parameter.default
        for parameter in manifest.parameters
        if parameter.default is not None
    }
    return ModelGraph(
        nodes=[
            NodeInstance(
                id="input",
                definition_id="input.tensor",
                position=Position(x=0, y=0),
                parameters={"shape": [8], "dtype": "float32"},
            ),
            NodeInstance(
                id="node",
                definition_id=definition_id,
                definition_version=manifest.version,
                position=Position(x=240, y=0),
                parameters=parameters,
            ),
        ],
        edges=[
            EdgeInstance(
                id="edge",
                source_node_id="input",
                source_port_id="output",
                target_node_id="node",
                target_port_id="input",
            )
        ],
    )


def _keras_source(definition_id: str) -> str:
    catalog = get_node_catalog()
    graph = _graph_for_node(definition_id, catalog)
    ir = compile_to_ir(
        graph,
        ExecutionBackend.PYTORCH,
        True,
        catalog,
        shape_summaries=[],
    )
    return KerasSourceExporter().generate(graph, ir, catalog)


def test_modern_nodes_are_canonical_catalog_entries() -> None:
    catalog = get_node_catalog()
    manifest_ids = {manifest.id for manifest in catalog.manifests}

    assert manifest_ids >= NEW_NODE_IDS
    assert len(catalog.manifests) >= 149

    for definition_id in NEW_NODE_IDS:
        manifest = catalog.get(definition_id)
        assert manifest.version == 1
        assert manifest.semantic_role == "operation"
        assert manifest.implementations["keras"].supported is True
        assert manifest.implementations["keras"].execution_backends == {
            ExecutionBackend.TENSORFLOW: True,
            ExecutionBackend.PYTORCH: True,
            ExecutionBackend.JAX: True,
        }

    for definition_id in ("layer.rms_norm", "layer.prelu", "layer.unit_norm"):
        manifest = catalog.get(definition_id)
        for native_target in ("tensorflow", "pytorch", "jax"):
            assert manifest.implementations[native_target].supported is True


def test_modern_core_nodes_emit_explicit_keras_layers() -> None:
    expected_layer = {
        "layer.rms_norm": "layers.RMSNormalization(",
        "layer.prelu": "layers.PReLU(",
        "layer.unit_norm": "layers.UnitNormalization(",
    }

    for definition_id, source_fragment in expected_layer.items():
        source = _keras_source(definition_id)
        assert source_fragment in source
        assert "No Keras adapter registered" not in source


def test_stop_gradient_uses_verified_backend_operations() -> None:
    catalog = get_node_catalog()
    manifest = catalog.get("math.stop_gradient")

    assert manifest.operation is not None
    assert manifest.operation.family == "unary"
    assert manifest.operation.shape_behavior == "preserve"
    assert manifest.implementations["keras"].symbol == "keras.ops.stop_gradient"
    assert manifest.implementations["tensorflow"].symbol == "tf.stop_gradient"
    assert manifest.implementations["pytorch"].symbol == "torch.detach"
    assert manifest.implementations["jax"].symbol == "jax.lax.stop_gradient"

    source = _keras_source("math.stop_gradient")
    assert "keras.ops.stop_gradient(" in source

    expected_native_source = {
        "tensorflow": "tf.stop_gradient(value)",
        "pytorch": "torch.detach(value)",
        "jax": "jax.lax.stop_gradient(value)",
    }
    for target, expected in expected_native_source.items():
        assert emit_operation_source(manifest, target, ["value"], {}) == expected
