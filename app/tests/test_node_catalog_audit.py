from __future__ import annotations

import pytest
from common.catalogs import loader
from common.catalogs.loader import get_node_catalog
from compilation.compiler import compile_to_ir
from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from exporters.keras_exporter import KerasSourceExporter
from exporters.node_emitters import registry_for_backend


###############################################################################
def test_catalog_contract_is_unique_and_round_trips() -> None:
    catalog = get_node_catalog()

    assert len(catalog.manifests) >= 145
    for manifest in catalog.manifests:
        assert len({port.id for port in manifest.inputs}) == len(manifest.inputs)
        assert len({port.id for port in manifest.outputs}) == len(manifest.outputs)
        assert len({parameter.id for parameter in manifest.parameters}) == len(
            manifest.parameters
        )
        assert manifest.__class__.model_validate(manifest.model_dump()) == manifest


def test_declarative_parameter_groups_expand_and_reject_group_collisions() -> None:
    catalog = get_node_catalog()
    dense = catalog.get("layer.dense")

    assert dense.parameter_groups == ["base_layer", "kernel_layer"]
    assert {parameter.id for parameter in dense.parameters} >= {
        "dtype",
        "kernel_initializer",
        "units",
    }

    with pytest.raises(RuntimeError, match="Parameter group collision"):
        loader._expand_parameter_groups(  # pyright: ignore[reportPrivateUsage]
            {"id": "test", "parameter_groups": ["first", "second"]},
            {
                "first": [{"id": "duplicate"}],
                "second": [{"id": "duplicate"}],
            },
        )


###############################################################################
def test_supported_keras_mappings_are_explicit_and_non_legacy() -> None:
    catalog = get_node_catalog()

    for manifest in catalog.manifests:
        mapping = manifest.implementations.get("keras")
        if mapping is None or not mapping.supported:
            continue
        assert mapping.symbol or mapping.adapter
        assert "legacy" not in (mapping.adapter or "").lower()


###############################################################################
def test_keras_layer_surface_matches_current_constructor_contract() -> None:
    catalog = get_node_catalog()

    expected_parameters = {
        "layer.dense": {
            "units",
            "activation",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "activity_regularizer",
            "kernel_constraint",
            "bias_constraint",
            "lora_rank",
            "lora_alpha",
            "quantization_config",
            "dtype",
            "trainable",
            "autocast",
        },
        "layer.dropout": {"rate", "noise_shape", "seed", "dtype", "trainable", "autocast"},
        "layer.batch_norm": {
            "axis",
            "momentum",
            "epsilon",
            "center",
            "scale",
            "beta_initializer",
            "gamma_initializer",
            "moving_mean_initializer",
            "moving_variance_initializer",
            "beta_regularizer",
            "gamma_regularizer",
            "beta_constraint",
            "gamma_constraint",
            "renorm",
            "renorm_clipping",
            "renorm_momentum",
            "synchronized",
            "dtype",
            "trainable",
            "autocast",
        },
        "conv.conv3d": {
            "filters",
            "kernel_size",
            "strides",
            "padding",
            "data_format",
            "dilation_rate",
            "groups",
            "activation",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "activity_regularizer",
            "kernel_constraint",
            "bias_constraint",
            "dtype",
            "trainable",
            "autocast",
        },
        "attention.multi_head": {
            "num_heads",
            "key_dim",
            "layers",
            "value_dim",
            "dropout",
            "use_bias",
            "output_shape",
            "attention_axes",
            "sliding_window",
            "flash_attention",
            "use_gate",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "activity_regularizer",
            "kernel_constraint",
            "bias_constraint",
            "seed",
            "dtype",
            "trainable",
            "autocast",
        },
    }

    for node_id, expected in expected_parameters.items():
        manifest = catalog.get(node_id)
        assert {parameter.id for parameter in manifest.parameters} == expected

    padding = next(
        parameter
        for parameter in catalog.get("conv.conv1d").parameters
        if parameter.id == "padding"
    )
    assert padding.default == "valid"


###############################################################################
def test_every_supported_keras_node_has_a_source_adapter() -> None:
    catalog = get_node_catalog()
    exporter = KerasSourceExporter()
    for manifest in catalog.manifests:
        mapping = manifest.implementations.get("keras")
        if mapping is None or not mapping.supported or manifest.semantic_role == "input":
            continue
        parameters = {
            parameter.id: parameter.default
            for parameter in manifest.parameters
            if parameter.default is not None
        }
        node = NodeInstance(
            id="node",
            definition_id=manifest.id,
            definition_version=manifest.version,
            position=Position(x=240, y=0),
            parameters=parameters,
        )
        input_count = 2 if manifest.inputs and manifest.inputs[0].accepts_multiple else len(manifest.inputs)
        input_nodes = [
            NodeInstance(
                id=f"input_{index + 1}",
                definition_id="input.tensor",
                position=Position(x=0, y=index * 160),
                parameters={"shape": [8, 8, 3], "dtype": "float32"},
            )
            for index in range(input_count)
        ]
        input_ports = (
            [manifest.inputs[0].id] * input_count
            if manifest.inputs and manifest.inputs[0].accepts_multiple
            else [port.id for port in manifest.inputs]
        )
        graph = ModelGraph(
            nodes=[*input_nodes, node],
            edges=[
                EdgeInstance(
                    id=f"edge_{index + 1}",
                    source_node_id=input_node.id,
                    source_port_id="output",
                    target_node_id="node",
                    target_port_id=target_port,
                )
                for index, (input_node, target_port) in enumerate(
                    zip(input_nodes, input_ports, strict=True)
                )
            ],
        )
        ir = compile_to_ir(
            graph,
            ExecutionBackend.PYTORCH,
            True,
            catalog,
            shape_summaries=[],
        )
        source = exporter.generate(graph, ir, catalog)
        assert "No Keras adapter registered" not in source


###############################################################################
def test_native_shape_layer_mappings_are_declarative() -> None:
    catalog = get_node_catalog()
    expected = {
        "layer.reshape": {"tensorflow": "reshape", "pytorch": "reshape", "jax": "reshape"},
        "layer.permute": {"tensorflow": "permute", "pytorch": "permute", "jax": "permute"},
        "layer.repeat": {"tensorflow": "repeat", "pytorch": "repeat", "jax": "repeat"},
    }

    for node_id, adapters in expected.items():
        manifest = catalog.get(node_id)
        for backend, adapter in adapters.items():
            mapping = manifest.implementations[backend]
            assert mapping.supported
            assert mapping.adapter == adapter
            assert mapping.symbol
            assert registry_for_backend(backend).has(adapter)
