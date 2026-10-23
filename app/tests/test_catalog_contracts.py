# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
from typing import cast

import pytest

from common.catalogs import loader
from common.catalogs.loader import RESOURCE_ROOT, get_node_catalog
from compilation.compiler import compile_to_ir
from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from exporters.keras_exporter import KerasSourceExporter
from exporters.node_emitters import build_catalog_adapter_audit, registry_for_backend
from exporters.node_emitters.contracts import EmissionResult, NodeEmitterRegistry

# Consolidated from test_adapter_registry_integration.py

MANAGED_ADAPTERS = {"constant", "parameter", "average", "split", "detection"}


###############################################################################
@pytest.mark.parametrize("backend", ["keras", "tensorflow", "pytorch", "jax"])
def test_registered_universal_adapters_cover_all_editable_parameters(backend: str) -> None:
    registry_for_backend(backend).validate_catalog(
        get_node_catalog(), adapter_ids=MANAGED_ADAPTERS
    )


###############################################################################
def test_keras_runtime_registry_covers_universal_runtime_adapters() -> None:
    registry = registry_for_backend("keras")

    assert all(
        registry.has_runtime(adapter)
        for adapter in (
            "constant",
            "parameter",
            "index",
            "average",
            "split",
            "detection",
        )
    )


###############################################################################
@pytest.mark.parametrize("backend", ["keras", "tensorflow", "pytorch", "jax"])
def test_source_exporters_use_authoritative_manifest_adapters(backend: str) -> None:
    registry = registry_for_backend(backend)

    assert all(
        registry.has_authoritative(adapter)
        for adapter in ("constant", "parameter", "index", "average", "split", "detection")
    )


###############################################################################
def test_registry_rejects_duplicate_adapter_ids() -> None:
    registry = NodeEmitterRegistry("test")

    registry.register("known", lambda _context: EmissionResult())
    with pytest.raises(ValueError, match="Duplicate node emitter adapter"):
        registry.register("known", lambda _context: EmissionResult())


###############################################################################
def test_registry_reports_unresolved_verified_mapping() -> None:
    registry = NodeEmitterRegistry("keras")

    with pytest.raises(ValueError, match="requires authoritative adapter 'constant'"):
        registry.validate_catalog(get_node_catalog(), adapter_ids={"constant"})


###############################################################################
def test_registry_reports_ignored_editable_parameters() -> None:
    registry = NodeEmitterRegistry("keras")
    registry.register("constant", lambda _context: EmissionResult(), authoritative=True)

    with pytest.raises(ValueError, match="ignores editable parameters"):
        registry.validate_catalog(get_node_catalog(), adapter_ids={"constant"})


###############################################################################
def test_catalog_adapter_audit_keeps_mapping_counts_separate_from_execution_proof() -> None:
    report = build_catalog_adapter_audit(
        get_node_catalog(),
        {
            backend: registry_for_backend(backend)
            for backend in ("keras", "tensorflow", "pytorch", "jax")
        },
    )

    assert report["evidence_basis"] == "declarative_catalog_and_registered_emitters_only"
    assert report["execution_proof"] is False
    assert report["active_manifest_count"] == 149
    for backend in ("keras", "tensorflow", "pytorch", "jax"):
        backend_report = report["backends"][backend]
        assert backend_report["active_manifests"] == 149
        assert backend_report["catalog_mapping_counts"] == {
            "verified": 149,
            "unsupported": 0,
            "unverified": 0,
            "missing": 0,
        }

    native_unresolved = report["backends"]["tensorflow"]["unresolved_adapters"]
    unresolved_ids = {item["adapter"] for item in native_unresolved}
    assert {"conv1d", "conv2d", "conv3d"}.isdisjoint(unresolved_ids)
    assert "depthwise" in unresolved_ids
    assert all(
        not backend_report["ignored_editable_parameters"]
        for backend_report in report["backends"].values()
    )
    assert report["native_to_keras_fallbacks"] == []

# Consolidated from test_node_catalog_audit.py

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

###############################################################################
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
        "layer.group_norm": {"tensorflow": "group_norm", "pytorch": "group_norm", "jax": "group_norm"},
        "layer.prelu": {"tensorflow": "prelu", "pytorch": "prelu", "jax": "prelu"},
        "layer.rms_norm": {"tensorflow": "rms_norm", "pytorch": "rms_norm", "jax": "rms_norm"},
        "layer.unit_norm": {"tensorflow": "unit_norm", "pytorch": "unit_norm", "jax": "unit_norm"},
        "conv.max_pool1d": {"tensorflow": "pool_nd", "pytorch": "pool_nd", "jax": "pool_nd"},
        "conv.max_pool2d": {"tensorflow": "pool_nd", "pytorch": "pool_nd", "jax": "pool_nd"},
        "conv.max_pool3d": {"tensorflow": "pool_nd", "pytorch": "pool_nd", "jax": "pool_nd"},
        "conv.average_pool1d": {"tensorflow": "pool_nd", "pytorch": "pool_nd", "jax": "pool_nd"},
        "conv.average_pool2d": {"tensorflow": "pool_nd", "pytorch": "pool_nd", "jax": "pool_nd"},
        "conv.average_pool3d": {"tensorflow": "pool_nd", "pytorch": "pool_nd", "jax": "pool_nd"},
        "conv.global_pool1d": {"tensorflow": "global_pool_nd", "pytorch": "global_pool_nd", "jax": "global_pool_nd"},
        "conv.global_pool2d": {"tensorflow": "global_pool_nd", "pytorch": "global_pool_nd", "jax": "global_pool_nd"},
        "conv.global_pool3d": {"tensorflow": "global_pool_nd", "pytorch": "global_pool_nd", "jax": "global_pool_nd"},
        "conv.upsampling": {"tensorflow": "upsampling", "pytorch": "upsampling", "jax": "upsampling"},
        "conv.padding": {"tensorflow": "padding", "pytorch": "padding", "jax": "padding"},
        "conv.cropping": {"tensorflow": "cropping", "pytorch": "cropping", "jax": "cropping"},
        "conv.conv3d": {"tensorflow": "conv3d", "pytorch": "conv3d", "jax": "conv3d"},
        "conv.depthwise": {"tensorflow": "depthwise", "pytorch": "depthwise", "jax": "depthwise"},
        "conv.separable": {"tensorflow": "separable", "pytorch": "separable", "jax": "separable"},
        "conv.transposed": {"tensorflow": "transposed", "pytorch": "transposed", "jax": "transposed"},
        "recurrent.simple_rnn": {"tensorflow": "simple_rnn", "pytorch": "simple_rnn", "jax": "simple_rnn"},
        "recurrent.bidirectional": {"tensorflow": "bidirectional", "pytorch": "bidirectional", "jax": "bidirectional"},
        "recurrent.stack": {"tensorflow": "stack", "pytorch": "stack", "jax": "stack"},
        "attention.self": {"tensorflow": "attention", "pytorch": "attention", "jax": "attention"},
        "attention.cross": {"tensorflow": "attention", "pytorch": "attention", "jax": "attention"},
        "attention.multi_head": {"tensorflow": "attention", "pytorch": "attention", "jax": "attention"},
        "transformer.encoder_block": {"tensorflow": "transformer_block", "pytorch": "transformer_block", "jax": "transformer_block"},
        "transformer.decoder_block": {"tensorflow": "transformer_block", "pytorch": "transformer_block", "jax": "transformer_block"},
        "transformer.stack": {"tensorflow": "transformer_block", "pytorch": "transformer_block", "jax": "transformer_block"},
        "transformer.positional_encoding": {"tensorflow": "positional_encoding", "pytorch": "positional_encoding", "jax": "positional_encoding"},
        "transformer.residual": {"tensorflow": "residual", "pytorch": "residual", "jax": "residual"},
    }

    for node_id, adapters in expected.items():
        manifest = catalog.get(node_id)
        for backend, adapter in adapters.items():
            mapping = manifest.implementations[backend]
            assert mapping.supported
            assert mapping.adapter == adapter
            assert mapping.symbol
            assert registry_for_backend(backend).has(adapter)

# Consolidated from test_parameter_surface_parity.py

###############################################################################
def _resource_node_ids() -> set[str]:
    identifiers: set[str] = set()
    for path in RESOURCE_ROOT.glob("*.json"):
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            items: list[object] = cast(list[object], raw)
        elif isinstance(raw, dict):
            items = cast(list[object], cast(dict[str, object], raw)["nodes"])
        else:
            raise TypeError(f"Catalog resource {path} must contain a list or object")
        for item in items:
            if not isinstance(item, dict):
                continue
            mapping = cast(dict[str, object], item)
            identifier = str(mapping.get("id", ""))
            if identifier.startswith(
                ("layer.", "conv.", "recurrent.", "attention.", "transformer.")
            ):
                identifiers.add(identifier)
    return identifiers


###############################################################################
def test_node_parameter_surfaces_are_declarative_and_match_catalog() -> None:
    catalog = get_node_catalog()
    groups, assignments = loader._parameter_group_catalog(  # pyright: ignore[reportPrivateUsage]
        loader._catalog_extensions()  # pyright: ignore[reportPrivateUsage]
    )
    surfaces = loader._parameter_surface_catalog()  # pyright: ignore[reportPrivateUsage]
    expected_nodes = _resource_node_ids()

    assert expected_nodes == set(surfaces)
    for node_id in sorted(expected_nodes):
        manifest = catalog.get(node_id)
        group_ids = {
            str(parameter["id"])
            for group_id in assignments.get(node_id, [])
            for parameter in groups[group_id]
        }
        surface_ids = [str(parameter["id"]) for parameter in surfaces[node_id]]
        manifest_ids = [parameter.id for parameter in manifest.parameters]

        assert len(manifest_ids) == len(set(manifest_ids))
        assert set(manifest_ids) == group_ids | set(surface_ids)
        assert manifest.parameter_groups == assignments.get(node_id, [])


###############################################################################
def test_pytorch_adapter_mappings_are_catalog_declarative() -> None:
    catalog = get_node_catalog()
    expected = {
        "conv.conv1d": "conv1d",
        "conv.conv2d": "conv2d",
        "conv.conv3d": "conv3d",
        "conv.depthwise": "depthwise",
        "conv.separable": "separable",
        "conv.transposed": "transposed",
        "conv.max_pool2d": "pool_nd",
        "layer.activation": "activation",
        "layer.batch_norm": "batch_norm",
        "layer.dense": "dense",
        "layer.dropout": "dropout",
        "layer.embedding": "embedding",
        "layer.flatten": "flatten",
        "layer.layer_norm": "layer_norm",
        "input.image": "identity",
        "input.tensor": "identity",
        "input.mask": "identity",
        "input.sequence": "identity",
        "input.tabular": "identity",
        "input.token": "identity",
        "merge.residual": "add",
        "merge.skip": "add",
        "output.classification": "identity",
        "output.custom": "identity",
        "output.multi_head": "identity",
        "output.regression": "identity",
        "output.segmentation": "identity",
        "output.sequence": "identity",
        "recurrent.gru": "gru",
        "recurrent.lstm": "lstm",
    }

    assert {
        node_id: catalog.get(node_id).implementations["pytorch"].adapter
        for node_id in expected
    } == expected


###############################################################################
def test_parameter_group_expansion_preserves_declarative_metadata() -> None:
    parameter = {
        "id": "scale",
        "display_name": "Scale",
        "type": "number",
        "scope": "runtime",
        "enabled_when": {"enabled": True},
        "value_schema": {"type": "number", "nullable": True},
    }
    expanded = loader._expand_parameter_groups(  # pyright: ignore[reportPrivateUsage]
        {"id": "example.node", "parameter_groups": ["shared"]},
        {"shared": [parameter]},
    )

    assert expanded["parameters"] == [parameter]


###############################################################################
def test_parameter_group_expansion_rejects_group_and_surface_collisions() -> None:
    with pytest.raises(RuntimeError, match="Parameter surface collision"):
        loader._expand_parameter_groups(  # pyright: ignore[reportPrivateUsage]
            {
                "id": "example.node",
                "parameter_groups": ["shared"],
                "parameters": [{"id": "scale", "type": "number"}],
            },
            {
                "shared": [
                    {"id": "scale", "type": "number"},
                ]
            },
        )
