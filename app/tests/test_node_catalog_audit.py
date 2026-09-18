from __future__ import annotations

from common.catalogs.loader import get_node_catalog
from compilation.compiler import compile_to_ir
from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from exporters.keras_exporter import KerasSourceExporter


###############################################################################
def test_catalog_contract_is_unique_and_round_trips() -> None:
    catalog = get_node_catalog()

    assert len(catalog.manifests) == 149
    assert "output.detection_placeholder" not in {manifest.id for manifest in catalog.manifests}
    for manifest in catalog.manifests:
        assert len({port.id for port in manifest.inputs}) == len(manifest.inputs)
        assert len({port.id for port in manifest.outputs}) == len(manifest.outputs)
        assert len({parameter.id for parameter in manifest.parameters}) == len(
            manifest.parameters
        )
        assert manifest.__class__.model_validate(manifest.model_dump()) == manifest


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
def test_formerly_universal_gaps_are_explicitly_implemented() -> None:
    catalog = get_node_catalog()
    definition_ids = {
        "input.constant",
        "input.parameter",
        "math.index",
        "merge.average",
        "merge.split",
        "output.detection",
    }

    for definition_id in definition_ids:
        manifest = catalog.get(definition_id)
        for implementation in ("keras", "tensorflow", "pytorch", "jax"):
            mapping = manifest.implementations[implementation]
            assert mapping.supported, f"{definition_id} is not supported by {implementation}"
            assert mapping.status.value == "verified"
            assert mapping.symbol or mapping.adapter


def _generated_source_graph(definition_id: str, parameters: dict[str, object]) -> ModelGraph:
    return ModelGraph(
        nodes=[
            NodeInstance(
                id="input",
                definition_id="input.tensor",
                position=Position(x=0, y=0),
                parameters={"shape": [4], "dtype": "float32"},
            ),
            NodeInstance(
                id="generated",
                definition_id=definition_id,
                position=Position(x=180, y=120),
                parameters=parameters,
            ),
            NodeInstance(
                id="add",
                definition_id="merge.add",
                position=Position(x=360, y=0),
            ),
            NodeInstance(
                id="output",
                definition_id="output.custom",
                position=Position(x=540, y=0),
            ),
        ],
        edges=[
            EdgeInstance(
                id="input-add",
                source_node_id="input",
                source_port_id="output",
                target_node_id="add",
                target_port_id="inputs",
            ),
            EdgeInstance(
                id="generated-add",
                source_node_id="generated",
                source_port_id="output",
                target_node_id="add",
                target_port_id="inputs",
            ),
            EdgeInstance(
                id="add-output",
                source_node_id="add",
                source_port_id="output",
                target_node_id="output",
                target_port_id="input",
            ),
        ],
    )


###############################################################################
def test_generated_sources_are_not_exported_as_model_inputs() -> None:
    catalog = get_node_catalog()
    exporter = KerasSourceExporter()
    cases = [
        ("input.constant", {"shape": [4], "value": 1.0, "dtype": "float32"}, "GeneratedConstant"),
        (
            "input.parameter",
            {"shape": [4], "initializer": "ones", "dtype": "float32", "trainable": True},
            "GeneratedParameter",
        ),
    ]

    for definition_id, parameters, expected_class in cases:
        graph = _generated_source_graph(definition_id, parameters)
        ir = compile_to_ir(graph, ExecutionBackend.PYTORCH, True, catalog, shape_summaries=[])
        assert ir.input_node_ids == ["input"]
        source = exporter.generate(graph, ir, catalog)
        assert expected_class in source
        input_lines = [line for line in source.splitlines() if "keras.Input(" in line]
        assert len(input_lines) == 1
        assert "name='input'" in input_lines[0]
