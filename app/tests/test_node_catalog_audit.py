from __future__ import annotations

from common.catalogs.loader import get_node_catalog


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


def test_supported_keras_mappings_are_explicit_and_non_legacy() -> None:
    catalog = get_node_catalog()

    for manifest in catalog.manifests:
        mapping = manifest.implementations.get("keras")
        if mapping is None or not mapping.supported:
            continue
        assert mapping.symbol or mapping.adapter
        assert "legacy" not in (mapping.adapter or "").lower()


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
