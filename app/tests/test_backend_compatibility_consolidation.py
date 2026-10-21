# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import pytest

from common.catalogs.loader import NodeCatalog, get_node_catalog
from domain.enums import ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
from domain.manifests import ParameterCapability
from services.validation_service import ValidationService
from validation.profile_compatibility import resolve_implementation_compatibility

###############################################################################
PROFILES = [
    (ExecutionBackend.TENSORFLOW, True),
    (ExecutionBackend.PYTORCH, True),
    (ExecutionBackend.JAX, True),
    (ExecutionBackend.TENSORFLOW, False),
    (ExecutionBackend.PYTORCH, False),
    (ExecutionBackend.JAX, False),
]

###############################################################################
def make_node(
    node_id: str,
    definition_id: str,
    parameters: dict[str, object] | None = None,
) -> NodeInstance:
    return NodeInstance(
        id=node_id,
        definition_id=definition_id,
        position=Position(x=0, y=0),
        parameters=parameters or {},
    )

###############################################################################
def chain_graph(
    definition_id: str,
    parameters: dict[str, object] | None = None,
    *,
    input_definition: str = "input.tabular",
    input_parameters: dict[str, object] | None = None,
    output_definition: str = "output.regression",
) -> ModelGraph:
    return ModelGraph(
        nodes=[
            make_node("input", input_definition, input_parameters or {"shape": [4], "dtype": "float32"}),
            make_node("operation", definition_id, parameters),
            make_node("output", output_definition),
        ],
        edges=[
            EdgeInstance(
                id="input-operation",
                source_node_id="input",
                source_port_id="output",
                target_node_id="operation",
                target_port_id="input",
            ),
            EdgeInstance(
                id="operation-output",
                source_node_id="operation",
                source_port_id="output",
                target_node_id="output",
                target_port_id="input",
            ),
        ],
    )


###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_common_dense_chain_is_valid_for_all_six_static_profiles(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        chain_graph(
            "layer.dense",
            {"units": 3, "activation": "softmax"},
            output_definition="output.classification",
        ),
        execution_backend,
        keras_enabled,
    )

    assert result.valid, result.diagnostics
    assert not any(
        item.code in {"unsupported_implementation", "unverified_implementation"}
        for item in result.diagnostics
    )

###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_simple_rnn_is_actionable_in_every_profile(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        chain_graph(
            "recurrent.simple_rnn",
            {"units": 4},
            input_definition="input.sequence",
            input_parameters={"shape": [12, 8], "dtype": "float32"},
            output_definition="output.sequence",
        ),
        execution_backend,
        keras_enabled,
    )

    assert result.valid, result.diagnostics

###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_dense_initializer_accepts_keras_but_rejects_non_default_native_values(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        chain_graph(
            "layer.dense",
            {"units": 2, "kernel_initializer": "he_normal"},
        ),
        execution_backend,
        keras_enabled,
    )

    if keras_enabled:
        assert result.valid, result.diagnostics
    else:
        assert not result.valid
        assert any(
            item.code == "unsupported_parameter_value"
            and item.parameter_id == "kernel_initializer"
            for item in result.diagnostics
        )

###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_native_initializer_defaults_remain_valid_while_non_defaults_are_blocked(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        chain_graph(
            "layer.dense",
            {"units": 2, "kernel_initializer": "glorot_uniform"},
        ),
        execution_backend,
        keras_enabled,
    )

    assert result.valid, result.diagnostics

###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_lstm_unit_forget_bias_uses_the_selected_backend_mapping(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    result = ValidationService(get_node_catalog()).validate(
        chain_graph(
            "recurrent.lstm",
            {"units": 4, "unit_forget_bias": False},
            input_definition="input.sequence",
            input_parameters={"shape": [12, 8], "dtype": "float32"},
            output_definition="output.sequence",
        ),
        execution_backend,
        keras_enabled,
    )

    expected_valid = keras_enabled or execution_backend is ExecutionBackend.PYTORCH
    assert result.valid is expected_valid, result.diagnostics
    if not expected_valid:
        assert any(
            item.code == "unsupported_parameter_option"
            and item.parameter_id == "unit_forget_bias"
            for item in result.diagnostics
        )

###############################################################################
def test_keras_backend_exclusion_has_a_distinct_reason_code() -> None:
    base_catalog = get_node_catalog()
    base_manifest = base_catalog.get("layer.dense")
    excluded_mapping = base_manifest.implementations["keras"].model_copy(
        update={
            "execution_backends": {
                ExecutionBackend.TENSORFLOW: False,
                ExecutionBackend.PYTORCH: True,
                ExecutionBackend.JAX: True,
            }
        }
    )
    implementations = dict(base_manifest.implementations)
    implementations["keras"] = excluded_mapping
    excluded_manifest = base_manifest.model_copy(
        deep=True,
        update={
            "id": "test.keras_excluded_dense",
            "display_name": "Keras excluded Dense",
            "implementations": implementations,
        },
    )
    catalog = NodeCatalog([*base_catalog.manifests, excluded_manifest])

    result = ValidationService(catalog).validate(
        chain_graph("test.keras_excluded_dense"),
        ExecutionBackend.TENSORFLOW,
        True,
    )

    diagnostic = next(item for item in result.diagnostics if item.node_id == "operation")
    assert not result.valid
    assert diagnostic.details["reason_code"] == "backend_excluded"
    assert "Keras" in diagnostic.message

###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_synthetic_unverified_node_mapping_is_unavailable(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    base_catalog = get_node_catalog()
    base_manifest = base_catalog.get("layer.dense")
    implementation_key = "keras" if keras_enabled else execution_backend.value
    implementations = dict(base_manifest.implementations)
    implementations[implementation_key] = implementations[implementation_key].model_copy(
        update={"status": "unverified"}
    )
    unverified_manifest = base_manifest.model_copy(
        deep=True,
        update={
            "id": "test.unverified_dense",
            "display_name": "Unverified Dense",
            "implementations": implementations,
        },
    )
    catalog = NodeCatalog([*base_catalog.manifests, unverified_manifest])

    result = ValidationService(catalog).validate(
        chain_graph("test.unverified_dense"),
        execution_backend,
        keras_enabled,
    )

    diagnostic = next(item for item in result.diagnostics if item.node_id == "operation")
    assert not result.valid
    assert diagnostic.code == "unverified_implementation"
    assert diagnostic.details["reason_code"] == "unverified_implementation"

###############################################################################
@pytest.mark.parametrize(("execution_backend", "keras_enabled"), PROFILES)
def test_synthetic_unverified_parameter_mapping_is_unavailable(
    execution_backend: ExecutionBackend,
    keras_enabled: bool,
) -> None:
    base_catalog = get_node_catalog()
    base_manifest = base_catalog.get("layer.dense")
    implementation_key = "keras" if keras_enabled else execution_backend.value
    native_mapping = base_manifest.implementations[implementation_key]
    parameters = dict(native_mapping.parameters)
    parameters["units"] = ParameterCapability(supported=True, status="unverified")
    implementations = dict(base_manifest.implementations)
    implementations[implementation_key] = native_mapping.model_copy(
        update={"parameters": parameters}
    )
    unverified_manifest = base_manifest.model_copy(
        deep=True,
        update={
            "id": "test.unverified_dense_parameter",
            "display_name": "Unverified Dense Parameter",
            "implementations": implementations,
        },
    )
    catalog = NodeCatalog([*base_catalog.manifests, unverified_manifest])

    result = ValidationService(catalog).validate(
        chain_graph("test.unverified_dense_parameter", {"units": 2}),
        execution_backend,
        keras_enabled,
    )

    assert not result.valid
    assert any(
        item.code == "unverified_parameter_mapping"
        and item.parameter_id == "units"
        for item in result.diagnostics
    )

###############################################################################
def test_profile_resolver_distinguishes_missing_unsupported_and_unverified_mappings() -> None:
    base = get_node_catalog().get("layer.dense")
    missing = base.model_copy(deep=True, update={"implementations": {}})
    unsupported_mapping = base.implementations["pytorch"].model_copy(
        update={"supported": False}
    )
    unsupported = base.model_copy(
        deep=True,
        update={"implementations": {"pytorch": unsupported_mapping}},
    )
    unverified_mapping = base.implementations["pytorch"].model_copy(
        update={"status": "unverified"}
    )
    unverified = base.model_copy(
        deep=True,
        update={"implementations": {"pytorch": unverified_mapping}},
    )

    assert resolve_implementation_compatibility(missing, ExecutionBackend.PYTORCH, False).reason_code == "missing_implementation"
    assert resolve_implementation_compatibility(unsupported, ExecutionBackend.PYTORCH, False).reason_code == "unsupported_implementation"
    assert resolve_implementation_compatibility(unverified, ExecutionBackend.PYTORCH, False).reason_code == "unverified_implementation"
