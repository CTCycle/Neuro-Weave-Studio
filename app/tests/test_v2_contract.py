from __future__ import annotations

import ast
from typing import Any, cast

import httpx

from common.catalogs.loader import get_node_catalog
from domain.enums import CodeTarget, ExecutionBackend


###############################################################################
def test_all_manifests_expose_complete_safe_parameter_contract() -> None:
    catalog = get_node_catalog()
    assert len(catalog.manifests) == 85
    for manifest in catalog.manifests:
        assert set(manifest.frameworks) == set(CodeTarget)
        ids = [item.id for item in manifest.parameters]
        assert len(ids) == len(set(ids))
        for parameter in manifest.parameters:
            assert parameter.description.strip()
            assert parameter.group.strip()
            assert isinstance(parameter.constraints, dict)
            assert isinstance(parameter.options, list)

###############################################################################
def test_manifest_capabilities_are_not_inflated_by_runtime_registry() -> None:
    catalog = get_node_catalog()
    assert catalog.get("layer.batch_norm").frameworks[CodeTarget.PYTORCH].supported
    assert catalog.get("conv.conv2d").frameworks[CodeTarget.PYTORCH].supported
    assert not catalog.get("attention.multi_head").frameworks[CodeTarget.PYTORCH].supported
    assert not catalog.get("attention.multi_head").frameworks[CodeTarget.TENSORFLOW].supported
    assert not catalog.get("transformer.encoder_block").frameworks[CodeTarget.TENSORFLOW].supported
    assert not catalog.get("layer.dense").frameworks[CodeTarget.JAX].supported

###############################################################################
def test_unproven_native_targets_remain_disabled() -> None:
    catalog = get_node_catalog()
    for target in (CodeTarget.TENSORFLOW, CodeTarget.JAX):
        assert all(not manifest.frameworks[target].supported for manifest in catalog.manifests)

###############################################################################
def test_target_parameter_capabilities_block_unmapped_values() -> None:
    from domain.graph import EdgeInstance, ModelGraph, NodeInstance, Position
    from services.validation_service import ValidationService

    catalog = get_node_catalog()

    def node(node_id: str, definition_id: str, parameters: dict[str, object]) -> NodeInstance:
        return NodeInstance(
            id=node_id,
            definition_id=definition_id,
            position=Position(x=0, y=0),
            parameters=parameters,
        )

    def edge(edge_id: str, source: str, target: str) -> EdgeInstance:
        return EdgeInstance(
            id=edge_id,
            source_node_id=source,
            source_port_id="output",
            target_node_id=target,
            target_port_id="input",
        )

    graph = ModelGraph(
        nodes=[
            node("input", "input.tabular", {"shape": [None, 4], "dtype": "float32"}),
            node("dense", "layer.dense", {"units": 2, "kernel_initializer": "he_normal"}),
            node("output", "output.regression", {}),
        ],
        edges=[edge("e1", "input", "dense"), edge("e2", "dense", "output")],
    )
    result = ValidationService(catalog).validate(graph, CodeTarget.PYTORCH)
    assert not result.valid
    assert any(item.code == "unsupported_parameter_value" and item.parameter_id == "kernel_initializer" for item in result.diagnostics)

    graph.nodes[1].parameters = {"units": 2, "input_size": 4}
    result = ValidationService(catalog).validate(graph, CodeTarget.PYTORCH)
    assert not result.valid
    assert any(item.code == "unknown_parameter" and item.parameter_id == "input_size" for item in result.diagnostics)

###############################################################################
def test_high_parameter_keras_layers_expose_safe_constructor_controls() -> None:
    catalog = get_node_catalog()
    expected = {
        "layer.dense": {
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "activity_regularizer",
            "kernel_constraint",
            "bias_constraint",
        },
        "conv.conv2d": {"strides", "dilation_rate", "groups", "kernel_initializer"},
        "recurrent.lstm": {"recurrent_activation", "dropout", "recurrent_dropout", "return_state", "seed"},
        "attention.multi_head": {"value_dim", "output_shape", "attention_axes", "use_bias"},
    }
    for identifier, required in expected.items():
        manifest = catalog.get(identifier)
        parameters = {parameter.id: parameter for parameter in manifest.parameters}
        assert required <= parameters.keys()
        for parameter_id in required:
            parameter = parameters[parameter_id]
            assert parameter.group
            assert parameter.description
            assert isinstance(parameter.constraints, dict)
            if parameter.type == "enum":
                assert parameter.options

###############################################################################
def test_metadata_reports_v2_targets_and_backend_states(client: httpx.Client) -> None:
    response: httpx.Response = client.get("/api/metadata")
    assert response.status_code == 200
    payload: dict[str, Any] = response.json()
    assert payload["version"] == "1.0.0"
    assert payload["schema_version"] == 2
    assert payload["code_targets"] == [item.value for item in CodeTarget]
    backend_items = cast(list[dict[str, Any]], payload["execution_backends"])
    assert [item["id"] for item in backend_items] == [item.value for item in ExecutionBackend]

###############################################################################
def test_project_and_binary_contract_v2(client: httpx.Client) -> None:
    graph: dict[str, Any] = {
        "schema_version": 2,
        "nodes": [],
        "edges": [],
        "model_configuration": {},
        "export_settings": {},
        "future_training_configuration": {},
    }
    created: httpx.Response = client.post(
        "/api/projects", json={"name": "v2", "code_target": "jax", "graph": graph}
    )
    assert created.status_code == 201
    created_payload: dict[str, Any] = created.json()
    assert created_payload["code_target"] == "jax"
    unknown: httpx.Response = client.post(
        "/api/execution/jobs",
        json={
            "operation": {"type": "keras_export", "project_id": created_payload["id"]},
            "worker": {
                "api": "native",
                "backend": "pytorch",
                "device_id": "cpu",
                "memory_policy": {"kind": "default"},
                "precision": "float32",
            },
        },
    )
    assert unknown.status_code == 422
    assert unknown.json()["detail"]["code"] == "model_validation_failed"

###############################################################################
def test_all_code_targets_generate_parseable_source(client: httpx.Client) -> None:
    graph: dict[str, Any] = {
        "schema_version": 2,
        "nodes": [
            {
                "id": "input",
                "definition_id": "input.tabular",
                "definition_version": 1,
                "position": {"x": 0, "y": 0},
                "parameters": {"shape": [None, 8], "dtype": "float32"},
            },
            {
                "id": "dense",
                "definition_id": "layer.dense",
                "definition_version": 1,
                "position": {"x": 200, "y": 0},
                "parameters": {"units": 2, "activation": "linear", "use_bias": True},
            },
            {
                "id": "output",
                "definition_id": "output.regression",
                "definition_version": 1,
                "position": {"x": 400, "y": 0},
                "parameters": {},
            },
        ],
        "edges": [
            {
                "id": "edge-1",
                "source_node_id": "input",
                "source_port_id": "output",
                "target_node_id": "dense",
                "target_port_id": "input",
            },
            {
                "id": "edge-2",
                "source_node_id": "dense",
                "source_port_id": "output",
                "target_node_id": "output",
                "target_port_id": "input",
            },
        ],
        "model_configuration": {},
        "export_settings": {},
        "future_training_configuration": {},
    }
    catalog = get_node_catalog()
    for target in CodeTarget:
        response: httpx.Response = client.post(
            "/api/compilation", json={"graph": graph, "code_target": target.value}
        )
        assert response.status_code == 200, response.text
        payload: dict[str, Any] = response.json()
        if catalog.get("layer.dense").frameworks[target].supported:
            assert payload["valid"] is True
            ast.parse(cast(str, payload["code"]))
        else:
            assert payload["valid"] is False
            assert any(item["code"] == "unsupported_code_target_feature" for item in payload["diagnostics"])
