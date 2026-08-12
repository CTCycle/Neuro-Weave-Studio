from __future__ import annotations

from typing import Any

from app import create_app

EXPECTED_SCHEMAS = {
    "HealthResponse",
    "MetadataResponse",
    "NodeCatalogResponse",
    "NodeManifest",
    "ProjectCreate",
    "ProjectDuplicateRequest",
    "ProjectListResponse",
    "ProjectDocument",
    "ProjectExportBundle",
    "ProjectUpdate",
    "ValidationRequest",
    "ValidationResult",
    "CompilationRequest",
    "CompilationResult",
    "SourceExportResponse",
    "KerasBinaryExportRequest",
    "SmokeTestRequest",
    "SmokeTestResponse",
}

###############################################################################
def _response_schema(operation: dict[str, Any], status: str = "200") -> dict[str, Any]:
    response = operation["responses"][status]
    return response["content"]["application/json"]["schema"]

###############################################################################
def test_openapi_uses_named_schemas_for_stable_contracts() -> None:
    document = create_app().openapi()
    schemas = document["components"]["schemas"]
    assert schemas.keys() >= EXPECTED_SCHEMAS

    paths = document["paths"]
    assert _response_schema(paths["/health"]["get"]) == {
        "$ref": "#/components/schemas/HealthResponse"
    }
    assert _response_schema(paths["/api/metadata"]["get"]) == {
        "$ref": "#/components/schemas/MetadataResponse"
    }
    assert _response_schema(paths["/api/nodes/catalog"]["get"]) == {
        "$ref": "#/components/schemas/NodeCatalogResponse"
    }
    assert "/api/nodes/{definition_id}/{version}" not in paths
    assert _response_schema(paths["/api/projects"]["get"]) == {
        "$ref": "#/components/schemas/ProjectListResponse"
    }
    assert _response_schema(paths["/api/validation"]["post"]) == {
        "$ref": "#/components/schemas/ValidationResult"
    }
    assert _response_schema(paths["/api/compilation"]["post"]) == {
        "$ref": "#/components/schemas/CompilationResult"
    }
    assert _response_schema(paths["/api/exports/{project_id}/source/{code_target}"]["get"]) == {
        "$ref": "#/components/schemas/SourceExportResponse"
    }

    assert paths["/api/projects"]["post"]["requestBody"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ProjectCreate"}
    assert paths["/api/projects/{project_id}"]["put"]["requestBody"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/ProjectUpdate"}
    assert paths["/api/projects/{project_id}/duplicate"]["post"]["requestBody"][
        "content"
    ]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ProjectDuplicateRequest"
    }
    assert paths["/api/projects/import"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/ProjectExportBundle"}

    validation_body = paths["/api/validation"]["post"]["requestBody"]
    assert validation_body["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ValidationRequest"
    }
    compilation_body = paths["/api/compilation"]["post"]["requestBody"]
    assert compilation_body["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/CompilationRequest"
    }
    binary_body = paths["/api/exports/{project_id}/keras-model"]["post"]["requestBody"]
    assert binary_body["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/KerasBinaryExportRequest"
    }
    smoke_body = paths["/api/execution/smoke-test"]["post"]
    assert _response_schema(smoke_body) == {"$ref": "#/components/schemas/SmokeTestResponse"}
    assert smoke_body["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SmokeTestRequest"
    }
