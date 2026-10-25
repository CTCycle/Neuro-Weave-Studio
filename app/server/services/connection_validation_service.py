# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from common.catalogs.loader import NodeCatalog
from domain.diagnostics import (
    ConnectionValidationResult,
    ValidationDiagnostic,
)
from domain.enums import DiagnosticSeverity, ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance
from domain.manifests import NodeManifest
from services.validation_service import ValidationService
from validation.profile_compatibility import resolve_implementation_compatibility


###############################################################################
class ConnectionValidationService:
    """Validate proposed edges against graph semantics and the active runtime profile."""

    # -------------------------------------------------------------------------
    def __init__(self, catalog: NodeCatalog, validator: ValidationService) -> None:
        self._catalog = catalog
        self._validator = validator

    # -------------------------------------------------------------------------
    def validate_connection(
        self,
        graph: ModelGraph,
        connection: EdgeInstance,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
    ) -> ConnectionValidationResult:
        result = self._validator.validate_connection(
            graph,
            connection,
            execution_backend,
            keras_enabled,
        )
        if not result.valid:
            return result

        nodes = {node.id: node for node in graph.nodes}
        diagnostics: list[ValidationDiagnostic] = []
        for node_id in (connection.source_node_id, connection.target_node_id):
            node = nodes.get(node_id)
            if node is None:
                continue
            diagnostic = self._profile_diagnostic(
                node,
                connection.id,
                execution_backend,
                keras_enabled,
            )
            if diagnostic is not None:
                diagnostics.append(diagnostic)

        if diagnostics:
            return ConnectionValidationResult(valid=False, diagnostics=diagnostics)
        return result

    # -------------------------------------------------------------------------
    def _profile_diagnostic(
        self,
        node: NodeInstance,
        edge_id: str,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
    ) -> ValidationDiagnostic | None:
        manifest = self._manifest(node)
        compatibility = resolve_implementation_compatibility(
            manifest,
            execution_backend,
            keras_enabled,
        )
        if not compatibility.available:
            diagnostic_code = (
                "unverified_implementation"
                if compatibility.reason_code == "unverified_implementation"
                else "unsupported_implementation"
            )
            return ValidationDiagnostic(
                code=diagnostic_code,
                severity=DiagnosticSeverity.COMPATIBILITY,
                message=compatibility.reason or "The node is unavailable for the selected execution profile.",
                node_id=node.id,
                edge_id=edge_id,
                execution_backend=execution_backend,
                keras_enabled=keras_enabled,
                details={
                    "reason_code": compatibility.reason_code,
                    "notes": compatibility.mapping.notes if compatibility.mapping is not None else None,
                },
            )

        return None

    # -------------------------------------------------------------------------
    def _manifest(self, node: NodeInstance) -> NodeManifest:
        return self._catalog.get(node.definition_id, node.definition_version)
