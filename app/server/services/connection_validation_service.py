from __future__ import annotations

from common.catalogs.loader import NodeCatalog
from domain.diagnostics import (
    ConnectionCandidatesValidationResult,
    ConnectionValidationResult,
    ValidationDiagnostic,
)
from domain.enums import DiagnosticSeverity, ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance
from domain.manifests import NodeManifest
from services.validation_service import ValidationService


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
    def validate_candidates(
        self,
        graph: ModelGraph,
        connections: list[EdgeInstance],
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
    ) -> ConnectionCandidatesValidationResult:
        valid_connection_ids = [
            connection.id
            for connection in connections
            if self.validate_connection(
                graph,
                connection,
                execution_backend,
                keras_enabled,
            ).valid
        ]
        return ConnectionCandidatesValidationResult(
            valid_connection_ids=valid_connection_ids,
        )

    # -------------------------------------------------------------------------
    def _profile_diagnostic(
        self,
        node: NodeInstance,
        edge_id: str,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
    ) -> ValidationDiagnostic | None:
        manifest = self._manifest(node)
        implementation_key = "keras" if keras_enabled else execution_backend.value
        implementation_label = "Keras" if keras_enabled else f"native {execution_backend.value}"
        mapping = manifest.implementations.get(implementation_key)
        backend_excluded = (
            keras_enabled
            and mapping is not None
            and bool(mapping.execution_backends)
            and mapping.execution_backends.get(execution_backend, True) is False
        )

        if mapping is None or not mapping.supported or backend_excluded:
            message = (
                f"{manifest.display_name} does not support Keras on the "
                f"{execution_backend.value} backend"
                if backend_excluded
                else f"{manifest.display_name} has no explicit {implementation_label} adapter"
            )
            return ValidationDiagnostic(
                code="unsupported_implementation",
                severity=DiagnosticSeverity.COMPATIBILITY,
                message=message,
                node_id=node.id,
                edge_id=edge_id,
                execution_backend=execution_backend,
                keras_enabled=keras_enabled,
            )

        if mapping.status == "unverified":
            return ValidationDiagnostic(
                code="unverified_implementation",
                severity=DiagnosticSeverity.COMPATIBILITY,
                message=f"{manifest.display_name} has not been verified for {implementation_label}",
                node_id=node.id,
                edge_id=edge_id,
                execution_backend=execution_backend,
                keras_enabled=keras_enabled,
                details={"notes": mapping.notes},
            )

        return None

    # -------------------------------------------------------------------------
    def _manifest(self, node: NodeInstance) -> NodeManifest:
        return self._catalog.get(node.definition_id, node.definition_version)
