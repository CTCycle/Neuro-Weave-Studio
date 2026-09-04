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
        mapping = manifest.implementations.get(implementation_key)
        reason: str | None = None

        if mapping is None or not mapping.supported or mapping.status == "unsupported":
            reason = f"{manifest.display_name} is not supported by the selected execution profile."
        elif mapping.status == "unverified":
            reason = f"{manifest.display_name} has not been verified for the selected execution profile."
        elif keras_enabled and not mapping.execution_backends.get(execution_backend, True):
            reason = (
                f"{manifest.display_name} is not supported by the Keras API on "
                f"the {execution_backend.value} backend."
            )

        if reason is None:
            return None

        return ValidationDiagnostic(
            code="connection_profile_incompatible",
            severity=DiagnosticSeverity.COMPATIBILITY,
            message=reason,
            node_id=node.id,
            edge_id=edge_id,
            execution_backend=execution_backend,
            keras_enabled=keras_enabled,
            details={
                "definition_id": node.definition_id,
                "definition_version": node.definition_version,
                "implementation": implementation_key,
            },
        )

    # -------------------------------------------------------------------------
    def _manifest(self, node: NodeInstance) -> NodeManifest:
        return self._catalog.get(node.definition_id, node.definition_version)
