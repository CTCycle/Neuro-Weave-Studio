# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import re
from collections import Counter
from contextlib import suppress
from typing import Any, cast

from common.catalogs.loader import NodeCatalog
from domain.compilation_catalog import CompilationCatalogProvider, get_compilation_catalog_provider
from domain.diagnostics import (
    ConnectionValidationResult,
    InferenceConfidence,
    ModelTypeInference,
    ModelTypeKind,
    OutputReference,
    ValidationDiagnostic,
    ValidationResult,
)
from domain.enums import DiagnosticSeverity, ExecutionBackend
from domain.graph import EdgeInstance, ModelGraph, NodeInstance, ShapeSummary
from domain.manifests import (
    NodeManifest,
    ParameterCapability,
    ParameterDefinition,
    ParameterValueSchema,
)
from domain.ports import materialized_inputs, materialized_outputs
from domain.tensor import TensorSpec
from validation.compile_configuration import validate_model_configuration
from validation.graph_utils import (
    nodes_reaching,
    reachable_nodes,
    topological_order,
    would_create_cycle,
)
from validation.output_resolution import (
    infer_model_type,
    is_effective_output,
    resolved_outputs,
)
from validation.output_resolution import input_nodes as catalog_input_nodes
from validation.output_resolution import source_nodes as catalog_source_nodes
from validation.profile_compatibility import resolve_implementation_compatibility
from validation.safe_expression import SafeExpressionError, validate_safe_expression
from validation.shape_inference import ShapeInferenceError, infer_shapes


###############################################################################
def _parameter_type_matches(parameter: ParameterDefinition, value: Any) -> bool:
    if parameter.value_schema is not None:
        return _value_matches_schema(parameter.value_schema, value)
    if value is None:
        return parameter.nullable
    if parameter.type == "boolean":
        return isinstance(value, bool)
    if parameter.type == "integer" or parameter.type == "axis":
        if parameter.type == "axis" and isinstance(value, list):
            items = cast(list[Any], value)
            return all(isinstance(item, int) and not isinstance(item, bool) for item in items)
        return isinstance(value, int) and not isinstance(value, bool)
    if parameter.type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if parameter.type == "enum":
        return not parameter.options or value in parameter.options
    if parameter.type in {"string", "expression", "dtype", "activation"}:
        return isinstance(value, str)
    if parameter.type in {"initializer", "regularizer", "constraint"}:
        return isinstance(value, (str, dict))
    if parameter.type == "object":
        return isinstance(value, dict)
    if parameter.type == "shape":
        dimensions = cast(list[Any], value) if isinstance(value, list) else []
        return isinstance(value, list) and all(
            dimension is None
            or (isinstance(dimension, int) and not isinstance(dimension, bool))
            or isinstance(dimension, list)
            for dimension in dimensions
        )
    if parameter.type in {"list", "tuple"}:
        return isinstance(value, list)
    return True

###############################################################################
def _value_matches_schema(schema: ParameterValueSchema, value: Any) -> bool:
    if value is None:
        return schema.nullable
    if schema.recursive and isinstance(value, list):
        items = cast(list[object], value)
        return all(_value_matches_schema(schema, item) for item in items)
    if schema.type == "object":
        if not isinstance(value, dict):
            return False
        if any(key not in value for key in schema.required_properties):
            return False
        return all(
            key not in value or _value_matches_schema(property_schema, value[key])
            for key, property_schema in schema.properties.items()
        )
    if schema.type in {"list", "tuple", "shape"}:
        if not isinstance(value, list):
            return False
        if schema.item is None:
            return True
        items = cast(list[Any], value)
        return all(_value_matches_schema(schema.item, item) for item in items)
    if schema.type == "integer" or schema.type == "axis":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema.type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if schema.type == "boolean":
        return isinstance(value, bool)
    if schema.type in {"string", "enum", "expression", "dtype", "activation"}:
        return isinstance(value, str)
    if schema.type in {"initializer", "regularizer", "constraint"}:
        return isinstance(value, (str, dict))
    return True

###############################################################################
def _condition_matches(condition: dict[str, Any], parameters: dict[str, Any]) -> bool:
    for parameter_id, expected in condition.items():
        actual = parameters.get(parameter_id)
        if isinstance(expected, dict):
            if "equals" in expected and actual != expected["equals"]:
                return False
            if "not_equals" in expected and actual == expected["not_equals"]:
                return False
            if "in" in expected and actual not in expected["in"]:
                return False
            if "not_in" in expected and actual in expected["not_in"]:
                return False
        elif isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True

###############################################################################
def _effective_default(
    parameter: ParameterDefinition, capability: ParameterCapability | None
) -> Any:
    if capability is not None and "default" in capability.model_fields_set:
        return capability.default
    if "framework_default" in parameter.model_fields_set:
        return parameter.framework_default
    return parameter.default

###############################################################################
def _graph_subset(graph: ModelGraph, node_ids: set[str]) -> ModelGraph:
    """Build the smallest graph needed to infer a selected node's inputs."""

    return graph.model_copy(
        deep=True,
        update={
            "nodes": [node for node in graph.nodes if node.id in node_ids],
            "edges": [
                edge
                for edge in graph.edges
                if edge.source_node_id in node_ids and edge.target_node_id in node_ids
            ],
            "output_bindings": [],
        },
    )

###############################################################################
def _infer_port_spec(
    graph: ModelGraph,
    catalog: NodeCatalog,
    node_id: str,
    port_id: str,
) -> TensorSpec | None:
    """Infer a port without making unrelated incomplete draft nodes fatal."""

    node_ids = nodes_reaching(graph, {node_id})
    try:
        summaries = infer_shapes(_graph_subset(graph, node_ids), catalog)
    except (ShapeInferenceError, ValueError):
        return None
    summary = next((item for item in summaries if item.node_id == node_id), None)
    return summary.outputs.get(port_id) if summary is not None else None

###############################################################################
def _shape_error_parameter_id(
    manifest: NodeManifest | None,
    message: str,
) -> str | None:
    """Recover the manifest parameter behind a shape-rule failure when known."""

    quoted = re.search(r"parameter\s+['\"]([^'\"]+)['\"]", message, re.IGNORECASE)
    if quoted is not None:
        return quoted.group(1)

    message_lower = message.lower()
    if manifest is not None:
        for parameter in manifest.parameters:
            if parameter.id.lower() in message_lower:
                return parameter.id

    aliases = (
        ("kernel and stride", "kernel_size"),
        ("cropping", "cropping"),
        ("padding", "padding"),
        ("axis", "axis"),
    )
    for phrase, parameter_id in aliases:
        if phrase in message_lower:
            return parameter_id
    return None

###############################################################################
def _shape_error_context(
    graph: ModelGraph,
    catalog: NodeCatalog,
    error: ShapeInferenceError | ValueError,
) -> tuple[str | None, str | None]:
    """Locate the first node that fails so diagnostics can identify its cause."""

    fallback_parameter_id = _shape_error_parameter_id(None, str(error))
    try:
        order = topological_order(graph)
    except ValueError:
        return None, fallback_parameter_id

    nodes = {node.id: node for node in graph.nodes}
    for index, node_id in enumerate(order):
        prefix_ids = set(order[: index + 1])
        try:
            infer_shapes(_graph_subset(graph, prefix_ids), catalog)
        except (ShapeInferenceError, ValueError) as prefix_error:
            manifest = None
            with suppress(KeyError, AttributeError):
                manifest = catalog.get(nodes[node_id].definition_id, nodes[node_id].definition_version)
            return node_id, _shape_error_parameter_id(manifest, str(prefix_error))
    return None, fallback_parameter_id

###############################################################################
def _shape_error_is_pending(
    graph: ModelGraph,
    node: NodeInstance,
    manifest: NodeManifest,
) -> bool:
    """Return whether a target is incomplete rather than shape-incompatible."""

    incoming = [edge for edge in graph.edges if edge.target_node_id == node.id]
    incoming_ports = {edge.target_port_id for edge in incoming}
    if any(
        port.required
        and _condition_matches(port.enabled_when, node.parameters)
        and port.id not in incoming_ports
        for port in materialized_inputs(manifest, node.parameters)
    ):
        return True
    if manifest.operation is not None and len(incoming) < manifest.operation.arity:
        return True
    return manifest.shape_rule.kind in {"merge_same", "concatenate"} and len(incoming) < 2

###############################################################################
class ValidationService:
    """Application-facing graph validation service."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        catalog: NodeCatalog,
        compilation_catalog_provider: CompilationCatalogProvider | None = None,
    ) -> None:
        self._catalog = catalog
        self._compilation_catalog_provider = (
            compilation_catalog_provider or get_compilation_catalog_provider()
        )

    # -------------------------------------------------------------------------
    def validate_connection(
        self,
        graph: ModelGraph,
        connection: EdgeInstance,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
    ) -> ConnectionValidationResult:
        """Validate one proposed edge without requiring a complete draft.

        The Builder must be able to connect nodes before the graph has a model
        input, output, or every required branch.  This focused endpoint keeps
        those draft-level errors out of the gesture while still applying the
        same manifest and shape contracts used by full graph validation.
        """

        diagnostics: list[ValidationDiagnostic] = []
        nodes = {node.id: node for node in graph.nodes}
        manifests: dict[str, NodeManifest] = {}
        for node in graph.nodes:
            try:
                manifests[node.id] = self._catalog.get(node.definition_id, node.definition_version)
            except KeyError:
                continue

        source = nodes.get(connection.source_node_id)
        target = nodes.get(connection.target_node_id)
        if source is None or target is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code="dangling_connection",
                    severity=DiagnosticSeverity.ERROR,
                    message="The proposed connection references a missing node",
                    edge_id=connection.id,
                )
            )
            return ConnectionValidationResult(valid=False, diagnostics=diagnostics)
        if source.id == target.id:
            diagnostics.append(
                ValidationDiagnostic(
                    code="self_connection",
                    severity=DiagnosticSeverity.ERROR,
                    message="A node cannot connect to itself",
                    node_id=source.id,
                    edge_id=connection.id,
                )
            )

        source_manifest = manifests.get(source.id)
        target_manifest = manifests.get(target.id)
        if source_manifest is None or target_manifest is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code="unknown_connection_definition",
                    severity=DiagnosticSeverity.ERROR,
                    message="Both nodes must use known catalog definitions before they can be connected",
                    edge_id=connection.id,
                )
            )
            return ConnectionValidationResult(valid=False, diagnostics=diagnostics)

        source_port = next(
            (
                port
                for port in materialized_outputs(source_manifest, source.parameters)
                if port.id == connection.source_port_id
            ),
            None,
        )
        target_port = next(
            (
                port
                for port in materialized_inputs(target_manifest, target.parameters)
                if port.id == connection.target_port_id
            ),
            None,
        )
        if source_port is None or target_port is None:
            diagnostics.append(
                ValidationDiagnostic(
                    code="invalid_port_reference",
                    severity=DiagnosticSeverity.ERROR,
                    message="The proposed connection must start at an output port and end at an input port",
                    edge_id=connection.id,
                    details={
                        "source_node_id": source.id,
                        "source_port_id": connection.source_port_id,
                        "target_node_id": target.id,
                        "target_port_id": connection.target_port_id,
                    },
                )
            )
            return ConnectionValidationResult(valid=False, diagnostics=diagnostics)

        if not _condition_matches(source_port.enabled_when, source.parameters):
            diagnostics.append(
                ValidationDiagnostic(
                    code="disabled_port",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Output port '{source_port.display_name}' is disabled by the source node parameters",
                    node_id=source.id,
                    edge_id=connection.id,
                    details={"port_id": source_port.id},
                )
            )
        if not _condition_matches(target_port.enabled_when, target.parameters):
            diagnostics.append(
                ValidationDiagnostic(
                    code="disabled_port",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Input port '{target_port.display_name}' is disabled by the target node parameters",
                    node_id=target.id,
                    edge_id=connection.id,
                    details={"port_id": target_port.id},
                )
            )
        expected_kind = (
            target_port.element_kind
            if target_port.kind.value == "tensor_list"
            else target_port.kind
        )
        kind_matches = source_port.kind == expected_kind
        if source_port.kind.value == "tensor_list" and target_port.kind.value == "tensor_list":
            kind_matches = True
        if not kind_matches:
            diagnostics.append(
                ValidationDiagnostic(
                    code="incompatible_port_kind",
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        f"Cannot connect {source_port.display_name} ({source_port.kind.value}) "
                        f"to {target_port.display_name} ({target_port.kind.value})"
                    ),
                    node_id=target.id,
                    edge_id=connection.id,
                    details={
                        "source_kind": source_port.kind.value,
                        "target_kind": target_port.kind.value,
                        "source_port_id": source_port.id,
                        "target_port_id": target_port.id,
                    },
                )
            )
        if any(
            edge.source_node_id == connection.source_node_id
            and edge.source_port_id == connection.source_port_id
            and edge.target_node_id == connection.target_node_id
            and edge.target_port_id == connection.target_port_id
            for edge in graph.edges
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    code="duplicate_connection",
                    severity=DiagnosticSeverity.ERROR,
                    message="This port connection already exists",
                    node_id=target.id,
                    edge_id=connection.id,
                )
            )
        if (
            not target_port.accepts_multiple
            and any(
                edge.target_node_id == connection.target_node_id
                and edge.target_port_id == connection.target_port_id
                for edge in graph.edges
            )
        ):
            diagnostics.append(
                ValidationDiagnostic(
                    code="multiple_connections_not_allowed",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Input port '{target_port.display_name}' accepts only one connection",
                    node_id=target.id,
                    edge_id=connection.id,
                    details={"port_id": target_port.id, "accepts_multiple": False},
                )
            )
        if would_create_cycle(graph, connection.source_node_id, connection.target_node_id):
            diagnostics.append(
                ValidationDiagnostic(
                    code="cycle_detected",
                    severity=DiagnosticSeverity.ERROR,
                    message="Connection blocked because it would create a cycle",
                    node_id=target.id,
                    edge_id=connection.id,
                )
            )

        if diagnostics:
            return ConnectionValidationResult(valid=False, diagnostics=diagnostics)

        candidate = graph.model_copy(deep=True)
        candidate.edges.append(connection)
        source_spec = _infer_port_spec(candidate, self._catalog, source.id, source_port.id)
        if source_spec is not None:
            if source_port.rank is not None and source_spec.rank != source_port.rank:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="port_rank_mismatch",
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"Source port '{source_port.display_name}' expects rank {source_port.rank}, "
                            f"received rank {source_spec.rank}"
                        ),
                        node_id=source.id,
                        edge_id=connection.id,
                        details={
                            "port_id": source_port.id,
                            "expected_rank": source_port.rank,
                            "received_rank": source_spec.rank,
                        },
                    )
                )
            if target_port.rank is not None and source_spec.rank != target_port.rank:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="port_rank_mismatch",
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"Input port '{target_port.display_name}' expects rank {target_port.rank}, "
                            f"received rank {source_spec.rank} from {source.label or source.id}"
                        ),
                        node_id=target.id,
                        edge_id=connection.id,
                        details={
                            "port_id": target_port.id,
                            "expected_rank": target_port.rank,
                            "received_rank": source_spec.rank,
                            "upstream_node_id": source.id,
                            "upstream_port_id": source_port.id,
                            "received_shape": source_spec.shape,
                        },
                    )
                )

        target_node_ids = nodes_reaching(candidate, {target.id})
        target_graph = _graph_subset(candidate, target_node_ids)
        try:
            target_summaries = infer_shapes(target_graph, self._catalog)
        except (ShapeInferenceError, ValueError) as exc:
            if not _shape_error_is_pending(target_graph, target, target_manifest):
                parameter_id = _shape_error_parameter_id(target_manifest, str(exc))
                diagnostics.append(
                    ValidationDiagnostic(
                        code="incompatible_connection_shape",
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            f"Cannot connect to '{target.label or target.id}': {exc}"
                        ),
                        node_id=target.id,
                        edge_id=connection.id,
                        parameter_id=parameter_id,
                        details={
                            "source_node_id": source.id,
                            "source_port_id": source_port.id,
                            "target_port_id": target_port.id,
                            "received_shape": source_spec.shape if source_spec is not None else None,
                            "upstream_node_id": source.id,
                            "reason": str(exc),
                        },
                    )
                )
        else:
            target_shape = next(
                (summary for summary in target_summaries if summary.node_id == target.id),
                None,
            )
            if target_shape is None and materialized_inputs(target_manifest, target.parameters):
                diagnostics.append(
                    ValidationDiagnostic(
                        code="shape_unknown",
                        severity=DiagnosticSeverity.WARNING,
                        message=f"The shape produced by '{target.label or target.id}' is not known yet",
                        node_id=target.id,
                        edge_id=connection.id,
                        details={"upstream_node_id": source.id, "upstream_port_id": source_port.id},
                    )
                )

        blocking_diagnostics = [
            item
            for item in diagnostics
            if item.severity
            in {DiagnosticSeverity.ERROR, DiagnosticSeverity.COMPATIBILITY}
        ]
        return ConnectionValidationResult(
            valid=not blocking_diagnostics,
            diagnostics=diagnostics,
        )

    # -------------------------------------------------------------------------
    def validate(
        self,
        graph: ModelGraph,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
    ) -> ValidationResult:
        diagnostics: list[ValidationDiagnostic] = []
        nodes = {node.id: node for node in graph.nodes}
        manifests: dict[str, NodeManifest] = {}
        implementation_label = "Keras" if keras_enabled else f"native {execution_backend.value}"

        for node in graph.nodes:
            try:
                manifest = self._catalog.get(node.definition_id, node.definition_version)
                manifests[node.id] = manifest
            except KeyError:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="unknown_node_definition",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Unknown node definition {node.definition_id}@{node.definition_version}",
                        node_id=node.id,
                    )
                )
                continue
            compatibility = resolve_implementation_compatibility(
                manifest,
                execution_backend,
                keras_enabled,
            )
            mapping = compatibility.mapping
            if not compatibility.available:
                diagnostic_code = (
                    "unverified_implementation"
                    if compatibility.reason_code == "unverified_implementation"
                    else "unsupported_implementation"
                )
                diagnostics.append(
                    ValidationDiagnostic(
                        code=diagnostic_code,
                        severity=DiagnosticSeverity.COMPATIBILITY,
                        message=compatibility.reason or "The node is unavailable for the selected execution profile.",
                        node_id=node.id,
                        execution_backend=execution_backend,
                        keras_enabled=keras_enabled,
                        details={
                            "reason_code": compatibility.reason_code,
                            "notes": mapping.notes if mapping is not None else None,
                        },
                    )
                )
            known_parameter_ids = {parameter.id for parameter in manifest.parameters}
            unknown_parameters = sorted(set(node.parameters) - known_parameter_ids)
            for parameter_id in unknown_parameters:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="unknown_parameter",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Parameter '{parameter_id}' is not declared by the node manifest",
                        node_id=node.id,
                        parameter_id=parameter_id,
                        execution_backend=execution_backend,
                        keras_enabled=keras_enabled,
                    )
                )
            for parameter in manifest.parameters:
                if not _condition_matches(parameter.visible_when, node.parameters):
                    continue
                supplied = parameter.id in node.parameters
                capability = mapping.parameters.get(parameter.id) if mapping is not None else None
                default = _effective_default(parameter, capability)
                value = node.parameters[parameter.id] if supplied else default
                required = parameter.required or (
                    bool(parameter.required_when)
                    and _condition_matches(parameter.required_when, node.parameters)
                )
                if required and not supplied:
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="missing_parameter",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"Parameter '{parameter.display_name}' is required",
                            node_id=node.id,
                            parameter_id=parameter.id,
                            execution_backend=execution_backend,
                            keras_enabled=keras_enabled,
                        )
                    )
                    continue
                if value is None and not supplied:
                    continue
                if value is None and not parameter.nullable:
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="non_nullable_parameter",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"Parameter '{parameter.display_name}' cannot be null",
                            node_id=node.id,
                            parameter_id=parameter.id,
                            execution_backend=execution_backend,
                            keras_enabled=keras_enabled,
                        )
                    )
                    continue
                if value is None:
                    continue
                if not _parameter_type_matches(parameter, value):
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="invalid_parameter_type",
                            severity=DiagnosticSeverity.ERROR,
                            message=(
                                f"Parameter '{parameter.display_name}' must be "
                                f"of type {parameter.type}"
                            ),
                            node_id=node.id,
                            parameter_id=parameter.id,
                            execution_backend=execution_backend,
                            keras_enabled=keras_enabled,
                        )
                    )
                    continue
                if (
                    capability is not None
                    and (not capability.supported or capability.status == "unsupported")
                    and supplied
                    and value != default
                ):
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="unsupported_parameter_value",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"Parameter '{parameter.display_name}' is not supported for {implementation_label}",
                            node_id=node.id,
                            parameter_id=parameter.id,
                            execution_backend=execution_backend,
                            keras_enabled=keras_enabled,
                            details={"value": value, "notes": capability.notes},
                        )
                    )
                if capability is not None and capability.status == "unverified":
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="unverified_parameter_mapping",
                            severity=DiagnosticSeverity.COMPATIBILITY,
                            message=f"Parameter '{parameter.display_name}' has not been verified for {implementation_label}",
                            node_id=node.id,
                            parameter_id=parameter.id,
                            execution_backend=execution_backend,
                            keras_enabled=keras_enabled,
                            details={"notes": capability.notes},
                        )
                    )
                if (
                    capability is not None
                    and capability.allowed_values is not None
                    and value not in capability.allowed_values
                ):
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="unsupported_parameter_option",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"{parameter.display_name} has no supported {implementation_label} mapping for this value",
                            node_id=node.id,
                            parameter_id=parameter.id,
                            execution_backend=execution_backend,
                            keras_enabled=keras_enabled,
                            details={"allowed": capability.allowed_values, "value": value},
                        )
                    )
                if (
                    parameter.options
                    and not isinstance(value, dict)
                    and value not in parameter.options
                ):
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="invalid_parameter_option",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"{parameter.display_name} has an unsupported value",
                            node_id=node.id,
                            parameter_id=parameter.id,
                            execution_backend=execution_backend,
                            keras_enabled=keras_enabled,
                            details={"allowed": parameter.options, "value": value},
                        )
                    )
                if (
                    supplied
                    and not _condition_matches(parameter.enabled_when, node.parameters)
                    and value != default
                ):
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="disabled_parameter",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"Parameter '{parameter.display_name}' is disabled by its current configuration",
                            node_id=node.id,
                            parameter_id=parameter.id,
                            execution_backend=execution_backend,
                            keras_enabled=keras_enabled,
                        )
                    )
                if parameter.type == "shape" and isinstance(value, list):
                    dimensions = cast(list[Any], value)
                    invalid_dimensions = [
                        dimension
                        for dimension in dimensions
                        if isinstance(dimension, int) and dimension == 0
                    ]
                    if invalid_dimensions:
                        diagnostics.append(
                            ValidationDiagnostic(
                                code="invalid_shape_dimension",
                                severity=DiagnosticSeverity.ERROR,
                                message="Shape dimensions cannot be zero",
                                node_id=node.id,
                                parameter_id=parameter.id,
                                execution_backend=execution_backend,
                                keras_enabled=keras_enabled,
                            )
                        )
                if parameter.type == "expression" and isinstance(value, str):
                    try:
                        validate_safe_expression(value)
                    except SafeExpressionError as exc:
                        diagnostics.append(
                            ValidationDiagnostic(
                                code="unsafe_expression",
                                severity=DiagnosticSeverity.ERROR,
                                message=str(exc),
                                node_id=node.id,
                                parameter_id=parameter.id,
                                execution_backend=execution_backend,
                                keras_enabled=keras_enabled,
                            )
                        )
                constraints = parameter.constraints
                if capability is not None:
                    constraints = {**constraints, **capability.constraints}
                minimum = constraints.get("min")
                maximum = constraints.get("max")
                if (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and isinstance(minimum, int | float)
                    and value < minimum
                ):
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="parameter_below_minimum",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"{parameter.display_name} must be at least {minimum}",
                            node_id=node.id,
                            parameter_id=parameter.id,
                            execution_backend=execution_backend,
                            keras_enabled=keras_enabled,
                        )
                    )
                if (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and isinstance(maximum, int | float)
                    and value > maximum
                ):
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="parameter_above_maximum",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"{parameter.display_name} must be at most {maximum}",
                            node_id=node.id,
                            parameter_id=parameter.id,
                            execution_backend=execution_backend,
                            keras_enabled=keras_enabled,
                        )
                    )
                if isinstance(value, list):
                    items = cast(list[Any], value)
                    expected_length = constraints.get("length")
                    minimum_length = constraints.get("min_length")
                    maximum_length = constraints.get("max_length")
                    if isinstance(expected_length, int) and len(items) != expected_length:
                        diagnostics.append(
                            ValidationDiagnostic(
                                code="parameter_length_invalid",
                                severity=DiagnosticSeverity.ERROR,
                                message=f"{parameter.display_name} must contain exactly {expected_length} values",
                                node_id=node.id,
                                parameter_id=parameter.id,
                                execution_backend=execution_backend,
                                keras_enabled=keras_enabled,
                            )
                        )
                    if isinstance(minimum_length, int) and len(items) < minimum_length:
                        diagnostics.append(
                            ValidationDiagnostic(
                                code="parameter_length_below_minimum",
                                severity=DiagnosticSeverity.ERROR,
                                message=f"{parameter.display_name} must contain at least {minimum_length} values",
                                node_id=node.id,
                                parameter_id=parameter.id,
                                execution_backend=execution_backend,
                                keras_enabled=keras_enabled,
                            )
                        )
                    if isinstance(maximum_length, int) and len(items) > maximum_length:
                        diagnostics.append(
                            ValidationDiagnostic(
                                code="parameter_length_above_maximum",
                                severity=DiagnosticSeverity.ERROR,
                                message=f"{parameter.display_name} must contain at most {maximum_length} values",
                                node_id=node.id,
                                parameter_id=parameter.id,
                                execution_backend=execution_backend,
                                keras_enabled=keras_enabled,
                            )
                        )
                    item_minimum = constraints.get("item_min")
                    item_maximum = constraints.get("item_max")
                    for item in items:
                        if isinstance(item, int | float) and not isinstance(item, bool):
                            if isinstance(item_minimum, int | float) and item < item_minimum:
                                diagnostics.append(
                                    ValidationDiagnostic(
                                        code="parameter_item_below_minimum",
                                        severity=DiagnosticSeverity.ERROR,
                                        message=f"Values in {parameter.display_name} must be at least {item_minimum}",
                                        node_id=node.id,
                                        parameter_id=parameter.id,
                                        execution_backend=execution_backend,
                                        keras_enabled=keras_enabled,
                                    )
                                )
                                break
                            if isinstance(item_maximum, int | float) and item > item_maximum:
                                diagnostics.append(
                                    ValidationDiagnostic(
                                        code="parameter_item_above_maximum",
                                        severity=DiagnosticSeverity.ERROR,
                                        message=f"Values in {parameter.display_name} must be at most {item_maximum}",
                                        node_id=node.id,
                                        parameter_id=parameter.id,
                                        execution_backend=execution_backend,
                                        keras_enabled=keras_enabled,
                                    )
                                )
                                break
        incoming_by_port: Counter[tuple[str, str]] = Counter()
        edge_contracts: Counter[tuple[str, str, str, str]] = Counter()
        for edge in graph.edges:
            source = nodes.get(edge.source_node_id)
            target = nodes.get(edge.target_node_id)
            if source is None or target is None:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="dangling_edge",
                        severity=DiagnosticSeverity.ERROR,
                        message="Edge references a missing node",
                        edge_id=edge.id,
                    )
                )
                continue
            if source.id == target.id:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="self_connection",
                        severity=DiagnosticSeverity.ERROR,
                        message="A node cannot connect to itself",
                        edge_id=edge.id,
                        node_id=source.id,
                    )
                )
            source_manifest = manifests.get(source.id)
            target_manifest = manifests.get(target.id)
            if source_manifest is None or target_manifest is None:
                continue
            source_port = next(
                (
                    port
                    for port in materialized_outputs(source_manifest, source.parameters)
                    if port.id == edge.source_port_id
                ),
                None,
            )
            target_port = next(
                (
                    port
                    for port in materialized_inputs(target_manifest, target.parameters)
                    if port.id == edge.target_port_id
                ),
                None,
            )
            if source_port is None or target_port is None:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="invalid_port_reference",
                        severity=DiagnosticSeverity.ERROR,
                        message="Edge references an unknown source or target port",
                        edge_id=edge.id,
                    )
                )
                continue
            if not _condition_matches(target_port.enabled_when, target.parameters):
                diagnostics.append(
                    ValidationDiagnostic(
                        code="disabled_port",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Input port '{target_port.display_name}' is disabled by node parameters",
                        edge_id=edge.id,
                        node_id=target.id,
                    )
                )
            if not _condition_matches(source_port.enabled_when, source.parameters):
                diagnostics.append(
                    ValidationDiagnostic(
                        code="disabled_port",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Output port '{source_port.display_name}' is disabled by node parameters",
                        edge_id=edge.id,
                        node_id=source.id,
                    )
                )
            expected_kind = (
                target_port.element_kind
                if target_port.kind.value == "tensor_list"
                else target_port.kind
            )
            kind_matches = source_port.kind == expected_kind
            if source_port.kind.value == "tensor_list" and target_port.kind.value == "tensor_list":
                kind_matches = True
            if not kind_matches:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="incompatible_port_kind",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Cannot connect {source_port.kind.value} to {target_port.kind.value}",
                        edge_id=edge.id,
                    )
                )
            contract = (
                edge.source_node_id,
                edge.source_port_id,
                edge.target_node_id,
                edge.target_port_id,
            )
            edge_contracts[contract] += 1
            if edge_contracts[contract] > 1:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="duplicate_connection",
                        severity=DiagnosticSeverity.ERROR,
                        message="The same port connection is declared more than once",
                        edge_id=edge.id,
                    )
                )
            incoming_by_port[(target.id, target_port.id)] += 1
            if (
                not target_port.accepts_multiple
                and incoming_by_port[(target.id, target_port.id)] > 1
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        code="multiple_connections_not_allowed",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Port '{target_port.display_name}' accepts only one connection",
                        node_id=target.id,
                    )
                )

        for node_id, manifest in manifests.items():
            for port in materialized_inputs(manifest, nodes[node_id].parameters):
                if (
                    port.required
                    and _condition_matches(
                        port.enabled_when,
                        nodes[node_id].parameters,
                    )
                    and incoming_by_port[(node_id, port.id)] == 0
                ):
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="missing_required_connection",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"Required input '{port.display_name}' is not connected",
                            node_id=node_id,
                            details={"port_id": port.id},
                        )
                    )

        resolved = resolved_outputs(graph, self._catalog)
        explicit_outputs = bool(graph.output_bindings)
        configuration_outputs = [output for output in resolved if is_effective_output(output)]
        configuration_diagnostics = validate_model_configuration(
            graph,
            execution_backend,
            keras_enabled,
            self._compilation_catalog_provider.get(),
            outputs=configuration_outputs,
        )
        diagnostics.extend(configuration_diagnostics)
        effective_outputs: list[OutputReference] = []
        binding_keys: set[tuple[str, str]] = set()
        output_names: Counter[str] = Counter()
        for output in resolved:
            binding = output.binding
            if output.node is None:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="unknown_output_node",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Output binding references missing node '{binding.node_id}'",
                        node_id=binding.node_id,
                    )
                )
                continue
            if output.manifest is None:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="unknown_output_definition",
                        severity=DiagnosticSeverity.ERROR,
                        message="Output binding references a node with an unknown definition",
                        node_id=binding.node_id,
                    )
                )
                continue
            if output.manifest.semantic_role == "input":
                diagnostics.append(
                    ValidationDiagnostic(
                        code="invalid_model_output",
                        severity=DiagnosticSeverity.ERROR,
                        message="Input nodes cannot be selected as model outputs",
                        node_id=binding.node_id,
                    )
                )
                continue
            if output.port is None:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="invalid_output_port",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Output binding references unknown port '{binding.port_id}'",
                        node_id=binding.node_id,
                    )
                )
                continue
            if not _condition_matches(output.port.enabled_when, output.node.parameters):
                diagnostics.append(
                    ValidationDiagnostic(
                        code="disabled_output_port",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Output port '{output.port.display_name}' is disabled by node parameters",
                        node_id=binding.node_id,
                    )
                )
                continue
            if output.port.kind.value not in {"tensor", "tensor_list", "mask", "scalar"}:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="invalid_output_port_kind",
                        severity=DiagnosticSeverity.ERROR,
                        message="Only value-producing ports can be model outputs",
                        node_id=binding.node_id,
                    )
                )
                continue
            if (binding.node_id, binding.port_id) in binding_keys:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="duplicate_model_output",
                        severity=DiagnosticSeverity.ERROR,
                        message="The same model output is selected more than once",
                        node_id=binding.node_id,
                    )
                )
                continue
            binding_keys.add((binding.node_id, binding.port_id))
            output_names[output.name] += 1
            effective_outputs.append(
                OutputReference(
                    node_id=binding.node_id,
                    port_id=binding.port_id,
                    name=output.name,
                    explicit=output.explicit,
                    task_hint=output.manifest.task_hint,
                )
            )
        if not effective_outputs:
            diagnostics.append(
                ValidationDiagnostic(
                    code="missing_model_output",
                    severity=DiagnosticSeverity.ERROR,
                    message="The model graph requires at least one valid output binding or terminal node",
                )
            )
        duplicate_output_names = [name for name, count in output_names.items() if count > 1]
        if duplicate_output_names:
            diagnostics.append(
                ValidationDiagnostic(
                    code="duplicate_model_output",
                    severity=DiagnosticSeverity.ERROR,
                    message="Model output labels must be unique",
                    details={"labels": duplicate_output_names},
                )
            )
        source_nodes = catalog_source_nodes(graph, self._catalog)
        if not source_nodes:
            diagnostics.append(
                ValidationDiagnostic(
                    code="missing_model_input",
                    severity=DiagnosticSeverity.ERROR,
                    message="The model graph requires at least one source node",
                )
            )

        input_nodes = catalog_input_nodes(graph, self._catalog)
        input_ids = {node.id for node in input_nodes}
        source_ids = {node.id for node in source_nodes}
        output_ids = {item.node_id for item in effective_outputs}
        input_reachable = reachable_nodes(graph, source_ids)
        output_reachable = nodes_reaching(graph, output_ids)
        active_node_ids = input_reachable & output_reachable
        if source_ids and output_ids:
            for node in graph.nodes:
                if node.id in active_node_ids:
                    continue
                if node.id in input_ids:
                    code = "unreachable_model_input"
                    message = "Model input does not contribute to any model output"
                elif node.id in source_ids:
                    code = "unreachable_source_node"
                    message = "Source node does not contribute to any model output"
                elif node.id in output_ids:
                    code = "unreachable_model_output"
                    message = "Model output cannot be reached from a model input"
                else:
                    code = "unreachable_graph_component"
                    message = "Graph node is not on a path from a model input to a model output"
                diagnostics.append(
                    ValidationDiagnostic(
                        code=code,
                        severity=DiagnosticSeverity.ERROR,
                        message=message,
                        node_id=node.id,
                    )
                )

        try:
            topological_order(graph)
        except ValueError:
            diagnostics.append(
                ValidationDiagnostic(
                    code="cycle_detected",
                    severity=DiagnosticSeverity.ERROR,
                    message="The graph must be a directed acyclic graph",
                )
            )

        shape_summaries: list[ShapeSummary] = []
        total_parameters: int | None = 0
        configuration_diagnostic_ids = {id(item) for item in configuration_diagnostics}
        if not any(
            item.severity == DiagnosticSeverity.ERROR
            and id(item) not in configuration_diagnostic_ids
            for item in diagnostics
        ):
            try:
                shape_summaries = infer_shapes(graph, self._catalog)
                shape_by_port = {
                    (summary.node_id, port_id): spec
                    for summary in shape_summaries
                    for port_id, spec in summary.outputs.items()
                }
                for edge in graph.edges:
                    source = nodes.get(edge.source_node_id)
                    target = nodes.get(edge.target_node_id)
                    source_manifest = manifests.get(edge.source_node_id)
                    target_manifest = manifests.get(edge.target_node_id)
                    if (
                        source is None
                        or target is None
                        or source_manifest is None
                        or target_manifest is None
                    ):
                        continue
                    source_port = next(
                        (
                            port
                            for port in materialized_outputs(source_manifest, source.parameters)
                            if port.id == edge.source_port_id
                        ),
                        None,
                    )
                    target_port = next(
                        (
                            port
                            for port in materialized_inputs(target_manifest, target.parameters)
                            if port.id == edge.target_port_id
                        ),
                        None,
                    )
                    source_spec = shape_by_port.get((edge.source_node_id, edge.source_port_id))
                    if source_port is None or target_port is None or source_spec is None:
                        continue
                    for port, label, node_id in (
                        (source_port, "source", edge.source_node_id),
                        (target_port, "target", edge.target_node_id),
                    ):
                        if port.rank is not None and source_spec.rank != port.rank:
                            diagnostics.append(
                                ValidationDiagnostic(
                                    code="port_rank_mismatch",
                                    severity=DiagnosticSeverity.ERROR,
                                    message=(
                                        f"{label.title()} port '{port.display_name}' expects rank "
                                        f"{port.rank}, received rank {source_spec.rank}"
                                    ),
                                    node_id=node_id,
                                    edge_id=edge.id,
                                    details={
                                        "port_id": port.id,
                                        "expected_rank": port.rank,
                                        "received_rank": source_spec.rank,
                                        "received_shape": source_spec.shape,
                                        "upstream_node_id": edge.source_node_id,
                                        "upstream_port_id": edge.source_port_id,
                                    },
                                )
                            )
                known_counts = [item.parameter_count for item in shape_summaries]
                total_parameters = (
                    sum(count for count in known_counts if count is not None)
                    if all(count is not None for count in known_counts)
                    else None
                )
            except (ShapeInferenceError, ValueError) as exc:
                shape_error_node_id, shape_error_parameter_id = _shape_error_context(
                    graph,
                    self._catalog,
                    exc,
                )
                shape_error_node = nodes.get(shape_error_node_id) if shape_error_node_id else None
                shape_error_message = str(exc)
                if shape_error_node is not None:
                    shape_error_message = (
                        f"Shape inference failed for '{shape_error_node.label or shape_error_node.id}': "
                        f"{shape_error_message}"
                    )
                diagnostics.append(
                    ValidationDiagnostic(
                        code="shape_inference_failed",
                        severity=DiagnosticSeverity.ERROR,
                        message=shape_error_message,
                        node_id=shape_error_node_id,
                        parameter_id=shape_error_parameter_id,
                        details={
                            "reason": str(exc),
                            "node_id": shape_error_node_id,
                            "parameter_id": shape_error_parameter_id,
                        },
                    )
                )

        errors = [item for item in diagnostics if item.severity == DiagnosticSeverity.ERROR]
        compatibility_errors = [
            item for item in diagnostics if item.severity == DiagnosticSeverity.COMPATIBILITY
        ]
        inferred_type: ModelTypeKind
        inferred_confidence: InferenceConfidence
        inferred_type, inferred_confidence, inferred_evidence = infer_model_type(
            graph, self._catalog, shape_summaries or None
        )
        return ValidationResult(
            valid=not errors and not compatibility_errors,
            diagnostics=diagnostics,
            shape_summaries=shape_summaries,
            total_parameters=total_parameters,
            effective_outputs=effective_outputs,
            output_mode="explicit" if explicit_outputs else "inferred",
            model_type=ModelTypeInference(
                kind=inferred_type,
                confidence=inferred_confidence,
                evidence=inferred_evidence,
            ),
        )
