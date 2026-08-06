from __future__ import annotations

from collections import Counter
from typing import Any, cast

from common.catalogs.loader import NodeCatalog
from compilation.adapters import adapter_supported
from domain.diagnostics import ValidationDiagnostic, ValidationResult
from domain.enums import CodeTarget, DiagnosticSeverity
from domain.graph import ModelGraph
from domain.manifests import NodeManifest, ParameterDefinition
from validation.graph_utils import topological_order
from validation.safe_expression import SafeExpressionError, validate_safe_expression
from validation.shape_inference import ShapeInferenceError, infer_shapes


###############################################################################
def _parameter_type_matches(parameter: ParameterDefinition, value: Any) -> bool:
    if value is None:
        return not parameter.required
    if parameter.type == "boolean":
        return isinstance(value, bool)
    if parameter.type == "integer" or parameter.type == "axis":
        return isinstance(value, int) and not isinstance(value, bool)
    if parameter.type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if parameter.type in {"string", "enum", "expression"}:
        return isinstance(value, str)
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
def _output_name(node: Any) -> str:
    # Layer names are owned by the user-facing node label.  The legacy
    # ``name`` parameter remains in some checked-in manifests for import
    # compatibility, but it is intentionally not an independent control.
    return node.label or node.id


###############################################################################
class ValidationService:
    """Application-facing graph validation service."""

    # -------------------------------------------------------------------------
    def __init__(self, catalog: NodeCatalog) -> None:
        self._catalog = catalog

    # -------------------------------------------------------------------------
    def validate(self, graph: ModelGraph, code_target: CodeTarget) -> ValidationResult:
        diagnostics: list[ValidationDiagnostic] = []
        nodes = {node.id: node for node in graph.nodes}
        manifests: dict[str, NodeManifest] = {}

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
            mapping = manifest.frameworks.get(code_target)
            if mapping is None or not mapping.supported or not adapter_supported(code_target, node.definition_id):
                diagnostics.append(
                    ValidationDiagnostic(
                        code="unsupported_code_target_feature",
                        severity=DiagnosticSeverity.COMPATIBILITY,
                        message=f"{manifest.display_name} has no explicit {code_target.value} adapter",
                        node_id=node.id,
                        code_target=code_target,
                    )
                )
            known_parameter_ids = {parameter.id for parameter in manifest.parameters}
            unknown_parameters = sorted(set(node.parameters) - known_parameter_ids)
            if unknown_parameters:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="unknown_parameters",
                        severity=DiagnosticSeverity.WARNING,
                        message="Node contains parameters not declared by its manifest",
                        node_id=node.id,
                        details={"parameters": unknown_parameters},
                    )
                )
            for parameter in manifest.parameters:
                value = node.parameters.get(parameter.id, parameter.default)
                if parameter.required and value is None:
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="missing_parameter",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"Parameter '{parameter.display_name}' is required",
                            node_id=node.id,
                        )
                    )
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
                        )
                    )
                    continue
                if parameter.type == "enum" and parameter.options and value not in parameter.options:
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="invalid_parameter_option",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"{parameter.display_name} has an unsupported value",
                            node_id=node.id,
                            details={"allowed": parameter.options, "value": value},
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
                            )
                        )
                minimum = parameter.constraints.get("min")
                maximum = parameter.constraints.get("max")
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
                        )
                    )
                if isinstance(value, list):
                    items = cast(list[Any], value)
                    expected_length = parameter.constraints.get("length")
                    minimum_length = parameter.constraints.get("min_length")
                    maximum_length = parameter.constraints.get("max_length")
                    if isinstance(expected_length, int) and len(items) != expected_length:
                        diagnostics.append(
                            ValidationDiagnostic(
                                code="parameter_length_invalid",
                                severity=DiagnosticSeverity.ERROR,
                                message=f"{parameter.display_name} must contain exactly {expected_length} values",
                                node_id=node.id,
                            )
                        )
                    if isinstance(minimum_length, int) and len(items) < minimum_length:
                        diagnostics.append(
                            ValidationDiagnostic(
                                code="parameter_length_below_minimum",
                                severity=DiagnosticSeverity.ERROR,
                                message=f"{parameter.display_name} must contain at least {minimum_length} values",
                                node_id=node.id,
                            )
                        )
                    if isinstance(maximum_length, int) and len(items) > maximum_length:
                        diagnostics.append(
                            ValidationDiagnostic(
                                code="parameter_length_above_maximum",
                                severity=DiagnosticSeverity.ERROR,
                                message=f"{parameter.display_name} must contain at most {maximum_length} values",
                                node_id=node.id,
                            )
                        )
                    item_minimum = parameter.constraints.get("item_min")
                    item_maximum = parameter.constraints.get("item_max")
                    for item in items:
                        if isinstance(item, int | float) and not isinstance(item, bool):
                            if isinstance(item_minimum, int | float) and item < item_minimum:
                                diagnostics.append(
                                    ValidationDiagnostic(
                                        code="parameter_item_below_minimum",
                                        severity=DiagnosticSeverity.ERROR,
                                        message=f"Values in {parameter.display_name} must be at least {item_minimum}",
                                        node_id=node.id,
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
                                    )
                                )
                                break
        incoming_by_port: Counter[tuple[str, str]] = Counter()
        outgoing_by_node: Counter[str] = Counter()
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
            source_manifest = manifests.get(source.id)
            target_manifest = manifests.get(target.id)
            if source_manifest is None or target_manifest is None:
                continue
            source_port = next((port for port in source_manifest.outputs if port.id == edge.source_port_id), None)
            target_port = next((port for port in target_manifest.inputs if port.id == edge.target_port_id), None)
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
            if source_port.kind != target_port.kind and target_port.kind.value != "tensor_list":
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
            outgoing_by_node[source.id] += 1
            if not target_port.accepts_multiple and incoming_by_port[(target.id, target_port.id)] > 1:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="multiple_connections_not_allowed",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Port '{target_port.display_name}' accepts only one connection",
                        node_id=target.id,
                    )
                )

        for node_id, manifest in manifests.items():
            for port in manifest.inputs:
                if port.required and incoming_by_port[(node_id, port.id)] == 0:
                    diagnostics.append(
                        ValidationDiagnostic(
                            code="missing_required_connection",
                            severity=DiagnosticSeverity.ERROR,
                            message=f"Required input '{port.display_name}' is not connected",
                            node_id=node_id,
                        )
                    )

        input_nodes = [node for node in graph.nodes if node.definition_id.startswith("input.")]
        output_nodes = [node for node in graph.nodes if node.definition_id.startswith("output.")]
        if not input_nodes:
            diagnostics.append(
                ValidationDiagnostic(
                    code="missing_model_input",
                    severity=DiagnosticSeverity.ERROR,
                    message="The model graph requires at least one input node",
                )
            )
        if not output_nodes:
            diagnostics.append(
                ValidationDiagnostic(
                    code="missing_model_output",
                    severity=DiagnosticSeverity.ERROR,
                    message="The model graph requires at least one output node",
                )
            )
        duplicate_output_names = [
            name
            for name, count in Counter(_output_name(node) for node in output_nodes).items()
            if count > 1
        ]
        if duplicate_output_names:
            diagnostics.append(
                ValidationDiagnostic(
                    code="duplicate_model_output",
                    severity=DiagnosticSeverity.ERROR,
                    message="Model output labels must be unique",
                    details={"labels": duplicate_output_names},
                )
            )
        for output in output_nodes:
            if outgoing_by_node[output.id] > 0:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="output_has_outgoing_edge",
                        severity=DiagnosticSeverity.WARNING,
                        message="A model output usually should not feed another graph node",
                        node_id=output.id,
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

        shape_summaries = []
        total_parameters: int | None = 0
        if not any(item.severity == DiagnosticSeverity.ERROR for item in diagnostics):
            try:
                shape_summaries = infer_shapes(graph, self._catalog)
                known_counts = [item.parameter_count for item in shape_summaries]
                total_parameters = (
                    sum(count for count in known_counts if count is not None)
                    if all(count is not None for count in known_counts)
                    else None
                )
            except (ShapeInferenceError, ValueError) as exc:
                diagnostics.append(
                    ValidationDiagnostic(
                        code="shape_inference_failed",
                        severity=DiagnosticSeverity.ERROR,
                        message=str(exc),
                    )
                )

        errors = [item for item in diagnostics if item.severity == DiagnosticSeverity.ERROR]
        compatibility_errors = [
            item for item in diagnostics if item.severity == DiagnosticSeverity.COMPATIBILITY
        ]
        return ValidationResult(
            valid=not errors and not compatibility_errors,
            diagnostics=diagnostics,
            shape_summaries=shape_summaries,
            total_parameters=total_parameters,
        )
