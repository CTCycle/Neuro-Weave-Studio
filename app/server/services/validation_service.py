from __future__ import annotations

from collections import Counter
from typing import Any, cast

from common.catalogs.loader import NodeCatalog
from domain.compilation_catalog import CompilationCatalogProvider, get_compilation_catalog_provider
from domain.diagnostics import (
    ModelTypeInference,
    OutputReference,
    ValidationDiagnostic,
    ValidationResult,
)
from domain.enums import DiagnosticSeverity, ExecutionBackend
from domain.graph import ModelGraph, NodeInstance
from domain.manifests import NodeManifest, ParameterCapability, ParameterDefinition, ParameterValueSchema
from validation.compile_configuration import validate_model_configuration
from validation.graph_utils import nodes_reaching, reachable_nodes, topological_order
from validation.output_resolution import (
    infer_model_type,
    input_nodes as catalog_input_nodes,
    resolved_outputs,
)
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
            return all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        return isinstance(value, int) and not isinstance(value, bool)
    if parameter.type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if parameter.type in {"string", "enum", "expression", "dtype", "activation"}:
        return isinstance(value, str)
    if parameter.type in {"initializer", "regularizer", "constraint"}:
        return isinstance(value, str) or isinstance(value, dict)
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
        return all(_value_matches_schema(schema.item, item) for item in value)
    if schema.type == "integer" or schema.type == "axis":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema.type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if schema.type == "boolean":
        return isinstance(value, bool)
    if schema.type in {"string", "enum", "expression", "dtype", "activation"}:
        return isinstance(value, str)
    if schema.type in {"initializer", "regularizer", "constraint"}:
        return isinstance(value, str) or isinstance(value, dict)
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
    if capability is not None and capability.default is not None:
        return capability.default
    if parameter.framework_default is not None:
        return parameter.framework_default
    return parameter.default


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
    def validate(
        self,
        graph: ModelGraph,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
    ) -> ValidationResult:
        diagnostics: list[ValidationDiagnostic] = []
        nodes = {node.id: node for node in graph.nodes}
        manifests: dict[str, NodeManifest] = {}
        configuration_diagnostics = validate_model_configuration(
            graph,
            execution_backend,
            keras_enabled,
            self._compilation_catalog_provider.get(),
        )
        diagnostics.extend(configuration_diagnostics)
        implementation_key = "keras" if keras_enabled else execution_backend.value
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
                diagnostics.append(
                    ValidationDiagnostic(
                        code="unsupported_implementation",
                        severity=DiagnosticSeverity.COMPATIBILITY,
                        message=message,
                        node_id=node.id,
                        execution_backend=execution_backend,
                        keras_enabled=keras_enabled,
                    )
                )
            elif mapping.status == "unverified":
                diagnostics.append(
                    ValidationDiagnostic(
                        code="unverified_implementation",
                        severity=DiagnosticSeverity.COMPATIBILITY,
                        message=f"{manifest.display_name} has not been verified for {implementation_label}",
                        node_id=node.id,
                        execution_backend=execution_backend,
                        keras_enabled=keras_enabled,
                        details={"notes": mapping.notes},
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
                if supplied and not _condition_matches(parameter.enabled_when, node.parameters):
                    if value != default:
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
                (port for port in source_manifest.outputs if port.id == edge.source_port_id), None
            )
            target_port = next(
                (port for port in target_manifest.inputs if port.id == edge.target_port_id), None
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

        resolved = resolved_outputs(graph, self._catalog)
        explicit_outputs = bool(graph.output_bindings)
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
        if not catalog_input_nodes(graph, self._catalog):
            diagnostics.append(
                ValidationDiagnostic(
                    code="missing_model_input",
                    severity=DiagnosticSeverity.ERROR,
                    message="The model graph requires at least one input node",
                )
            )

        input_nodes = catalog_input_nodes(graph, self._catalog)
        input_ids = {node.id for node in input_nodes}
        output_ids = {item.node_id for item in effective_outputs}
        input_reachable = reachable_nodes(graph, input_ids)
        output_reachable = nodes_reaching(graph, output_ids)
        active_node_ids = input_reachable & output_reachable
        if input_ids and output_ids:
            for node in graph.nodes:
                if node.id in active_node_ids:
                    continue
                if node.id in input_ids:
                    code = "unreachable_model_input"
                    message = "Model input does not contribute to any model output"
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
                            for port in source_manifest.outputs
                            if port.id == edge.source_port_id
                        ),
                        None,
                    )
                    target_port = next(
                        (port for port in target_manifest.inputs if port.id == edge.target_port_id),
                        None,
                    )
                    source_spec = shape_by_port.get((edge.source_node_id, edge.source_port_id))
                    if source_port is None or target_port is None or source_spec is None:
                        continue
                    for port, label in ((source_port, "source"), (target_port, "target")):
                        if port.rank is not None and source_spec.rank != port.rank:
                            diagnostics.append(
                                ValidationDiagnostic(
                                    code="port_rank_mismatch",
                                    severity=DiagnosticSeverity.ERROR,
                                    message=(
                                        f"{label.title()} port '{port.display_name}' expects rank "
                                        f"{port.rank}, received rank {source_spec.rank}"
                                    ),
                                    edge_id=edge.id,
                                )
                            )
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
