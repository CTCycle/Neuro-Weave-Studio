from __future__ import annotations

from .contracts import (
    EmissionResult,
    NodeEmitterContext,
    NodeEmitterRegistry,
    emit_multi_output_assignment,
)


def _simple(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()), "")
    return EmissionResult(output_expressions=emit_multi_output_assignment(source, context.materialized_output_ports))


def _constant(context: NodeEmitterContext) -> EmissionResult:
    parameters = context.normalized_parameters
    expression = f"torch.as_tensor({parameters.get('value', 0)!r}, dtype=getattr(torch, {parameters.get('dtype', 'float32')!r})).expand({tuple(parameters.get('shape', [1]))!r})"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def _parameter(context: NodeEmitterContext) -> EmissionResult:
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: f"self.{context.canonical_ir_node.variable_name}"})


def _index(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    value = context.normalized_parameters.get("expression", ":")
    expression = value if isinstance(value, str) and value.startswith("x") else f"x[{value}]"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression.replace("x", source, 1)})


def _average(context: NodeEmitterContext) -> EmissionResult:
    inputs = list(context.input_expressions_by_target_port.values())
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: f"torch.stack([{', '.join(inputs)}], dim=0).sum(dim=0) / {len(inputs)}"})


def _split(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    expression = f"torch.tensor_split({source}, {int(parameters.get('sections', 2))}, dim={int(parameters.get('axis', -1))})"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


def _detection(context: NodeEmitterContext) -> EmissionResult:
    expression = "(" + ", ".join(context.input_expressions_by_target_port.values()) + ")"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


def build_registry() -> NodeEmitterRegistry:
    registry = NodeEmitterRegistry("pytorch")
    for adapter, emitter in {
        "identity": _simple,
        "constant": _constant,
        "parameter": _parameter,
        "index": _index,
        "average": _average,
        "split": _split,
        "detection": _detection,
        "safe_expression": _simple,
    }.items():
        registry.register(adapter, emitter)
    return registry
