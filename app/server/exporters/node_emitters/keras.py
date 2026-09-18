from __future__ import annotations

from .contracts import (
    EmissionResult,
    NodeEmitterContext,
    NodeEmitterRegistry,
    emit_multi_output_assignment,
)


def _identity(context: NodeEmitterContext) -> EmissionResult:
    expression = next(iter(context.input_expressions_by_target_port.values()), "")
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


def _constant(context: NodeEmitterContext) -> EmissionResult:
    parameters = context.normalized_parameters
    expression = (
        f"keras.ops.broadcast_to(keras.ops.cast(keras.ops.convert_to_tensor({parameters.get('value', 0)!r}), "
        f"{parameters.get('dtype', 'float32')!r}), {parameters.get('shape', [1])!r})"
    )
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def _parameter(context: NodeEmitterContext) -> EmissionResult:
    parameters = context.normalized_parameters
    expression = (
        f"ParameterSource(shape={parameters.get('shape', [1])!r}, "
        f"initializer={parameters.get('initializer', 'zeros')!r}, "
        f"value_dtype={parameters.get('dtype', 'float32')!r}, "
        f"trainable={parameters.get('trainable', True)!r})()"
    )
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def _index(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    value = context.normalized_parameters.get("expression", ":")
    expression = value if isinstance(value, str) and value.startswith("x") else f"x[{value}]"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression.replace("x", source, 1)})


def _average(context: NodeEmitterContext) -> EmissionResult:
    inputs = list(context.input_expressions_by_target_port.values())
    expression = f"keras.ops.sum(keras.ops.stack([{', '.join(inputs)}], axis=0), axis=0) / {len(inputs)}"
    return EmissionResult(output_expressions={context.materialized_output_ports[0].id: expression})


def _split(context: NodeEmitterContext) -> EmissionResult:
    source = next(iter(context.input_expressions_by_target_port.values()))
    parameters = context.normalized_parameters
    expression = f"keras.ops.split({source}, {int(parameters.get('sections', 2))}, axis={int(parameters.get('axis', -1))})"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


def _detection(context: NodeEmitterContext) -> EmissionResult:
    expression = "[" + ", ".join(context.input_expressions_by_target_port.values()) + "]"
    return EmissionResult(output_expressions=emit_multi_output_assignment(expression, context.materialized_output_ports))


def build_registry() -> NodeEmitterRegistry:
    registry = NodeEmitterRegistry("keras")
    for adapter, emitter in {
        "identity": _identity,
        "constant": _constant,
        "parameter": _parameter,
        "index": _index,
        "average": _average,
        "split": _split,
        "detection": _detection,
        "safe_expression": _identity,
    }.items():
        registry.register(adapter, emitter)
    return registry
