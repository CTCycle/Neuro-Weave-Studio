from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from domain.manifests import ImplementationMapping, NodeManifest
from validation.safe_expression import validate_safe_expression


###############################################################################
class OperationEmissionError(ValueError):
    """Raised when a catalog operation cannot be emitted for a target."""


_SAFE_FUNCTION_SYMBOLS = {
    "keras": {
        "abs": "keras.ops.absolute",
        "clip": "keras.ops.clip",
        "exp": "keras.ops.exp",
        "log": "keras.ops.log",
        "maximum": "keras.ops.maximum",
        "minimum": "keras.ops.minimum",
        "pow": "keras.ops.power",
        "softmax": "keras.ops.softmax",
        "sqrt": "keras.ops.sqrt",
    },
    "tensorflow": {
        "abs": "tf.math.abs",
        "clip": "tf.clip_by_value",
        "exp": "tf.math.exp",
        "log": "tf.math.log",
        "maximum": "tf.maximum",
        "minimum": "tf.minimum",
        "pow": "tf.math.pow",
        "softmax": "tf.nn.softmax",
        "sqrt": "tf.math.sqrt",
    },
    "pytorch": {
        "abs": "torch.abs",
        "clip": "torch.clamp",
        "exp": "torch.exp",
        "log": "torch.log",
        "maximum": "torch.maximum",
        "minimum": "torch.minimum",
        "pow": "torch.pow",
        "softmax": "torch.softmax",
        "sqrt": "torch.sqrt",
    },
    "jax": {
        "abs": "jnp.abs",
        "clip": "jnp.clip",
        "exp": "jnp.exp",
        "log": "jnp.log",
        "maximum": "jnp.maximum",
        "minimum": "jnp.minimum",
        "pow": "jnp.power",
        "softmax": "jax.nn.softmax",
        "sqrt": "jnp.sqrt",
    },
}
_BINARY_OPERATOR_SYMBOLS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "**",
    ast.Mod: "%",
}
_UNARY_OPERATOR_SYMBOLS: dict[type[ast.unaryop], str] = {ast.UAdd: "+", ast.USub: "-"}

###############################################################################
def _mapping(manifest: NodeManifest, implementation_key: str) -> ImplementationMapping:
    mapping = manifest.implementations.get(implementation_key)
    if mapping is None or not mapping.supported:
        raise OperationEmissionError(
            f"No supported {implementation_key} operation mapping for {manifest.id}"
        )
    if mapping.symbol is None and mapping.adapter is None:
        raise OperationEmissionError(
            f"Operation mapping {implementation_key} for {manifest.id} has no symbol or adapter"
        )
    return mapping

###############################################################################
def _resolved_parameters(manifest: NodeManifest, parameters: Mapping[str, Any]) -> dict[str, Any]:
    resolved = {parameter.id: parameter.default for parameter in manifest.parameters}
    resolved.update(parameters)
    return resolved

###############################################################################
def _parameter_name(mapping: ImplementationMapping, parameter_id: str, backend: str) -> str:
    capability = mapping.parameters.get(parameter_id)
    if capability is not None and capability.mapping:
        return capability.mapping
    if parameter_id == "axis" and backend == "pytorch":
        return "dim"
    if parameter_id == "keepdims" and backend == "pytorch":
        return "keepdim"
    return parameter_id

###############################################################################
def _literal(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    if isinstance(value, list):
        items = cast(list[Any], value)
        return "[" + ", ".join(_literal(item) for item in items) + "]"
    raise OperationEmissionError("Operation parameters must be scalar or serializable lists")

###############################################################################
def _symbol(mapping: ImplementationMapping, parameters: Mapping[str, Any]) -> str:
    if mapping.adapter == "reduction":
        operation = str(parameters.get("operation", "sum"))
        try:
            return mapping.variant_symbols[operation]
        except KeyError as exc:
            raise OperationEmissionError(f"Unsupported reduction operation '{operation}'") from exc
    if mapping.symbol is None:
        raise OperationEmissionError("Operation mapping does not expose a callable symbol")
    return mapping.symbol

###############################################################################
def _keywords(
    manifest: NodeManifest,
    mapping: ImplementationMapping,
    parameters: Mapping[str, Any],
    backend: str,
) -> list[str]:
    if manifest.operation is None:
        return []
    return [
        f"{_parameter_name(mapping, parameter_id, backend)}={_literal(parameters.get(parameter_id))}"
        for parameter_id in manifest.operation.keywords
        if parameter_id in parameters
    ]

###############################################################################
def _safe_expression_source(expression: str, source: str, backend: str) -> str:
    validate_safe_expression(expression)
    tree = ast.parse(expression, mode="eval")

    def render_slice(node: ast.AST) -> str:
        if isinstance(node, ast.Tuple):
            values = [render_slice(element) for element in node.elts]
            return ", ".join(values) + ("," if len(values) == 1 else "")
        if isinstance(node, ast.Slice):
            lower = "" if node.lower is None else render(node.lower)
            upper = "" if node.upper is None else render(node.upper)
            step = "" if node.step is None else f":{render(node.step)}"
            return f"{lower}:{upper}{step}"
        if isinstance(node, ast.Constant) and node.value is Ellipsis:
            return "..."
        return render(node)

    def render(node: ast.AST) -> str:
        if isinstance(node, ast.Expression):
            return render(node.body)
        if isinstance(node, ast.Name):
            if node.id == "x":
                return source
            if node.id in _SAFE_FUNCTION_SYMBOLS[backend]:
                return _SAFE_FUNCTION_SYMBOLS[backend][node.id]
            raise OperationEmissionError(f"Expression name '{node.id}' is not allowed")
        if isinstance(node, ast.Constant):
            return _literal(node.value)
        if isinstance(node, ast.BinOp):
            operator = _BINARY_OPERATOR_SYMBOLS.get(type(node.op))
            if operator is None:
                raise OperationEmissionError("Expression operator is not allowed")
            return f"({render(node.left)} {operator} {render(node.right)})"
        if isinstance(node, ast.UnaryOp):
            operator = _UNARY_OPERATOR_SYMBOLS.get(type(node.op))
            if operator is None:
                raise OperationEmissionError("Expression operator is not allowed")
            return f"({operator}{render(node.operand)})"
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise OperationEmissionError("Expression call is not allowed")
            symbol = _SAFE_FUNCTION_SYMBOLS[backend].get(node.func.id)
            if symbol is None or node.keywords:
                raise OperationEmissionError("Expression call is not allowed")
            arguments = [render(argument) for argument in node.args]
            if node.func.id == "softmax":
                axis = "dim" if backend == "pytorch" else "axis"
                arguments.append(f"{axis}=-1")
            return f"{symbol}({', '.join(arguments)})"
        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name) or node.value.id != "x":
                raise OperationEmissionError("Only the input tensor may be sliced")
            if isinstance(node.slice, ast.Slice) and all(
                value is None for value in (node.slice.lower, node.slice.upper, node.slice.step)
            ):
                return source
            return f"{render(node.value)}[{render_slice(node.slice)}]"
        raise OperationEmissionError(f"Expression element '{type(node).__name__}' is not allowed")

    return render(tree)

###############################################################################
def _dtype_literal(value: Any, backend: str) -> str:
    dtype = str(value)
    if backend == "keras":
        return repr(dtype)
    prefix = {"tensorflow": "tf", "pytorch": "torch", "jax": "jnp"}[backend]
    return f"{prefix}.{dtype}"

###############################################################################
def _binary_expression(symbol: str, inputs: Sequence[str]) -> str:
    if len(inputs) < 2:
        raise OperationEmissionError("Binary operations require at least two inputs")
    result = f"{symbol}({inputs[0]}, {inputs[1]})"
    for value in inputs[2:]:
        result = f"{symbol}({result}, {value})"
    return result

###############################################################################
def emit_operation_source(
    manifest: NodeManifest,
    implementation_key: str,
    inputs: Sequence[str],
    parameters: Mapping[str, Any],
) -> str:
    """Emit a safe source expression from typed catalog metadata."""

    operation = manifest.operation
    if operation is None:
        raise OperationEmissionError(f"{manifest.id} is not a catalog operation")
    parameters = _resolved_parameters(manifest, parameters)
    mapping = _mapping(manifest, implementation_key)
    backend = "keras" if implementation_key == "keras" else implementation_key
    if not inputs:
        raise OperationEmissionError(f"Operation {manifest.id} requires an input")
    if mapping.adapter == "safe_expression":
        expression = parameters.get("expression")
        if not isinstance(expression, str):
            raise OperationEmissionError("Safe mathematical expression must be a string")
        return _safe_expression_source(expression, inputs[0], backend)

    symbol = _symbol(mapping, parameters)
    if operation.family in {"binary", "elementwise"}:
        return _binary_expression(symbol, inputs)
    if operation.family == "unary":
        arguments = [inputs[0]]
        arguments.extend(
            _literal(parameters.get(parameter_id)) for parameter_id in operation.arguments
        )
        return f"{symbol}({', '.join([*arguments, *_keywords(manifest, mapping, parameters, backend)])})"
    if operation.family == "reduction":
        arguments = [inputs[0]]
        if backend == "pytorch" and symbol in {"torch.var", "torch.std"}:
            arguments.append("correction=0")
        return f"{symbol}({', '.join([*arguments, *_keywords(manifest, mapping, parameters, backend)])})"

    behavior = operation.shape_behavior
    if behavior == "reshape":
        target = _literal(parameters.get("target_shape", [-1]))
        if backend == "keras":
            return f"layers.Reshape(target_shape={target})({inputs[0]})"
        elif backend == "tensorflow":
            target = f"[tf.shape({inputs[0]})[0], *{target}]"
        elif backend == "pytorch":
            target = f"({inputs[0]}.shape[0], *{target})"
        else:
            target = f"({inputs[0]}.shape[0], *{target})"
        return f"{symbol}({inputs[0]}, {target})"
    if behavior == "transpose":
        axes = parameters.get("axes", [])
        if not axes:
            if backend == "pytorch":
                return f"{inputs[0]}.permute(*reversed(range({inputs[0]}.ndim)))"
            return f"{symbol}({inputs[0]})"
        axes_literal = _literal(axes)
        if backend == "pytorch":
            return f"{symbol}({inputs[0]}, {axes_literal})"
        keyword = "perm" if backend == "tensorflow" else "axes"
        return f"{symbol}({inputs[0]}, {keyword}={axes_literal})"
    if behavior == "expand_dims":
        keyword = _parameter_name(mapping, "axis", backend)
        return f"{symbol}({inputs[0]}, {keyword}={_literal(parameters.get('axis', -1))})"
    if behavior == "squeeze":
        keyword = _parameter_name(mapping, "axis", backend)
        return f"{symbol}({inputs[0]}, {keyword}={_literal(parameters.get('axis', -1))})"
    if behavior == "flatten":
        if backend == "keras":
            return f"layers.Flatten()({inputs[0]})"
        if backend == "pytorch":
            return f"{symbol}({inputs[0]}, start_dim=1)"
        if backend == "tensorflow":
            return f"{symbol}({inputs[0]}, [tf.shape({inputs[0]})[0], -1])"
        return f"{symbol}({inputs[0]}, ({inputs[0]}.shape[0], -1))"
    if behavior == "repeat":
        arguments = [inputs[0], _literal(parameters.get("repeats", 1))]
        arguments.extend(_keywords(manifest, mapping, parameters, backend))
        return f"{symbol}({', '.join(arguments)})"
    if behavior == "tile":
        multiples = _literal(parameters.get("multiples", []))
        if backend == "pytorch":
            multiples = f"tuple({multiples})"
        return f"{symbol}({inputs[0]}, {multiples})"
    if behavior == "cast":
        dtype = _dtype_literal(parameters.get("dtype", "float32"), backend)
        if backend == "pytorch":
            return f"{inputs[0]}.to(dtype={dtype})"
        if backend == "jax":
            return f"{inputs[0]}.astype({dtype})"
        return f"{symbol}({inputs[0]}, {dtype})"
    if behavior in {"concatenate", "stack"}:
        axis = _literal(parameters.get("axis", -1 if behavior == "concatenate" else 0))
        keyword = "dim" if backend == "pytorch" else "axis"
        return f"{symbol}([{', '.join(inputs)}], {keyword}={axis})"
    if behavior == "matmul":
        return _binary_expression(symbol, inputs)
    if behavior == "take":
        index = _literal(parameters.get("index", 0))
        axis = _literal(parameters.get("axis", -1))
        if backend == "pytorch":
            return f"{symbol}({inputs[0]}, dim={axis}, index={index})"
        return f"{symbol}({inputs[0]}, {index}, axis={axis})"
    if behavior == "preserve":
        return f"{symbol}({inputs[0]})"
    raise OperationEmissionError(f"No source emitter for operation behavior '{behavior}'")

###############################################################################
def evaluate_safe_expression(
    expression: str,
    value: Any,
    functions: Mapping[str, Callable[..., Any]],
    backend: str = "keras",
) -> Any:
    """Evaluate a validated expression using only caller-provided callables."""

    validate_safe_expression(expression)
    tree = ast.parse(expression, mode="eval")
    symbols = _SAFE_FUNCTION_SYMBOLS[backend]

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Name):
            if node.id == "x":
                return value
            symbol = symbols.get(node.id)
            if symbol is None:
                raise OperationEmissionError(f"Expression name '{node.id}' is not allowed")
            return functions[symbol]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            operator = type(node.op)
            if operator is ast.Add:
                return left + right
            if operator is ast.Sub:
                return left - right
            if operator is ast.Mult:
                return left * right
            if operator is ast.Div:
                return left / right
            if operator is ast.Pow:
                return left**right
            if operator is ast.Mod:
                return left % right
            raise OperationEmissionError("Expression operator is not allowed")
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise OperationEmissionError("Expression operator is not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise OperationEmissionError("Expression call is not allowed")
            function = evaluate(node.func)
            return function(*(evaluate(argument) for argument in node.args))
        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name) or node.value.id != "x":
                raise OperationEmissionError("Only the input tensor may be sliced")
            if isinstance(node.slice, ast.Slice) and all(
                value is None for value in (node.slice.lower, node.slice.upper, node.slice.step)
            ):
                return value

            def evaluate_slice(slice_node: ast.AST) -> Any:
                if isinstance(slice_node, ast.Tuple):
                    return tuple(evaluate_slice(element) for element in slice_node.elts)
                if isinstance(slice_node, ast.Slice):
                    return slice(
                        None if slice_node.lower is None else evaluate(slice_node.lower),
                        None if slice_node.upper is None else evaluate(slice_node.upper),
                        None if slice_node.step is None else evaluate(slice_node.step),
                    )
                if isinstance(slice_node, ast.Constant) and slice_node.value is Ellipsis:
                    return Ellipsis
                return evaluate(slice_node)

            return value[evaluate_slice(node.slice)]
        raise OperationEmissionError(f"Expression element '{type(node).__name__}' is not allowed")

    return evaluate(tree)

###############################################################################
def apply_runtime_operation(
    manifest: NodeManifest,
    implementation_key: str,
    inputs: Sequence[Any],
    parameters: Mapping[str, Any],
    functions: Mapping[str, Callable[..., Any]],
) -> Any:
    """Apply a catalog operation without importing or evaluating user code."""

    operation = manifest.operation
    if operation is None:
        raise OperationEmissionError(f"{manifest.id} is not a catalog operation")
    parameters = _resolved_parameters(manifest, parameters)
    mapping = _mapping(manifest, implementation_key)
    if mapping.adapter == "safe_expression":
        expression = parameters.get("expression")
        if not isinstance(expression, str):
            raise OperationEmissionError("Safe mathematical expression must be a string")
        return evaluate_safe_expression(expression, inputs[0], functions, implementation_key)
    symbol = _symbol(mapping, parameters)
    function = functions.get(symbol)
    if function is None:
        raise OperationEmissionError(f"No runtime callable registered for {symbol}")
    if operation.family in {"binary", "elementwise"}:
        result = function(inputs[0], inputs[1])
        for value in inputs[2:]:
            result = function(result, value)
        return result
    if operation.family == "unary":
        arguments = [
            inputs[0],
            *(parameters.get(parameter_id) for parameter_id in operation.arguments),
        ]
        keywords = {
            _parameter_name(mapping, parameter_id, implementation_key): parameters[parameter_id]
            for parameter_id in operation.keywords
            if parameter_id in parameters
        }
        return function(*arguments, **keywords)
    if operation.family == "reduction":
        keywords = {
            _parameter_name(mapping, parameter_id, implementation_key): parameters[parameter_id]
            for parameter_id in operation.keywords
            if parameter_id in parameters
        }
        if symbol in {"keras.ops.var", "keras.ops.std"}:
            return function(inputs[0], **keywords)
        return function(inputs[0], **keywords)
    behavior = operation.shape_behavior
    if behavior in {"concatenate", "stack"}:
        return function(
            list(inputs), axis=parameters.get("axis", -1 if behavior == "concatenate" else 0)
        )
    if behavior == "cast":
        return function(inputs[0], parameters.get("dtype", "float32"))
    if behavior == "reshape":
        target = parameters.get("target_shape", [-1])
        batch_dimension = inputs[0].shape[0]
        if implementation_key == "keras":
            shape_function = functions.get("keras.ops.shape")
            if shape_function is not None:
                batch_dimension = shape_function(inputs[0])[0]
        return function(inputs[0], [batch_dimension, *target])
    if behavior == "take":
        return function(inputs[0], parameters.get("index", 0), axis=parameters.get("axis", -1))
    if behavior == "repeat":
        return function(inputs[0], parameters.get("repeats", 1), axis=parameters.get("axis", -1))
    if behavior == "tile":
        return function(inputs[0], parameters.get("multiples", []))
    if behavior in {"expand_dims", "squeeze"}:
        return function(inputs[0], axis=parameters.get("axis", -1))
    if behavior == "flatten":
        batch_dimension = inputs[0].shape[0]
        if implementation_key == "keras":
            shape_function = functions.get("keras.ops.shape")
            if shape_function is not None:
                batch_dimension = shape_function(inputs[0])[0]
        return function(inputs[0], [batch_dimension, -1])
    if behavior == "transpose":
        axes = parameters.get("axes", [])
        return function(inputs[0], axes=axes or None)
    if behavior == "matmul":
        return function(inputs[0], inputs[1])
    return function(inputs[0])
