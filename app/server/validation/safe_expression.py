from __future__ import annotations

import ast

_ALLOWED_FUNCTIONS = {
    "abs",
    "clip",
    "exp",
    "log",
    "maximum",
    "minimum",
    "pow",
    "softmax",
    "sqrt",
}
_ALLOWED_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARY_OPERATORS = (ast.UAdd, ast.USub)

###############################################################################
def _integer_literal(node: ast.AST) -> int | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, _ALLOWED_UNARY_OPERATORS)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    ):
        value = int(node.operand.value)
        return value if isinstance(node.op, ast.UAdd) else -value
    return None

###############################################################################
def _validate_slice(node: ast.AST) -> None:
    if isinstance(node, ast.Tuple):
        for element in node.elts:
            _validate_slice(element)
        return
    if isinstance(node, ast.Slice):
        for boundary in (node.lower, node.upper, node.step):
            if boundary is not None and _integer_literal(boundary) is None:
                raise SafeExpressionError("Slice boundaries must be integer literals")
        if node.step is not None and _integer_literal(node.step) == 0:
            raise SafeExpressionError("Slice step cannot be zero")
        return
    if isinstance(node, ast.Constant) and node.value is Ellipsis:
        return
    if _integer_literal(node) is not None:
        return
    raise SafeExpressionError("Slice indices must be integer literals, slices, or ellipsis")

###############################################################################
def _validate_expression_node(node: ast.AST) -> None:
    if isinstance(node, (ast.Expression, ast.Load)):
        if isinstance(node, ast.Expression):
            _validate_expression_node(node.body)
        return
    if isinstance(node, ast.Name):
        if node.id != "x" and node.id not in _ALLOWED_FUNCTIONS:
            raise SafeExpressionError(f"Name '{node.id}' is not allowed")
        return
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int | float) or isinstance(node.value, bool):
            raise SafeExpressionError("Only numeric constants are allowed")
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINARY_OPERATORS):
            raise SafeExpressionError("Binary operator is not allowed")
        _validate_expression_node(node.left)
        _validate_expression_node(node.right)
        return
    if isinstance(node, _ALLOWED_BINARY_OPERATORS + _ALLOWED_UNARY_OPERATORS):
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY_OPERATORS):
            raise SafeExpressionError("Unary operator is not allowed")
        _validate_expression_node(node.operand)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            raise SafeExpressionError("Only whitelisted function calls are allowed")
        if node.keywords:
            raise SafeExpressionError("Keyword arguments are not allowed")
        for argument in node.args:
            _validate_expression_node(argument)
        return
    if isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name) or node.value.id != "x":
            raise SafeExpressionError("Only the input tensor may be sliced")
        _validate_slice(node.slice)
        return
    raise SafeExpressionError(f"Expression element '{type(node).__name__}' is not allowed")

###############################################################################
class SafeExpressionError(ValueError):
    pass

###############################################################################
def validate_safe_expression(expression: str) -> None:
    if not expression.strip():
        raise SafeExpressionError("Expression cannot be empty")
    if len(expression) > 500:
        raise SafeExpressionError("Expression exceeds the 500-character limit")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SafeExpressionError("Expression syntax is invalid") from exc
    _validate_expression_node(tree)
