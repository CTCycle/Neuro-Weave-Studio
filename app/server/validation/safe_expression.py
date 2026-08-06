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

    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load)):
            continue
        if isinstance(node, ast.Name):
            if node.id != "x" and node.id not in _ALLOWED_FUNCTIONS:
                raise SafeExpressionError(f"Name '{node.id}' is not allowed")
            continue
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, int | float) or isinstance(node.value, bool):
                raise SafeExpressionError("Only numeric constants are allowed")
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINARY_OPERATORS):
                raise SafeExpressionError("Binary operator is not allowed")
            continue
        if isinstance(node, _ALLOWED_BINARY_OPERATORS + _ALLOWED_UNARY_OPERATORS):
            continue
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY_OPERATORS):
                raise SafeExpressionError("Unary operator is not allowed")
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
                raise SafeExpressionError("Only whitelisted function calls are allowed")
            if node.keywords:
                raise SafeExpressionError("Keyword arguments are not allowed")
            continue
        raise SafeExpressionError(f"Expression element '{type(node).__name__}' is not allowed")
