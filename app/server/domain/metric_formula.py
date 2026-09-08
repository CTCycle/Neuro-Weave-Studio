from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, cast


class MetricFormulaError(ValueError):
    """Raised when a user formula is outside the supported safe subset."""


@dataclass(frozen=True, slots=True)
class FormulaNode:
    op: str
    value: float | None = None
    args: tuple[FormulaNode, ...] = ()

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op}
        if self.value is not None:
            result["value"] = self.value
        if self.args:
            result["args"] = [item.to_json() for item in self.args]
        return result

    @classmethod
    def from_json(cls, value: object) -> FormulaNode:
        if not isinstance(value, dict):
            raise MetricFormulaError("Stored metric formula AST is invalid")
        mapping = cast(dict[str, object], value)
        op = mapping.get("op")
        if not isinstance(op, str):
            raise MetricFormulaError("Stored metric formula AST is invalid")
        raw_args = mapping.get("args", [])
        if not isinstance(raw_args, list):
            raise MetricFormulaError("Stored metric formula arguments are invalid")
        raw_value = mapping.get("value")
        number = float(raw_value) if isinstance(raw_value, int | float) else None
        return cls(
            op=op,
            value=number,
            args=tuple(cls.from_json(item) for item in cast(list[object], raw_args)),
        )


_MAX_LENGTH = 500
_MAX_DEPTH = 18
_MAX_NODES = 80
_FUNCTIONS = {"abs", "square", "sqrt", "log", "exp", "mean", "sum"}
_BINARY = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
}


def _read_group(value: str, start: int) -> tuple[str, int]:
    opening = value[start]
    closing = "}" if opening == "{" else ")"
    depth = 0
    for index in range(start, len(value)):
        char = value[index]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return value[start + 1 : index], index + 1
    raise MetricFormulaError("Formula contains an unclosed group")


def _replace_frac(value: str) -> str:
    marker = "\\frac"
    index = value.find(marker)
    while index >= 0:
        cursor = index + len(marker)
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if cursor >= len(value) or value[cursor] not in "{(":
            raise MetricFormulaError("\\frac requires numerator and denominator groups")
        numerator, cursor = _read_group(value, cursor)
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if cursor >= len(value) or value[cursor] not in "{(":
            raise MetricFormulaError("\\frac requires numerator and denominator groups")
        denominator, cursor = _read_group(value, cursor)
        replacement = f"({_replace_frac(numerator)})/({_replace_frac(denominator)})"
        value = value[:index] + replacement + value[cursor:]
        index = value.find(marker, index + len(replacement))
    return value


def _replace_command_groups(value: str, command: str, function: str) -> str:
    marker = f"\\{command}"
    index = value.find(marker)
    while index >= 0:
        cursor = index + len(marker)
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if cursor >= len(value) or value[cursor] not in "{(":
            raise MetricFormulaError(f"\\{command} requires one grouped argument")
        argument, cursor = _read_group(value, cursor)
        replacement = f"{function}({_normalize_latex(argument)})"
        value = value[:index] + replacement + value[cursor:]
        index = value.find(marker, index + len(replacement))
    return value


def _replace_absolute(value: str) -> str:
    value = value.replace("\\left|", "|").replace("\\right|", "|")
    value = value.replace("\\lvert", "|").replace("\\rvert", "|")
    while "|" in value:
        start = value.find("|")
        end = value.find("|", start + 1)
        if end < 0:
            raise MetricFormulaError("Absolute-value bars must be paired")
        value = value[:start] + f"abs({value[start + 1:end]})" + value[end + 1 :]
    return value


def _replace_sums(value: str) -> str:
    pattern = re.compile(r"\\sum\s*_(?:\{[^{}]*\}|[^\s^]+)\s*\^(?:\{[^{}]*\}|[^\s]+)")
    value = pattern.sub("sum", value)
    value = re.sub(r"\bsum\s*\{", "sum(", value)
    value = re.sub(r"(?:\(\s*1(?:\.0+)?\s*\)|1)\s*/\s*(?:\(\s*n\s*\)|n)\s*\*?\s*sum\s*\(", "mean(", value)
    return value


def _strip_equation_label(value: str) -> str:
    depth = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = set(pairs.values())
    for index, char in enumerate(value):
        if char in pairs:
            depth += 1
        elif char in closing:
            depth = max(0, depth - 1)
        elif char == "=" and depth == 0 and (index + 1 >= len(value) or value[index + 1] != "="):
            return value[index + 1 :].strip()
    return value


def _normalize_latex(value: str) -> str:
    if len(value) > _MAX_LENGTH:
        raise MetricFormulaError("Formula exceeds the 500-character limit")
    value = value.strip().strip("$").strip()
    value = _strip_equation_label(value)
    value = value.replace("ŷ", "yhat").replace("\\hat{y}", "yhat").replace("\\hat y", "yhat")
    index = r"(?:_\{[^{}]*\}|_[A-Za-z0-9]+)"
    value = re.sub(r"\\hat\s*(?:\{y\}|y)" + index + "?", "yhat", value)
    value = re.sub(r"\by\s*" + index, "y", value)
    value = re.sub(r"\byhat\s*" + index, "yhat", value)
    value = value.replace("\\left", "").replace("\\right", "")
    value = re.sub(r"\\operatorname\s*\{[^{}]*\}", "", value)
    value = value.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    value = value.replace("\\,", " ").replace("\\;", " ")
    value = _replace_frac(value)
    for command, function in (("sqrt", "sqrt"), ("log", "log"), ("exp", "exp")):
        value = _replace_command_groups(value, command, function)
    value = _replace_sums(value)
    value = _replace_absolute(value)
    value = value.replace("{", "(").replace("}", ")")
    value = re.sub(r"\^\s*\(\s*2\s*\)", "**2", value)
    value = re.sub(r"\^\s*2\b", "**2", value)
    value = re.sub(r"\b(mean|sum)\(([^()]*)\)\*\*2", r"\1((\2)**2)", value)
    value = re.sub(r"\babs\s*\(([^()]*)\)", r"abs(\1)", value)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise MetricFormulaError("Formula cannot be empty")
    return value


def _from_python(node: ast.AST, depth: int, count: list[int]) -> FormulaNode:
    count[0] += 1
    if count[0] > _MAX_NODES or depth > _MAX_DEPTH:
        raise MetricFormulaError("Formula is too complex")
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return FormulaNode("constant", float(node.value))
    if isinstance(node, ast.Name):
        if node.id == "y":
            return FormulaNode("target")
        if node.id == "yhat":
            return FormulaNode("prediction")
        raise MetricFormulaError(f"Name '{node.id}' is not allowed")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        child = _from_python(node.operand, depth + 1, count)
        return child if isinstance(node.op, ast.UAdd) else FormulaNode("neg", args=(child,))
    if isinstance(node, ast.BinOp):
        operation = next((name for operator, name in _BINARY.items() if isinstance(node.op, operator)), None)
        if operation is None:
            if isinstance(node.op, ast.Pow):
                exponent = _from_python(node.right, depth + 1, count)
                if exponent.op != "constant" or exponent.value != 2:
                    raise MetricFormulaError("Only square (power 2) is supported")
                return FormulaNode("square", args=(_from_python(node.left, depth + 1, count),))
            raise MetricFormulaError("Only +, -, *, and / are supported")
        return FormulaNode(operation, args=(_from_python(node.left, depth + 1, count), _from_python(node.right, depth + 1, count)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        if node.keywords or len(node.args) != 1:
            raise MetricFormulaError(f"{node.func.id} accepts exactly one argument")
        return FormulaNode(node.func.id, args=(_from_python(node.args[0], depth + 1, count),))
    raise MetricFormulaError("Formula contains an unsupported expression")


def parse_metric_formula(formula: str) -> FormulaNode:
    """Parse the bounded LaTeX-like formula subset without evaluating it."""

    normalized = _normalize_latex(formula)
    normalized = re.sub(r"(?<=\))\s*(?=[A-Za-z_])", "*", normalized)
    normalized = re.sub(r"(?<=\d)\s*(?=[A-Za-z_])", "*", normalized)
    normalized = re.sub(r"(?<=\))\s*(?=\()", "*", normalized)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise MetricFormulaError("Formula syntax is invalid") from exc
    return _from_python(tree.body, 0, [0])


def emit_metric_formula(node: FormulaNode, backend: str, target: str = "y_true", prediction: str = "y_pred") -> str:
    """Emit an expression using only the selected backend's tensor primitives."""

    if node.op == "target":
        return target
    if node.op == "prediction":
        return prediction
    if node.op == "constant":
        return repr(node.value)
    if node.op == "neg":
        return f"(-{emit_metric_formula(node.args[0], backend, target, prediction)})"
    if node.op in {"add", "sub", "mul", "div"}:
        operator = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[node.op]
        return f"({emit_metric_formula(node.args[0], backend, target, prediction)} {operator} {emit_metric_formula(node.args[1], backend, target, prediction)})"
    function = {
        "abs": "abs",
        "square": "square",
        "sqrt": "sqrt",
        "log": "log",
        "exp": "exp",
        "mean": "mean",
        "sum": "sum",
    }.get(node.op)
    if function is None:
        raise MetricFormulaError(f"Unknown formula node '{node.op}'")
    argument = emit_metric_formula(node.args[0], backend, target, prediction)
    if backend == "keras":
        return f"keras.ops.{function}({argument})"
    if backend == "pytorch":
        return f"torch.{function}({argument})"
    if backend == "tensorflow":
        return f"tf.{function}({argument})"
    if backend == "jax":
        return f"jnp.{function}({argument})"
    raise MetricFormulaError(f"Unsupported formula backend '{backend}'")


def evaluate_metric_formula(
    node: FormulaNode,
    target: Any,
    prediction: Any,
    operations: dict[str, Any],
) -> Any:
    """Evaluate a validated tree through caller-supplied tensor operations."""

    if node.op == "target":
        return target
    if node.op == "prediction":
        return prediction
    if node.op == "constant":
        return node.value
    if node.op == "neg":
        return -evaluate_metric_formula(node.args[0], target, prediction, operations)
    if node.op in {"add", "sub", "mul", "div"}:
        left = evaluate_metric_formula(node.args[0], target, prediction, operations)
        right = evaluate_metric_formula(node.args[1], target, prediction, operations)
        return {"add": left + right, "sub": left - right, "mul": left * right, "div": left / right}[node.op]
    argument = evaluate_metric_formula(node.args[0], target, prediction, operations)
    function = operations.get(node.op)
    if not callable(function):
        raise MetricFormulaError(f"No operation registered for '{node.op}'")
    return function(argument)
