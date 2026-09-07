from __future__ import annotations

import re

from validation.safe_expression import SafeExpressionError, validate_safe_expression


###############################################################################
class CustomMathFormulaError(ValueError):
    pass


_MAX_FORMULA_LENGTH = 1200
_FUNCTION_COMMANDS = {
    "sqrt": "sqrt",
    "log": "log",
    "exp": "exp",
}
_OPERATOR_FUNCTIONS = {
    "abs": "abs",
    "clip": "clip",
    "maximum": "maximum",
    "minimum": "minimum",
    "pow": "pow",
    "softmax": "softmax",
}


###############################################################################
def _read_group(value: str, start: int) -> tuple[str, int]:
    if start >= len(value) or value[start] not in "{(":
        raise CustomMathFormulaError("Expected a grouped expression")
    opening = value[start]
    closing = "}" if opening == "{" else ")"
    depth = 0
    for index in range(start, len(value)):
        character = value[index]
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return value[start + 1 : index], index + 1
    raise CustomMathFormulaError("Formula contains an unclosed group")


###############################################################################
def _replace_fractions(value: str) -> str:
    marker = "\\frac"
    index = value.find(marker)
    while index >= 0:
        cursor = index + len(marker)
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        numerator, cursor = _read_group(value, cursor)
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        denominator, cursor = _read_group(value, cursor)
        replacement = f"(({_normalize_latex(numerator)})/({_normalize_latex(denominator)}))"
        value = value[:index] + replacement + value[cursor:]
        index = value.find(marker, index + len(replacement))
    return value


###############################################################################
def _replace_grouped_command(value: str, command: str, function: str) -> str:
    marker = f"\\{command}"
    index = value.find(marker)
    while index >= 0:
        cursor = index + len(marker)
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        argument, cursor = _read_group(value, cursor)
        replacement = f"{function}({_normalize_latex(argument)})"
        value = value[:index] + replacement + value[cursor:]
        index = value.find(marker, index + len(replacement))
    return value


###############################################################################
def _replace_operatorname(value: str) -> str:
    pattern = re.compile(r"\\operatorname\s*\{([A-Za-z]+)\}")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1).casefold()
        try:
            return _OPERATOR_FUNCTIONS[name]
        except KeyError as exc:
            raise CustomMathFormulaError(f"Function '{match.group(1)}' is not supported") from exc

    return pattern.sub(replace, value)


###############################################################################
def _strip_equation_label(value: str) -> str:
    depth = 0
    for index, character in enumerate(value):
        if character in "({[":
            depth += 1
        elif character in ")}]":
            depth = max(0, depth - 1)
        elif character == "=" and depth == 0:
            return value[index + 1 :].strip()
    return value


###############################################################################
def _normalize_latex(formula: str) -> str:
    value = formula.strip().strip("$").strip()
    value = _strip_equation_label(value)
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    value = value.replace("\\,", " ").replace("\\;", " ")
    value = _replace_fractions(value)
    for command, function in _FUNCTION_COMMANDS.items():
        value = _replace_grouped_command(value, command, function)
    value = _replace_operatorname(value)
    value = value.replace("{", "(").replace("}", ")")
    value = value.replace("^", "**")
    value = re.sub(r"(?<=\d)\s*(?=x\b)", "*", value)
    value = re.sub(r"(?<=\))\s*(?=x\b|\()", "*", value)
    value = re.sub(r"\bx\s*(?=\()", "x*", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


###############################################################################
def parse_custom_math_formula(formula: str) -> str:
    if len(formula) > _MAX_FORMULA_LENGTH:
        raise CustomMathFormulaError(
            f"Formula exceeds the {_MAX_FORMULA_LENGTH}-character authoring limit"
        )
    expression = _normalize_latex(formula)
    if not expression:
        raise CustomMathFormulaError("Formula cannot be empty")
    try:
        validate_safe_expression(expression)
    except SafeExpressionError as exc:
        raise CustomMathFormulaError(str(exc)) from exc
    return expression
