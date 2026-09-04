from __future__ import annotations

from typing import Any, cast

from domain.manifests import NodeManifest, ParameterDefinition, ParameterValueSchema
from validation.safe_expression import SafeExpressionError, validate_safe_expression


class ParameterValueError(ValueError):
    pass


def condition_matches(condition: dict[str, Any], parameters: dict[str, Any]) -> bool:
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
        return all(_value_matches_schema(schema.item, item) for item in cast(list[Any], value))
    if schema.type in {"integer", "axis"}:
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


def parameter_type_matches(parameter: ParameterDefinition, value: Any) -> bool:
    if parameter.value_schema is not None:
        return _value_matches_schema(parameter.value_schema, value)
    if value is None:
        return parameter.nullable
    if parameter.type == "boolean":
        return isinstance(value, bool)
    if parameter.type in {"integer", "axis"}:
        if parameter.type == "axis" and isinstance(value, list):
            return all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in cast(list[Any], value)
            )
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
        return isinstance(value, list) and all(
            dimension is None
            or (isinstance(dimension, int) and not isinstance(dimension, bool))
            or isinstance(dimension, list)
            for dimension in cast(list[Any], value)
        )
    if parameter.type in {"list", "tuple"}:
        return isinstance(value, list)
    return True


def _default(parameter: ParameterDefinition) -> Any:
    if "editor_default" in parameter.model_fields_set:
        return parameter.editor_default
    return parameter.default


def _validate_constraints(parameter: ParameterDefinition, value: Any) -> None:
    constraints = parameter.constraints
    minimum = constraints.get("min")
    maximum = constraints.get("max")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if isinstance(minimum, int | float) and value < minimum:
            raise ParameterValueError(
                f"Parameter '{parameter.display_name}' must be at least {minimum}"
            )
        if isinstance(maximum, int | float) and value > maximum:
            raise ParameterValueError(
                f"Parameter '{parameter.display_name}' must be at most {maximum}"
            )
    if not isinstance(value, list):
        return
    items = cast(list[Any], value)
    expected_length = constraints.get("length")
    minimum_length = constraints.get("min_length")
    maximum_length = constraints.get("max_length")
    if isinstance(expected_length, int) and len(items) != expected_length:
        raise ParameterValueError(
            f"Parameter '{parameter.display_name}' must contain exactly {expected_length} values"
        )
    if isinstance(minimum_length, int) and len(items) < minimum_length:
        raise ParameterValueError(
            f"Parameter '{parameter.display_name}' must contain at least {minimum_length} values"
        )
    if isinstance(maximum_length, int) and len(items) > maximum_length:
        raise ParameterValueError(
            f"Parameter '{parameter.display_name}' must contain at most {maximum_length} values"
        )
    item_minimum = constraints.get("item_min")
    item_maximum = constraints.get("item_max")
    numeric_items = [
        item
        for item in items
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]
    if isinstance(item_minimum, int | float) and any(item < item_minimum for item in numeric_items):
        raise ParameterValueError(
            f"Parameter '{parameter.display_name}' contains a value below {item_minimum}"
        )
    if isinstance(item_maximum, int | float) and any(item > item_maximum for item in numeric_items):
        raise ParameterValueError(
            f"Parameter '{parameter.display_name}' contains a value above {item_maximum}"
        )


def normalize_manifest_parameters(
    manifest: NodeManifest,
    supplied: dict[str, Any],
) -> dict[str, Any]:
    known = {parameter.id: parameter for parameter in manifest.parameters}
    unknown = sorted(set(supplied) - set(known))
    if unknown:
        raise ParameterValueError(
            f"Unknown parameters for {manifest.id}: {', '.join(unknown)}"
        )
    values = {parameter.id: _default(parameter) for parameter in manifest.parameters}
    values.update(supplied)
    for parameter in manifest.parameters:
        if not condition_matches(parameter.visible_when, values):
            continue
        supplied_value = parameter.id in supplied
        value = values.get(parameter.id)
        required = parameter.required or (
            bool(parameter.required_when) and condition_matches(parameter.required_when, values)
        )
        if required and not supplied_value and value is None:
            raise ParameterValueError(f"Parameter '{parameter.display_name}' is required")
        if value is None:
            if supplied_value and not parameter.nullable:
                raise ParameterValueError(
                    f"Parameter '{parameter.display_name}' cannot be null"
                )
            continue
        if not parameter_type_matches(parameter, value):
            raise ParameterValueError(
                f"Parameter '{parameter.display_name}' must be of type {parameter.type}"
            )
        if parameter.options and not isinstance(value, dict) and value not in parameter.options:
            raise ParameterValueError(
                f"Parameter '{parameter.display_name}' has an unsupported value"
            )
        if (
            supplied_value
            and not condition_matches(parameter.enabled_when, values)
            and value != _default(parameter)
        ):
            raise ParameterValueError(
                f"Parameter '{parameter.display_name}' is disabled by the current configuration"
            )
        if parameter.type == "shape" and isinstance(value, list) and any(
            item == 0 for item in cast(list[Any], value) if isinstance(item, int)
        ):
            raise ParameterValueError("Shape dimensions cannot be zero")
        if parameter.type == "expression" and isinstance(value, str):
            try:
                validate_safe_expression(value)
            except SafeExpressionError as exc:
                raise ParameterValueError(str(exc)) from exc
        _validate_constraints(parameter, value)
    return values
