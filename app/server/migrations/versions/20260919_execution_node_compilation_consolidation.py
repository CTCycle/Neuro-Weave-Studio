"""Normalize execution contracts and remove obsolete persisted nodes.

This migration intentionally performs the placeholder-node removal described in
the serialization contract. The old detection placeholder carried no typed
boxes, scores, or classes, so incident edges and output bindings are removed
without guessing their meaning.
"""

from __future__ import annotations

import json
from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "20260919_execution_node_compilation_consolidation"
down_revision = "20260915_remove_graph_schema_tracking"
branch_labels = None
depends_on = None

_ADAM_DEFAULTS = {
    "learning_rate": 0.001,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-7,
}


###############################################################################
def _object(value: Any, default_id: str | None = None) -> dict[str, Any] | None:
    if value is None:
        return {"id": default_id, "parameters": {}} if default_id else None
    if isinstance(value, str):
        return {"id": value, "parameters": {}}
    if isinstance(value, dict):
        identifier = value.get("id", default_id)
        if not isinstance(identifier, str) or not identifier:
            return {"id": default_id or "adam", "parameters": {}}
        parameters = value.get("parameters", {})
        return {
            "id": identifier,
            "parameters": dict(parameters) if isinstance(parameters, dict) else {},
        }
    return {"id": default_id or "adam", "parameters": {}}


def _plan(value: Any, *, native_backend: str | None = None) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, dict) else {}
    source = raw.get("compilation")
    if isinstance(source, dict):
        raw = {**raw, **source}
    if native_backend in {"tensorflow", "jax"}:
        optimizer = {"id": "adam", "parameters": dict(_ADAM_DEFAULTS)}
        losses: dict[str, Any] = {}
        metrics: dict[str, list[Any]] = {}
    else:
        optimizer = _object(raw.get("optimizer", raw.get("optimizer_id")), "adam")
        optimizer_parameters = raw.get("optimizer_parameters")
        if optimizer is not None and not optimizer["parameters"] and isinstance(optimizer_parameters, dict):
            optimizer["parameters"] = dict(optimizer_parameters)
        losses = {
            str(selector): _object(choice) or {"id": "mean_squared_error", "parameters": {}}
            for selector, choice in dict(raw.get("losses", raw.get("loss_ids", {}))).items()
        } if isinstance(raw.get("losses", raw.get("loss_ids", {})), dict) else {}
        metrics_raw = raw.get("metrics", raw.get("metric_ids", {}))
        metrics = {}
        if isinstance(metrics_raw, dict):
            for selector, choices in metrics_raw.items():
                if isinstance(choices, list):
                    metrics[str(selector)] = [_object(choice) or {"id": str(choice), "parameters": {}} for choice in choices]
                elif choices is not None:
                    metrics[str(selector)] = [_object(choices) or {"id": str(choices), "parameters": {}}]
    return {
        "optimizer": optimizer or {"id": "adam", "parameters": {}},
        "losses": losses,
        "metrics": metrics,
        "default_initializer": _object(raw.get("default_initializer"), "glorot_uniform"),
        "default_regularizer": None
        if raw.get("default_regularizer", "none") in (None, "none")
        else _object(raw.get("default_regularizer")),
        "default_constraint": None
        if raw.get("default_constraint", "none") in (None, "none")
        else _object(raw.get("default_constraint")),
    }


def _compiler(value: Any, backend: str) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, dict) else {}
    existing = raw.get("compiler")
    if isinstance(existing, dict):
        parameters = existing.get("parameters", {})
        return {
            "id": existing.get(
                "id",
                {
                    "tensorflow": "tensorflow_function",
                    "pytorch": "pytorch_compile",
                    "jax": "jax_jit",
                }[backend],
            ),
            "parameters": dict(parameters) if isinstance(parameters, dict) else {},
        }
    if backend == "tensorflow":
        parameters = {"jit_compile": bool(raw.get("jit_compile", False))}
        identifier = "tensorflow_function"
    elif backend == "pytorch":
        parameters = {
            "enabled": bool(raw.get("compile_enabled", False)),
            "mode": raw.get("mode", "default"),
            "fullgraph": bool(raw.get("fullgraph", False)),
            "dynamic": bool(raw.get("dynamic", False)),
        }
        identifier = "pytorch_compile"
    else:
        parameters = {"enabled": bool(raw.get("jit_enabled", False))}
        identifier = "jax_jit"
    return {"id": identifier, "parameters": parameters}


def _configuration(value: Any) -> dict[str, Any]:
    raw = dict(_json(value, {}))
    keras_raw = raw.get("keras", raw)
    native_raw = raw.get("native", {})
    native_raw = native_raw if isinstance(native_raw, dict) else {}
    return {
        "keras": {
            "compilation": _plan(keras_raw),
            "mixed_precision": bool(keras_raw.get("mixed_precision", False)) if isinstance(keras_raw, dict) else False,
            "random_seed": keras_raw.get("random_seed", 42) if isinstance(keras_raw, dict) else 42,
        },
        "native": {
            backend: {
                "compilation": _plan(native_raw.get(backend, {}), native_backend=backend),
                "compiler": _compiler(native_raw.get(backend, {}), backend),
                "random_seed": (
                    native_raw.get(backend, {}).get("random_seed", 42)
                    if isinstance(native_raw.get(backend, {}), dict)
                    else 42
                ),
            }
            for backend in ("tensorflow", "pytorch", "jax")
        },
    }


def _json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def _without_placeholder(value: Any) -> dict[str, Any]:
    raw = dict(_json(value, {}))
    removed = {
        str(node.get("id"))
        for node in raw.get("nodes", [])
        if isinstance(node, dict) and node.get("definition_id") == "output.detection_placeholder"
    }
    if removed:
        raw["nodes"] = [
            node for node in raw.get("nodes", [])
            if not isinstance(node, dict) or node.get("id") not in removed
        ]
        raw["edges"] = [
            edge for edge in raw.get("edges", [])
            if not isinstance(edge, dict)
            or (edge.get("source_node_id") not in removed and edge.get("target_node_id") not in removed)
        ]
        raw["output_bindings"] = [
            binding for binding in raw.get("output_bindings", [])
            if not isinstance(binding, dict) or binding.get("node_id") not in removed
        ]
    raw["schema_version"] = 6
    raw["model_configuration"] = _configuration(raw.get("model_configuration", {}))
    return raw


def _formula_capabilities(expression: Any) -> dict[str, bool]:
    try:
        from domain.metric_formula import FormulaNode, MetricFormulaError, emit_metric_formula

        tree = FormulaNode.from_json(expression)
        result: dict[str, bool] = {}
        for backend in ("keras", "tensorflow", "pytorch", "jax"):
            try:
                emit_metric_formula(tree, backend)
            except MetricFormulaError:
                result[backend] = False
            else:
                result[backend] = True
        return result
    except Exception:
        return {backend: False for backend in ("keras", "tensorflow", "pytorch", "jax")}


def _graph_template(spec: Any) -> Any:
    raw = dict(spec) if isinstance(spec, dict) else {}
    if raw.get("type") != "composite":
        return raw
    components = raw.get("components", [])
    if not isinstance(components, list) or not components:
        return raw
    normalized = []
    for index, component in enumerate(components):
        item = dict(component) if isinstance(component, dict) else {}
        normalized.append(
            {
                "id": f"component_{index}",
                "definition_id": item.get("definition_id", ""),
                "definition_version": item.get("definition_version", 1),
                "label": item.get("label"),
                "parameters": dict(item.get("parameters", {})) if isinstance(item.get("parameters", {}), dict) else {},
            }
        )
    edges = [
        {
            "source_component_id": normalized[index]["id"],
            "source_port_id": "output",
            "target_component_id": normalized[index + 1]["id"],
            "target_port_id": "input",
        }
        for index in range(len(normalized) - 1)
    ]
    return {
        "type": "graph_template",
        "components": normalized,
        "edges": edges,
        "input_bindings": [
            {
                "template_port_id": "input",
                "component_id": normalized[0]["id"],
                "port_id": "input",
            }
        ],
        "output_bindings": [
            {
                "template_port_id": "output",
                "component_id": normalized[-1]["id"],
                "port_id": "output",
            }
        ],
    }


###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("custom_metric_definitions", recreate="always") as batch_op:
        batch_op.add_column(
            sa.Column("capabilities_json", sa.JSON(), nullable=False, server_default="{}")
        )
        batch_op.alter_column("capabilities_json", server_default=None)

    connection = op.get_bind()
    placeholder_rows = connection.execute(
        sa.text(
            "SELECT project_id, node_id FROM project_nodes WHERE definition_id = 'output.detection_placeholder'"
        )
    ).mappings().all()
    placeholder_ids: dict[str, set[str]] = {}
    for placeholder in placeholder_rows:
        placeholder_ids.setdefault(str(placeholder["project_id"]), set()).add(
            str(placeholder["node_id"])
        )
    settings = connection.execute(
        sa.text("SELECT project_id, model_configuration_json, output_bindings_json FROM project_graph_settings")
    ).mappings().all()
    for row in settings:
        removed = placeholder_ids.get(str(row["project_id"]), set())
        bindings = [
            binding for binding in _json(row["output_bindings_json"], [])
            if not isinstance(binding, dict) or binding.get("node_id") not in removed
        ]
        connection.execute(
            sa.text(
                "UPDATE project_graph_settings SET model_configuration_json = :configuration, output_bindings_json = :bindings WHERE project_id = :project_id"
            ),
            {
                "project_id": row["project_id"],
                "configuration": json.dumps(_configuration(row["model_configuration_json"])),
                "bindings": json.dumps(bindings),
            },
        )

    connection.execute(
        sa.text(
            "DELETE FROM project_edges WHERE EXISTS (SELECT 1 FROM project_nodes n WHERE n.project_id = project_edges.project_id AND n.node_id IN (project_edges.source_node_id, project_edges.target_node_id) AND n.definition_id = 'output.detection_placeholder')"
        )
    )
    connection.execute(
        sa.text("DELETE FROM project_nodes WHERE definition_id = 'output.detection_placeholder'")
    )

    revisions = connection.execute(
        sa.text("SELECT id, graph_snapshot_json FROM project_revisions")
    ).mappings().all()
    for row in revisions:
        connection.execute(
            sa.text("UPDATE project_revisions SET graph_snapshot_json = :snapshot WHERE id = :id"),
            {"id": row["id"], "snapshot": json.dumps(_without_placeholder(row["graph_snapshot_json"]))},
        )

    custom_nodes = connection.execute(
        sa.text("SELECT id, spec_json FROM custom_node_definitions")
    ).mappings().all()
    for row in custom_nodes:
        connection.execute(
            sa.text("UPDATE custom_node_definitions SET spec_json = :spec WHERE id = :id"),
            {"id": row["id"], "spec": json.dumps(_graph_template(_json(row["spec_json"], {})))},
        )

    custom_metrics = connection.execute(
        sa.text("SELECT id, expression_json FROM custom_metric_definitions")
    ).mappings().all()
    for row in custom_metrics:
        connection.execute(
            sa.text("UPDATE custom_metric_definitions SET capabilities_json = :capabilities WHERE id = :id"),
            {"id": row["id"], "capabilities": json.dumps(_formula_capabilities(_json(row["expression_json"], {})))},
        )

    connection.execute(sa.text("DELETE FROM generated_code_snapshots"))


###############################################################################
def downgrade() -> None:
    with op.batch_alter_table("custom_metric_definitions", recreate="always") as batch_op:
        batch_op.drop_column("capabilities_json")
