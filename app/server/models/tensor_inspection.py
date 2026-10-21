# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from common.catalogs.loader import get_node_catalog
from domain.graph import ModelGraph
from validation.graph_utils import ordered_incoming_edges, topological_order

DEFAULT_INSPECTION_SEED = 1729
MAX_INSPECTION_NODES = 256
MAX_HOST_STAT_ELEMENTS = 200_000
MAX_STAT_SAMPLE_ELEMENTS = 50_000
MAX_FULL_VALUE_ELEMENTS = 16_384
MAX_SAMPLE_VALUE_ELEMENTS = 512
MAX_SESSION_VALUE_ELEMENTS = 100_000
HISTOGRAM_BINS = 24

###############################################################################
def _condition_matches(condition: dict[str, Any], parameters: dict[str, Any]) -> bool:
    for parameter_id, expected in condition.items():
        actual = parameters.get(parameter_id)
        if isinstance(expected, dict):
            if "equals" in expected and actual != expected["equals"]:
                return False
            if "not_equals" in expected and actual == expected["not_equals"]:
                return False
            if "in" in expected and actual not in cast(list[Any], expected["in"]):
                return False
            if "not_in" in expected and actual in cast(list[Any], expected["not_in"]):
                return False
        elif isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True

###############################################################################
def _enabled_output_ports(manifest: Any, parameters: dict[str, Any]) -> list[Any]:
    return [port for port in cast(list[Any], manifest.outputs) if _condition_matches(port.enabled_when, parameters)]

###############################################################################
def _flatten_symbolic(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in cast(list[Any] | tuple[Any, ...], value):
            flattened.extend(_flatten_symbolic(item))
        return flattened
    return [value]

###############################################################################
def _concrete_shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", ())
    result: list[int] = []
    for dimension in shape:
        try:
            concrete = int(dimension)
        except (TypeError, ValueError):
            concrete = 1
        result.append(concrete if concrete > 0 else 1)
    return result

###############################################################################
def _element_count(shape: list[int]) -> int:
    return math.prod(shape) if shape else 1

###############################################################################
def _json_value(value: Any) -> int | float | bool | str | None:
    if isinstance(value, (np.bool_, bool)):
        return bool(cast(Any, value))
    if isinstance(value, (np.integer, int)):
        return int(cast(Any, value))
    if isinstance(value, (np.floating, float)):
        numeric = float(cast(Any, value))
        if math.isnan(numeric):
            return "NaN"
        if math.isinf(numeric):
            return "Infinity" if numeric > 0 else "-Infinity"
        return numeric
    if value is None:
        return None
    return str(value)

###############################################################################
def _numeric_statistics(values: np.ndarray, *, total_elements: int, sampled: bool) -> dict[str, object] | None:
    if values.dtype.kind not in "biufc":
        return None
    flattened = np.asarray(values).reshape(-1)
    if flattened.dtype.kind == "c":
        return None
    numeric = flattened.astype(np.float64, copy=False)
    nan_mask = np.isnan(numeric)
    inf_mask = np.isinf(numeric)
    finite = numeric[np.isfinite(numeric)]
    inspected = int(numeric.size)
    statistics: dict[str, object] = {
        "sampled": sampled,
        "sample_size": inspected,
        "total_elements": total_elements,
        "nan_count": int(np.count_nonzero(nan_mask)),
        "infinite_count": int(np.count_nonzero(inf_mask)),
        "finite_count": int(finite.size),
        "zero_proportion": float(np.count_nonzero(numeric == 0) / inspected) if inspected else 0.0,
    }
    if finite.size:
        statistics.update(
            {
                "minimum": float(np.min(finite)),
                "maximum": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "standard_deviation": float(np.std(finite)),
                "median": float(np.median(finite)),
            }
        )
    else:
        statistics.update(
            {
                "minimum": None,
                "maximum": None,
                "mean": None,
                "standard_deviation": None,
                "median": None,
            }
        )
    return statistics

###############################################################################
def _histogram(values: np.ndarray) -> dict[str, object] | None:
    if values.dtype.kind not in "biuf":
        return None
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return None
    if np.min(finite) == np.max(finite):
        center = float(finite[0])
        width = max(abs(center) * 0.05, 0.5)
        edges = np.linspace(center - width, center + width, HISTOGRAM_BINS + 1)
        counts, edges = np.histogram(finite, bins=edges)
    else:
        counts, edges = np.histogram(finite, bins=HISTOGRAM_BINS)
    return {
        "counts": [int(item) for item in counts],
        "edges": [float(item) for item in edges],
        "sampled": True,
    }

###############################################################################
def summarize_array(
    values: np.ndarray,
    *,
    original_shape: list[int] | None = None,
    total_elements: int | None = None,
    statistics_sampled: bool = False,
    value_budget: int = MAX_SESSION_VALUE_ELEMENTS,
) -> tuple[dict[str, object], int]:
    array = np.asarray(values)
    shape = original_shape or [int(item) for item in array.shape]
    element_count = total_elements if total_elements is not None else int(array.size)
    flattened = array.reshape(-1)
    statistics = _numeric_statistics(flattened, total_elements=element_count, sampled=statistics_sampled)
    histogram = _histogram(flattened)

    if not statistics_sampled and element_count <= MAX_FULL_VALUE_ELEMENTS and element_count <= value_budget:
        selected = flattened
        storage_mode = "full"
    elif value_budget > 0:
        selected = flattened[: min(MAX_SAMPLE_VALUE_ELEMENTS, flattened.size, value_budget)]
        storage_mode = "sampled"
    else:
        selected = flattened[:0]
        storage_mode = "summary"

    used = int(selected.size)
    return (
        {
            "shape": shape,
            "dtype": str(array.dtype),
            "element_count": element_count,
            "storage_mode": storage_mode,
            "displayed_element_count": used,
            "values": [_json_value(item) for item in selected.tolist()],
            "value_indices": list(range(used)),
            "statistics": statistics,
            "histogram": histogram,
        },
        used,
    )

###############################################################################
def _sample_tensor_for_host(keras_module: Any, tensor: Any) -> tuple[np.ndarray, bool]:
    shape = _concrete_shape(tensor)
    count = _element_count(shape)
    flattened = keras_module.ops.reshape(tensor, (-1,))
    if count > MAX_HOST_STAT_ELEMENTS:
        flattened = flattened[:MAX_STAT_SAMPLE_ELEMENTS]
        sampled = True
    else:
        sampled = False
    return np.asarray(keras_module.ops.convert_to_numpy(flattened)), sampled

###############################################################################
def _tensor_summary(
    keras_module: Any,
    tensor: Any,
    *,
    port_id: str,
    value_budget: int,
) -> tuple[dict[str, object], int]:
    shape = _concrete_shape(tensor)
    total = _element_count(shape)
    host_values, sampled = _sample_tensor_for_host(keras_module, tensor)
    summary, used = summarize_array(
        host_values,
        original_shape=shape,
        total_elements=total,
        statistics_sampled=sampled,
        value_budget=value_budget,
    )
    summary["port_id"] = port_id
    return summary, used

###############################################################################
def _unavailable_node(node: Any, manifest: Any, index: int, reason: str) -> dict[str, object]:
    return {
        "node_id": node.id,
        "node_name": node.label or manifest.display_name,
        "node_type": node.definition_id,
        "topological_index": index,
        "inspectable": False,
        "reason": reason,
        "inputs": [],
        "outputs": [],
        "predecessor_node_ids": [],
        "successor_node_ids": [],
    }

###############################################################################
def run_keras_tensor_inspection(
    graph: ModelGraph,
    model: Any,
    model_input: Any,
    keras_module: Any,
    *,
    seed: int,
    checkpoint: Any,
) -> tuple[list[Any], dict[str, object]]:
    catalog = get_node_catalog()
    order = topological_order(graph)
    if len(order) > MAX_INSPECTION_NODES:
        output_values = model(model_input, training=False)
        outputs: list[Any] = list(cast(list[Any] | tuple[Any, ...], output_values)) if isinstance(output_values, (list, tuple)) else [output_values]
        return outputs, {
            "available": False,
            "reason_code": "inspection_graph_too_large",
            "reason": f"Tensor inspection is limited to {MAX_INSPECTION_NODES} nodes per execution session.",
            "seed": seed,
        }

    node_by_id = {node.id: node for node in graph.nodes}
    plans: list[dict[str, object]] = []
    inspection_symbols: list[Any] = []
    for index, node_id in enumerate(order):
        checkpoint()
        node = node_by_id[node_id]
        manifest = catalog.get(node.definition_id, node.definition_version)
        try:
            layer = model.get_layer(node.id)
        except (ValueError, KeyError):
            plans.append(
                _unavailable_node(
                    node,
                    manifest,
                    index,
                    "This node is implemented as a composite or backend operation without a stable node-level Keras output.",
                )
            )
            continue
        symbols = _flatten_symbolic(layer.output)
        ports = _enabled_output_ports(manifest, node.parameters)
        if not symbols or len(symbols) != len(ports):
            plans.append(
                _unavailable_node(
                    node,
                    manifest,
                    index,
                    "The runtime output structure does not match the node's enabled output ports.",
                )
            )
            continue
        start = len(inspection_symbols)
        inspection_symbols.extend(symbols)
        plans.append(
            {
                "node_id": node.id,
                "node_name": node.label or manifest.display_name,
                "node_type": node.definition_id,
                "topological_index": index,
                "inspectable": True,
                "reason": None,
                "symbol_start": start,
                "symbol_count": len(symbols),
                "port_ids": [port.id for port in ports],
                "inputs": [],
                "outputs": [],
                "predecessor_node_ids": [],
                "successor_node_ids": [],
            }
        )

    final_symbols = _flatten_symbolic(model.outputs)
    probe_outputs = [*final_symbols, *inspection_symbols]
    probe = keras_module.Model(inputs=model.inputs, outputs=probe_outputs, name="neuro_weave_tensor_probe")
    raw_values = probe(model_input, training=False)
    execution_values: list[Any] = list(cast(list[Any] | tuple[Any, ...], raw_values)) if isinstance(raw_values, (list, tuple)) else [raw_values]
    final_values: list[Any] = execution_values[: len(final_symbols)]
    inspected_values: list[Any] = execution_values[len(final_symbols) :]

    remaining_budget = MAX_SESSION_VALUE_ELEMENTS
    summaries_by_port: dict[tuple[str, str], dict[str, object]] = {}
    for plan in plans:
        if not plan["inspectable"]:
            continue
        start = int(cast(int, plan.pop("symbol_start")))
        count = int(cast(int, plan.pop("symbol_count")))
        port_ids = cast(list[str], plan.pop("port_ids"))
        output_summaries: list[dict[str, object]] = []
        for port_id, tensor in zip(port_ids, inspected_values[start : start + count], strict=True):
            checkpoint()
            summary, used = _tensor_summary(
                keras_module,
                tensor,
                port_id=port_id,
                value_budget=remaining_budget,
            )
            remaining_budget = max(remaining_budget - used, 0)
            output_summaries.append(summary)
            summaries_by_port[(cast(str, plan["node_id"]), port_id)] = summary
        plan["outputs"] = output_summaries

    predecessor_map: dict[str, list[str]] = {node_id: [] for node_id in order}
    successor_map: dict[str, list[str]] = {node_id: [] for node_id in order}
    for edge in graph.edges:
        if edge.source_node_id not in predecessor_map or edge.target_node_id not in predecessor_map:
            continue
        if edge.source_node_id not in predecessor_map[edge.target_node_id]:
            predecessor_map[edge.target_node_id].append(edge.source_node_id)
        if edge.target_node_id not in successor_map[edge.source_node_id]:
            successor_map[edge.source_node_id].append(edge.target_node_id)

    for plan in plans:
        node_id = cast(str, plan["node_id"])
        node = node_by_id[node_id]
        manifest = catalog.get(node.definition_id, node.definition_version)
        inputs: list[dict[str, object]] = []
        for edge in ordered_incoming_edges(graph, node_id, manifest, node.parameters):
            source = summaries_by_port.get((edge.source_node_id, edge.source_port_id))
            if source is None:
                inputs.append(
                    {
                        "node_id": edge.source_node_id,
                        "port_id": edge.source_port_id,
                        "target_port_id": edge.target_port_id,
                        "shape": None,
                        "dtype": None,
                    }
                )
                continue
            inputs.append(
                {
                    "node_id": edge.source_node_id,
                    "port_id": edge.source_port_id,
                    "target_port_id": edge.target_port_id,
                    "shape": source["shape"],
                    "dtype": source["dtype"],
                }
            )
        plan["inputs"] = inputs
        plan["predecessor_node_ids"] = predecessor_map[node_id]
        plan["successor_node_ids"] = successor_map[node_id]

    return final_values, {
        "available": True,
        "seed": seed,
        "initialization": "untrained_random_initialization",
        "representative_input": "synthetic_zero_filled",
        "training": "not_run",
        "execution_order": order,
        "nodes": plans,
        "limits": {
            "max_full_value_elements": MAX_FULL_VALUE_ELEMENTS,
            "max_sample_value_elements": MAX_SAMPLE_VALUE_ELEMENTS,
            "max_session_value_elements": MAX_SESSION_VALUE_ELEMENTS,
            "max_host_stat_elements": MAX_HOST_STAT_ELEMENTS,
        },
    }

###############################################################################
def native_tensor_inspection_unavailable() -> dict[str, object]:
    return {
        "available": False,
        "reason_code": "native_intermediate_inspection_unsupported",
        "reason": (
            "Node-level tensor inspection is not yet exposed for native-only execution profiles. "
            "The native validation path uses backend-specific generated execution and deterministic smoke-test state, "
            "so NeuroWeave does not present those values as normal initialized activations."
        ),
    }
