# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import numpy as np

from common.catalogs.loader import get_node_catalog
from domain.enums import ExecutionBackend
from domain.graph import ModelGraph
from domain.runtime import RuntimeApi
from domain.tensor_inspection import (
    DataMode,
    TensorDescriptor,
    TensorHistogram,
    TensorInspectionNode,
    TensorInspectionSession,
    TensorStatistics,
)
from validation.graph_utils import adjacency, incoming_edges, topological_order

MAX_TENSOR_BYTES = 32 * 1024 * 1024
MAX_SESSION_BYTES = 256 * 1024 * 1024
MAX_EXACT_ELEMENTS = 1_000_000
MAX_SAMPLE_ELEMENTS = 65_536
HISTOGRAM_BINS = 64

###############################################################################
def _utc_now() -> datetime:
    return datetime.now(UTC)

###############################################################################
def _as_numpy(value: Any) -> np.ndarray:
    detached = value.detach() if callable(getattr(value, "detach", None)) else value
    # Only PyTorch tensors need an explicit CPU hop here.  TensorFlow eager
    # tensors expose a deprecated ``cpu`` compatibility method, so probing
    # and calling it would emit warnings on every captured activation.
    module_name = type(detached).__module__
    on_cpu = (
        detached.cpu()
        if module_name.startswith("torch") and callable(getattr(detached, "cpu", None))
        else detached
    )
    if hasattr(on_cpu, "numpy") and callable(on_cpu.numpy):
        on_cpu = on_cpu.numpy()
    return np.asarray(on_cpu)

###############################################################################
def _safe_array(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array.dtype.kind in {"O", "U", "S", "V", "c"}:
        raise ValueError(f"unsupported_dtype: tensor datatype '{array.dtype}' is not supported")
    return array

###############################################################################
def _sample(array: np.ndarray) -> tuple[np.ndarray, DataMode]:
    flat = array.reshape(-1)
    if flat.size <= MAX_EXACT_ELEMENTS:
        return flat, "exact"
    if flat.size <= MAX_SAMPLE_ELEMENTS:
        return flat, "sampled"
    indexes = np.linspace(0, flat.size - 1, MAX_SAMPLE_ELEMENTS, dtype=np.int64)
    return flat[indexes], "sampled"

###############################################################################
def _statistics(array: np.ndarray) -> tuple[TensorStatistics, TensorHistogram]:
    bins: list[float]
    counts: list[int]
    sample, mode = _sample(array)
    numeric = sample.astype(np.float64, copy=False)
    finite = np.isfinite(numeric)
    finite_values = numeric[finite]
    nan_count = int(np.isnan(numeric).sum())
    infinite_count = int(np.isinf(numeric).sum())
    if finite_values.size:
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        mean = float(np.mean(finite_values))
        standard_deviation = float(np.std(finite_values))
        median = float(np.median(finite_values))
        zero_proportion = float(np.count_nonzero(finite_values == 0) / numeric.size)
        if minimum == maximum:
            bins = [minimum, maximum]
            counts = [int(finite_values.size)]
        else:
            counts_array, edges = np.histogram(finite_values, bins=HISTOGRAM_BINS)
            bins = [float(item) for item in edges.tolist()]
            counts = [int(item) for item in counts_array.tolist()]
    else:
        minimum = maximum = mean = standard_deviation = median = zero_proportion = None
        bins, counts = [], []
    return (
        TensorStatistics(
            mode=mode,
            # ``count`` describes the population used for these statistics.
            # The descriptor's ``element_count`` remains the full tensor size,
            # so sampled statistics must never look like exact full-tensor
            # statistics to API consumers.
            count=int(sample.size),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            standard_deviation=standard_deviation,
            median=median,
            zero_proportion=zero_proportion,
            nan_count=nan_count,
            infinite_count=infinite_count,
        ),
        TensorHistogram(mode=mode, bins=bins, counts=counts),
    )

###############################################################################
def _port_items(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(cast(list[Any] | tuple[Any, ...], value))
    return [value]

###############################################################################
def _trace_values(trace: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    raw = trace.get(node_id, {})
    if isinstance(raw, Mapping):
        typed_raw = cast(Mapping[Any, Any], raw)
        return {str(key): value for key, value in typed_raw.items()}
    return {"output": raw}

###############################################################################
def _depths(graph: ModelGraph, ordered: list[str]) -> dict[str, int]:
    depths = {node_id: 0 for node_id in ordered}
    for node_id in ordered:
        for target in adjacency(graph).get(node_id, []):
            depths[target] = max(depths.get(target, 0), depths[node_id] + 1)
    return depths

###############################################################################
def persist_inspection(
    graph: ModelGraph,
    *,
    project_id: str,
    project_revision: int,
    backend: ExecutionBackend,
    api: RuntimeApi,
    worker: Any,
    seed: int,
    input_description: str,
    trace: Mapping[str, Any],
    inputs: Sequence[Any],
    output_root: Path,
    compiler_mode: str = "normal",
) -> dict[str, object]:
    """Persist a bounded, self-contained inspection bundle outside the job workspace."""

    session_id = uuid4().hex
    staging = output_root / f".{session_id}.tmp"
    destination = output_root / session_id
    staging.mkdir(parents=True, exist_ok=False)
    try:
        catalog = get_node_catalog()
        nodes_by_id = {node.id: node for node in graph.nodes}
        ordered = topological_order(graph)
        depths = _depths(graph, ordered)
        incoming = incoming_edges(graph)
        stored_bytes = 0
        node_records: list[TensorInspectionNode] = []

        def store(value: Any, *, node_id: str, direction: str, port_id: str, item_index: int | None) -> TensorDescriptor:
            nonlocal stored_bytes
            array = _safe_array(value)
            tensor_id = f"{node_id}:{direction}:{port_id}:{item_index if item_index is not None else 0}"
            # Tensor ids are intentionally readable API identifiers, but they
            # contain separators that are not valid Windows filenames. Keep a
            # separate safe storage id so the filesystem bundle works on every
            # supported platform while preserving the public tensor id.
            storage_id = tensor_id.replace(":", "__")
            statistics, histogram = _statistics(array)
            storage_mode = "summary_only"
            value_count = 0
            if array.nbytes <= MAX_TENSOR_BYTES and stored_bytes + array.nbytes <= MAX_SESSION_BYTES:
                np.save(staging / f"{storage_id}.npy", array, allow_pickle=False)
                stored_bytes += int(array.nbytes)
                storage_mode = "full"
                value_count = int(array.size)
            else:
                sample, _ = _sample(array)
                # A full array can be too large for one tensor while its
                # bounded sample still fits the session budget. Store that
                # sample regardless of the original tensor size; otherwise a
                # large tensor needlessly degrades to metadata-only output.
                if stored_bytes + sample.nbytes <= MAX_SESSION_BYTES:
                    np.save(staging / f"{storage_id}.npy", sample, allow_pickle=False)
                    stored_bytes += int(sample.nbytes)
                    storage_mode = "sampled"
                    value_count = int(sample.size)
            descriptor = TensorDescriptor(
                id=tensor_id,
                storage_id=storage_id,
                node_id=node_id,
                direction=direction,  # type: ignore[arg-type]
                port_id=port_id,
                item_index=item_index,
                shape=[int(item) for item in array.shape],
                dtype=str(array.dtype),
                element_count=int(array.size),
                storage_mode=storage_mode,  # type: ignore[arg-type]
                value_count=value_count,
                statistics=statistics,
                histogram=histogram,
            )
            return descriptor

        output_refs: dict[tuple[str, str, int | None], TensorDescriptor] = {}
        for node_id in ordered:
            node = nodes_by_id[node_id]
            manifest = catalog.get(node.definition_id, node.definition_version)
            output_values = _trace_values(trace, node_id)
            output_descriptors: list[TensorDescriptor] = []
            for port in manifest.outputs:
                if port.id not in output_values:
                    continue
                for item_index, value in enumerate(_port_items(output_values[port.id])):
                    descriptor = store(value, node_id=node_id, direction="output", port_id=port.id, item_index=item_index if len(_port_items(output_values[port.id])) > 1 else None)
                    output_descriptors.append(descriptor)
                    output_refs[(node_id, port.id, item_index)] = descriptor
            input_descriptors: list[TensorDescriptor] = []
            for edge in incoming.get(node_id, []):
                source = output_refs.get((edge.source_node_id, edge.source_port_id, 0))
                if source is None:
                    continue
                input_descriptors.append(source.model_copy(update={"id": f"{node_id}:input:{edge.target_port_id}:{edge.id}", "node_id": node_id, "direction": "input", "port_id": edge.target_port_id, "storage_id": source.storage_id}))
            node_records.append(
                TensorInspectionNode(
                    node_id=node_id,
                    node_name=node.label or node.id,
                    node_type=node.definition_id,
                    topological_index=len(node_records),
                    depth=depths[node_id],
                    predecessors=[edge.source_node_id for edge in incoming.get(node_id, [])],
                    successors=adjacency(graph).get(node_id, []),
                    inspectable=bool(output_descriptors),
                    unavailable_reason=(
                        None
                        if output_descriptors
                        else "capture_unsupported: the selected runtime did not expose an intermediate output for this node"
                    ),
                    inputs=input_descriptors,
                    outputs=output_descriptors,
                )
            )

        for _index, value in enumerate(inputs):
            # Input nodes are already represented in the trace. This loop only validates
            # the synthetic tensors so a backend cannot silently return an unsupported type.
            _safe_array(value)

        now = _utc_now()
        session = TensorInspectionSession(
            id=session_id,
            project_id=UUID(project_id),
            project_revision=project_revision,
            execution_backend=backend,
            runtime_api=api,
            worker=worker,
            seed=seed,
            input_description=input_description,
            compiler_mode=("bypassed_for_inspection" if compiler_mode != "normal" else "normal"),
            created_at=now,
            expires_at=now + timedelta(seconds=600),
            nodes=node_records,
        )
        (staging / "session.json").write_text(
            json.dumps(session.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
        )
        os.replace(staging, destination)
        return {
            "session_id": session_id,
            "seed": seed,
            "expires_at": session.expires_at.isoformat(),
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
