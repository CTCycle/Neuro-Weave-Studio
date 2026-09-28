from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from configurations.settings import Settings
from domain.enums import ExecutionBackend
from domain.graph import ModelGraph
from domain.runtime import RuntimeApi, WorkerConfiguration
from domain.tensor_inspection import TensorInspectionViewRequest
from models import tensor_inspection_runtime
from models.tensor_inspection_runtime import persist_inspection
from services.tensor_inspection_service import (
    TensorInspectionError,
    TensorInspectionService,
)


def inspection_graph() -> ModelGraph:
    return ModelGraph.model_validate(
        {
            "schema_version": 6,
            "nodes": [
                {
                    "id": "sequence_input",
                    "definition_id": "input.sequence",
                    "position": {"x": 0, "y": 0},
                    "parameters": {"shape": [2, 3], "dtype": "float32"},
                },
                {
                    "id": "sequence_layer",
                    "definition_id": "recurrent.simple_rnn",
                    "position": {"x": 300, "y": 0},
                    "parameters": {"units": 3, "return_sequences": True, "return_state": True},
                },
                {
                    "id": "sequence_output",
                    "definition_id": "output.sequence",
                    "position": {"x": 600, "y": 0},
                    "parameters": {},
                },
            ],
            "edges": [
                {
                    "id": "sequence_edge_1",
                    "source_node_id": "sequence_input",
                    "source_port_id": "output",
                    "target_node_id": "sequence_layer",
                    "target_port_id": "input",
                },
                {
                    "id": "sequence_edge_2",
                    "source_node_id": "sequence_layer",
                    "source_port_id": "output",
                    "target_node_id": "sequence_output",
                    "target_port_id": "input",
                },
            ],
        }
    )


def trace_values() -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    inputs = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    sequence = np.array(
        [[[1.0, np.nan, np.inf], [-1.0, 0.0, 2.0]]],
        dtype=np.float32,
    )
    hidden_state = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    return (
        {
            "sequence_input": {"output": inputs},
            "sequence_layer": {"output": sequence, "hidden_state": hidden_state},
            "sequence_output": {"output": sequence},
        },
        inputs,
    )


def create_bundle(output_root: Path, *, trace: dict[str, dict[str, np.ndarray]] | None = None) -> str:
    graph = inspection_graph()
    default_trace, inputs = trace_values()
    result = persist_inspection(
        graph,
        project_id=str(uuid4()),
        project_revision=4,
        backend=ExecutionBackend.PYTORCH,
        api=RuntimeApi.KERAS,
        worker=WorkerConfiguration(backend=ExecutionBackend.PYTORCH, device_id="cpu"),
        seed=73,
        input_description="Seeded batch-size-1 synthetic tensors.",
        trace=trace or default_trace,
        inputs=[inputs],
        output_root=output_root,
    )
    return str(result["session_id"])


def inspection_service(root: Path) -> TensorInspectionService:
    return TensorInspectionService(Settings(_env_file=None, tensor_inspection_root=root))


def test_persists_inspection_ports_statistics_and_bounded_rank_three_views(tmp_path: Path) -> None:
    root = tmp_path / "inspections"
    session_id = create_bundle(root)
    service = inspection_service(root)

    session = service.get(session_id)
    assert session.execution_backend is ExecutionBackend.PYTORCH
    assert session.runtime_api is RuntimeApi.KERAS
    assert session.seed == 73
    assert session.weight_source == "initialized_untrained"
    assert session.compiler_mode == "normal"
    assert [node.node_id for node in session.nodes] == [
        "sequence_input",
        "sequence_layer",
        "sequence_output",
    ]

    recurrent = service.node(session_id, "sequence_layer")
    assert recurrent.inspectable
    assert recurrent.predecessors == ["sequence_input"]
    assert recurrent.successors == ["sequence_output"]
    assert {tensor.port_id for tensor in recurrent.outputs} == {"output", "hidden_state"}
    assert recurrent.outputs[0].storage_id != recurrent.outputs[0].id
    assert (root / session_id / f"{recurrent.outputs[0].storage_id}.npy").is_file()
    assert recurrent.inputs[0].storage_id == session.nodes[0].outputs[0].storage_id

    output = next(tensor for tensor in recurrent.outputs if tensor.port_id == "output")
    stats = output.statistics
    assert stats is not None
    assert stats.count == 6
    assert stats.nan_count == 1
    assert stats.infinite_count == 1
    assert stats.minimum == -1
    assert stats.maximum == 2
    assert stats.mean == pytest.approx(0.5)
    assert len(output.histogram.counts) == 64
    assert sum(output.histogram.counts) == 4

    view = service.view(
        session_id,
        TensorInspectionViewRequest(
            tensor_id=output.id,
            display_axes=[1, 2],
            fixed_indices={0: 0},
            limit=64,
        ),
    )
    assert view.shape == [1, 2, 3]
    assert view.display_axes == [1, 2]
    assert view.fixed_indices == {0: 0}
    assert view.view_shape == [2, 3]
    assert view.total_values == 6
    assert view.values == [[1.0, "NaN", "Infinity"], [-1.0, 0.0, 2.0]]

    with pytest.raises(TensorInspectionError, match="outside axis") as error:
        service.view(
            session_id,
            TensorInspectionViewRequest(
                tensor_id=output.id,
                display_axes=[1, 2],
                fixed_indices={0: 1},
            ),
        )
    assert error.value.code == "slice_out_of_bounds"

    service.delete(session_id)
    assert not (root / session_id).exists()
    with pytest.raises(TensorInspectionError) as deleted:
        service.get(session_id)
    assert deleted.value.code == "session_expired"


def test_bounds_large_tensor_storage_to_sampled_or_summary_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tensor_inspection_runtime, "MAX_EXACT_ELEMENTS", 2)
    monkeypatch.setattr(tensor_inspection_runtime, "MAX_SAMPLE_ELEMENTS", 4)
    monkeypatch.setattr(tensor_inspection_runtime, "MAX_TENSOR_BYTES", 16)
    monkeypatch.setattr(tensor_inspection_runtime, "MAX_SESSION_BYTES", 512)

    sampled_root = tmp_path / "sampled"
    sampled_id = create_bundle(sampled_root)
    sampled_service = inspection_service(sampled_root)
    sampled = sampled_service.node(sampled_id, "sequence_layer")
    output = next(tensor for tensor in sampled.outputs if tensor.port_id == "output")
    assert output.storage_mode == "sampled"
    assert output.value_count == 4
    assert output.statistics is not None and output.statistics.mode == "sampled"
    sampled_view = sampled_service.view(
        sampled_id,
        TensorInspectionViewRequest(
            tensor_id=output.id,
            display_axes=[1, 2],
            fixed_indices={0: 0},
            limit=8,
        ),
    )
    assert sampled_view.mode == "sampled"
    assert sampled_view.display_axes == [0]
    assert sampled_view.fixed_indices == {}
    assert sampled_view.view_shape == [4]
    assert sampled_view.total_values == 4

    monkeypatch.setattr(tensor_inspection_runtime, "MAX_SESSION_BYTES", 1)
    summary_root = tmp_path / "summary"
    summary_id = create_bundle(summary_root)
    summary_service = inspection_service(summary_root)
    summary = summary_service.node(summary_id, "sequence_layer")
    summary_output = next(tensor for tensor in summary.outputs if tensor.port_id == "output")
    assert summary_output.storage_mode == "summary_only"
    assert summary_output.value_count == 0
    with pytest.raises(TensorInspectionError) as payload_error:
        summary_service.view(summary_id, TensorInspectionViewRequest(tensor_id=summary_output.id))
    assert payload_error.value.code == "payload_limit"
    assert payload_error.value.status_code == 413


def test_cleanup_expires_old_sessions_and_limits_retained_sessions(tmp_path: Path) -> None:
    root = tmp_path / "inspections"
    session_ids = [create_bundle(root) for _ in range(4)]
    first_session = root / session_ids[0] / "session.json"
    payload = json.loads(first_session.read_text(encoding="utf-8"))
    payload["expires_at"] = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
    first_session.write_text(json.dumps(payload), encoding="utf-8")

    service = inspection_service(root)
    assert len([path for path in root.iterdir() if path.is_dir()]) == 3
    assert not (root / session_ids[0]).exists()

    retained_id = session_ids[-1]
    retained_session = root / retained_id / "session.json"
    payload = json.loads(retained_session.read_text(encoding="utf-8"))
    payload["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    retained_session.write_text(json.dumps(payload), encoding="utf-8")
    service.cleanup()
    assert not (root / retained_id).exists()
