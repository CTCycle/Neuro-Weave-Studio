from __future__ import annotations

import json
import math
import shutil
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np

from configurations.settings import Settings
from domain.tensor_inspection import (
    TensorDescriptor,
    TensorInspectionNode,
    TensorInspectionSession,
    TensorInspectionViewRequest,
    TensorInspectionViewResponse,
)

MAX_VIEW_PAYLOAD_BYTES = 256 * 1024


def _json_safe(value: Any) -> Any:
    """Keep NaN/Infinity values valid for strict JSON responses."""

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in cast(list[Any], value)]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in cast(tuple[Any, ...], value)]
    return value


class TensorInspectionError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int = 404) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(message)


class TensorInspectionService:
    def __init__(self, settings: Settings) -> None:
        self._root = settings.tensor_inspection_root
        self._ttl = settings.tensor_inspection_ttl_seconds
        self._root.mkdir(parents=True, exist_ok=True)
        self.cleanup()

    def cleanup(self) -> None:
        now = time.time()
        sessions: list[tuple[float, Path]] = []
        for path in self._root.iterdir():
            if not path.is_dir():
                continue
            try:
                payload = json.loads((path / "session.json").read_text(encoding="utf-8"))
                expires = datetime.fromisoformat(str(payload["expires_at"])).timestamp()
                if expires <= now:
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    sessions.append((expires, path))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                shutil.rmtree(path, ignore_errors=True)
        sessions.sort(key=lambda item: item[0], reverse=True)
        for _, path in sessions[3:]:
            shutil.rmtree(path, ignore_errors=True)

    def _path(self, session_id: str) -> Path:
        if not session_id or not session_id.isalnum() or len(session_id) > 64:
            raise TensorInspectionError("session_expired", "The tensor inspection session is unavailable")
        path = (self._root / session_id).resolve()
        if path.parent != self._root.resolve() or not path.is_dir():
            raise TensorInspectionError("session_expired", "The tensor inspection session is unavailable")
        return path

    def _load(self, session_id: str, *, touch: bool = True) -> tuple[Path, TensorInspectionSession]:
        path = self._path(session_id)
        try:
            session = TensorInspectionSession.model_validate_json(
                (path / "session.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise TensorInspectionError("session_expired", "The tensor inspection session is invalid") from exc
        if session.expires_at <= datetime.now(UTC):
            shutil.rmtree(path, ignore_errors=True)
            raise TensorInspectionError("session_expired", "The tensor inspection session has expired")
        if touch:
            session = session.model_copy(
                update={"expires_at": datetime.now(UTC) + timedelta(seconds=self._ttl)}
            )
            with suppress(OSError):
                (path / "session.json").write_text(
                    json.dumps(session.model_dump(mode="json")), encoding="utf-8"
                )
        return path, session

    def get(self, session_id: str) -> TensorInspectionSession:
        return self._load(session_id)[1]

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _descriptor(session: TensorInspectionSession, tensor_id: str) -> TensorDescriptor:
        for node in session.nodes:
            for descriptor in [*node.inputs, *node.outputs]:
                if descriptor.id == tensor_id or descriptor.storage_id == tensor_id:
                    return descriptor
        raise TensorInspectionError("slice_out_of_bounds", "The selected tensor is not in this session", status_code=422)

    def node(self, session_id: str, node_id: str) -> TensorInspectionNode:
        _, session = self._load(session_id)
        for node in session.nodes:
            if node.node_id == node_id:
                return node
        raise TensorInspectionError("node_uninspectable", "The selected node is not in this session", status_code=422)

    def view(self, session_id: str, request: TensorInspectionViewRequest) -> TensorInspectionViewResponse:
        path, session = self._load(session_id)
        descriptor = self._descriptor(session, request.tensor_id)
        value_path = path / f"{descriptor.storage_id}.npy"
        if not value_path.is_file():
            raise TensorInspectionError(
                "payload_limit",
                "This tensor is available as metadata and sampled summaries only; precise values exceeded the session limit",
                status_code=413,
            )
        array = np.load(value_path, allow_pickle=False)
        shape = list(descriptor.shape)
        if descriptor.storage_mode == "sampled":
            # A sampled payload is a flattened deterministic sample, not a
            # coordinate-preserving copy of the original tensor. Ignore any
            # axis/slice request from the full tensor and expose the sample as
            # a one-dimensional view instead of attempting to index it with
            # the original rank.
            shape = [int(array.size)]
        rank = len(shape)
        if descriptor.storage_mode == "sampled":
            axes = [0] if rank else []
            fixed = {}
        else:
            axes = request.display_axes or ([] if rank == 0 else [0] if rank == 1 else [rank - 2, rank - 1])
            fixed = {int(axis): int(index) for axis, index in request.fixed_indices.items()}
        if any(axis < 0 or axis >= rank for axis in axes) or len(axes) > 2 or len(set(axes)) != len(axes):
            raise TensorInspectionError("slice_out_of_bounds", "Display axes are outside the tensor rank", status_code=422)
        if any(axis < 0 or axis >= rank for axis in fixed):
            raise TensorInspectionError(
                "slice_out_of_bounds", "Fixed indices are outside the tensor rank", status_code=422
            )
        if any(axis in axes for axis in fixed):
            raise TensorInspectionError(
                "slice_out_of_bounds",
                "A display axis cannot also be fixed; choose another display axis",
                status_code=422,
            )
        for axis in range(rank):
            if axis in axes:
                continue
            index = fixed.get(axis, 0)
            if index < 0 or index >= shape[axis]:
                raise TensorInspectionError("slice_out_of_bounds", f"Index {index} is outside axis {axis}", status_code=422)
        slicer: list[int | slice] = [fixed.get(axis, 0) if axis not in axes else slice(None) for axis in range(rank)]
        selected = np.asarray(array[tuple(slicer)])
        original_display_order = [axis for axis in range(rank) if axis in axes]
        if len(axes) > 1 and original_display_order != axes:
            selected = np.transpose(
                selected,
                axes=[original_display_order.index(axis) for axis in axes],
            )
        total = int(selected.size)
        downsampled = False
        if total > request.limit and len(axes) == 2 and selected.ndim == 2 and request.limit > 256:
            # Preserve a useful 2D shape for heatmaps instead of returning the
            # first N cells as a misleading one-dimensional strip.  The
            # evenly-spaced indices are deterministic and bounded by the
            # caller's point limit (64x64 for the default heatmap request).
            rows, columns = (int(selected.shape[0]), int(selected.shape[1]))
            # Keep both dimensions within the 64x64 canvas bound used by the
            # frontend and guarantee that the product never exceeds the
            # requested point budget, including very thin or very wide arrays.
            axis_limit = max(1, int(np.sqrt(request.limit)))
            row_count = max(1, min(rows, axis_limit))
            column_count = max(1, min(columns, axis_limit, request.limit // row_count))
            row_indices = np.linspace(0, rows - 1, row_count, dtype=np.int64)
            column_indices = np.linspace(0, columns - 1, column_count, dtype=np.int64)
            selected = selected[np.ix_(row_indices, column_indices)]
            downsampled = True
            view_shape = [row_count, column_count]
            values = selected.tolist()
        else:
            offset = min(request.offset, total)
            flat = selected.reshape(-1)[offset : offset + request.limit]
            view_shape = list(selected.shape) if total <= request.limit else [int(flat.size)]
            values = flat.tolist() if total > request.limit else selected.tolist()
        values = _json_safe(values)
        if len(json.dumps(values, ensure_ascii=False, separators=(",", ":"))) > MAX_VIEW_PAYLOAD_BYTES:
            raise TensorInspectionError(
                "payload_limit",
                "The requested tensor view exceeds the 256 KiB response limit; reduce the point limit or slice dimensions",
                status_code=413,
            )
        return TensorInspectionViewResponse(
            tensor_id=request.tensor_id,
            shape=list(descriptor.shape),
            dtype=descriptor.dtype,
            display_axes=axes,
            fixed_indices=fixed,
            values=values,
            view_shape=view_shape,
            mode=(
                "sampled"
                if descriptor.storage_mode == "sampled"
                else "downsampled"
                if downsampled
                else "exact"
            ),
            total_values=total,
        )
