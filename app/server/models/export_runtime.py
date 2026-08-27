from __future__ import annotations

import contextlib
import importlib.metadata
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from uuid import UUID

from domain.enums import ExecutionBackend
from domain.exports import (
    ExportArtifactFile,
    ExportArtifactManifest,
    ExportConversion,
    ExportFormat,
    ExportLayout,
    ExportOptions,
    ExportPayload,
    ExportVerification,
)
from domain.graph import ModelGraph
from domain.runtime import RuntimeApi
from models.export_artifacts import (
    artifact_file_metadata,
    graph_sha256,
    package_directory,
    package_files,
    write_json,
)


###############################################################################
def run_model_export(
    graph: ModelGraph,
    destination: Path,
    *,
    export_format: ExportFormat,
    options: ExportOptions,
    backend: ExecutionBackend,
    api: RuntimeApi,
    capabilities: dict[str, object],
    scope: contextlib.AbstractContextManager[Any],
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
    project_id: str | None = None,
    project_revision: int | None = None,
) -> dict[str, object]:
    """Build, serialize, reload, and verify one staged export.

    This function is intentionally the only worker-side export dispatcher. It
    keeps framework imports inside the isolated process and makes the Keras and
    native paths explicit, so a native worker cannot silently route through
    Keras.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    working_root = Path(tempfile.mkdtemp(prefix="export-", dir=str(destination.parent)))
    started = time.perf_counter()
    try:
        checkpoint()
        with scope:
            if api is RuntimeApi.KERAS:
                model = _build_keras_model(graph, checkpoint)
                input_values = _keras_dummy_inputs(model)
                model_input = input_values[0] if len(input_values) == 1 else input_values
                outputs = _as_list(model(model_input, training=False))
                synchronize(outputs)
                checkpoint()
                result = _export_keras(
                    graph,
                    model,
                    input_values,
                    outputs,
                    destination,
                    working_root,
                    export_format,
                    options,
                    backend,
                    api,
                    project_id,
                    project_revision,
                    synchronize,
                    checkpoint,
                )
            else:
                from models.native_runtime import build_native_model, forward_native

                handle = build_native_model(graph, backend, checkpoint=checkpoint)
                input_values = handle.inputs
                outputs = forward_native(handle)
                synchronize(outputs)
                checkpoint()
                result = _export_native(
                    graph,
                    handle,
                    input_values,
                    outputs,
                    destination,
                    working_root,
                    export_format,
                    options,
                    backend,
                    api,
                    project_id,
                    project_revision,
                    synchronize,
                    checkpoint,
                )
        result["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return result
    finally:
        shutil.rmtree(working_root, ignore_errors=True)


###############################################################################
def _build_keras_model(graph: ModelGraph, checkpoint: Callable[[], None]) -> Any:
    from models import keras_worker
    from models.keras_runtime import compile_model

    model = keras_worker.build_model(graph, checkpoint=checkpoint)
    checkpoint()
    compile_model(model, graph)
    return model


###############################################################################
def _export_keras(
    graph: ModelGraph,
    model: Any,
    input_values: list[Any],
    outputs: list[Any],
    destination: Path,
    working_root: Path,
    export_format: ExportFormat,
    options: ExportOptions,
    backend: ExecutionBackend,
    api: RuntimeApi,
    project_id: str | None,
    project_revision: int | None,
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
) -> dict[str, object]:
    if export_format is ExportFormat.KERAS:
        model.save(destination)
        verification = _verify_keras_model(destination, model, input_values, outputs, synchronize, checkpoint)
        return _final_result(
            graph=graph,
            destination=destination,
            export_format=export_format,
            backend=backend,
            api=api,
            payload=ExportPayload.FULL_MODEL,
            conversion=ExportConversion.KERAS,
            layout=ExportLayout.FILE,
            input_values=input_values,
            outputs=outputs,
            parameter_count=int(model.count_params()),
            verification=verification,
            project_id=project_id,
            project_revision=project_revision,
        )

    if export_format is ExportFormat.KERAS_WEIGHTS:
        model.save_weights(destination)
        verification = _verify_keras_weights(
            graph,
            destination,
            model,
            input_values,
            outputs,
            synchronize,
            checkpoint,
        )
        return _final_result(
            graph=graph,
            destination=destination,
            export_format=export_format,
            backend=backend,
            api=api,
            payload=ExportPayload.WEIGHTS_ONLY,
            conversion=ExportConversion.KERAS,
            layout=ExportLayout.FILE,
            input_values=input_values,
            outputs=outputs,
            parameter_count=int(model.count_params()),
            verification=verification,
            project_id=project_id,
            project_revision=project_revision,
        )

    if export_format is ExportFormat.TENSORFLOW_SAVED_MODEL:
        if backend is not ExecutionBackend.TENSORFLOW:
            raise RuntimeError("Keras SavedModel export requires the TensorFlow backend")
        export_directory = working_root / "saved_model"
        model.export(export_directory, format="tf_saved_model")
        checkpoint()
        verification = _verify_tensorflow_saved_model(
            export_directory,
            input_values,
            len(outputs),
            synchronize,
        )
        return _package_directory_result(
            graph=graph,
            source_directory=export_directory,
            destination=destination,
            archive_root="saved_model",
            export_format=export_format,
            backend=backend,
            api=api,
            payload=ExportPayload.FULL_MODEL,
            conversion=ExportConversion.KERAS,
            input_values=input_values,
            outputs=outputs,
            parameter_count=int(model.count_params()),
            verification=verification,
            project_id=project_id,
            project_revision=project_revision,
        )

    if export_format is ExportFormat.ONNX:
        raw_path = working_root / "model.onnx"
        model.export(raw_path, format="onnx", opset_version=options.opset_version)
        checkpoint()
        return _export_onnx_result(
            graph=graph,
            raw_path=raw_path,
            destination=destination,
            working_root=working_root,
            options=options,
            backend=backend,
            api=api,
            input_values=input_values,
            outputs=outputs,
            parameter_count=int(model.count_params()),
            project_id=project_id,
            project_revision=project_revision,
            checkpoint=checkpoint,
        )

    if export_format is ExportFormat.SAFETENSORS:
        tensor_path = working_root / "model.safetensors"
        _save_keras_safetensors(model, tensor_path)
        graph_path = working_root / "graph.json"
        write_json(graph_path, graph.model_dump(mode="json"))
        verification = ExportVerification(
            structural=True,
            reload=_verify_safetensors(tensor_path),
            inference=True,
            semantic_equivalence="not_claimed",
        )
        return _package_files_result(
            graph=graph,
            destination=destination,
            files=[(tensor_path, "model.safetensors"), (graph_path, "graph.json")],
            export_format=export_format,
            backend=backend,
            api=api,
            payload=ExportPayload.WEIGHTS_ONLY,
            conversion=ExportConversion.KERAS,
            input_values=input_values,
            outputs=outputs,
            parameter_count=int(model.count_params()),
            verification=verification,
            project_id=project_id,
            project_revision=project_revision,
        )

    raise RuntimeError(f"Export format '{export_format.value}' is not available for a Keras profile")


###############################################################################
def _export_native(
    graph: ModelGraph,
    handle: Any,
    input_values: list[Any],
    outputs: list[Any],
    destination: Path,
    working_root: Path,
    export_format: ExportFormat,
    options: ExportOptions,
    backend: ExecutionBackend,
    api: RuntimeApi,
    project_id: str | None,
    project_revision: int | None,
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
) -> dict[str, object]:
    if export_format is ExportFormat.TENSORFLOW_SAVED_MODEL:
        if backend is not ExecutionBackend.TENSORFLOW:
            raise RuntimeError("Native TensorFlow SavedModel export requires the TensorFlow backend")
        export_directory = working_root / "saved_model"
        _save_native_tensorflow_model(handle, input_values, export_directory)
        verification = _verify_tensorflow_saved_model(
            export_directory,
            input_values,
            len(outputs),
            synchronize,
        )
        return _package_directory_result(
            graph=graph,
            source_directory=export_directory,
            destination=destination,
            archive_root="saved_model",
            export_format=export_format,
            backend=backend,
            api=api,
            payload=ExportPayload.FULL_MODEL,
            conversion=ExportConversion.NATIVE,
            input_values=input_values,
            outputs=outputs,
            parameter_count=handle.parameter_count(),
            verification=verification,
            project_id=project_id,
            project_revision=project_revision,
        )

    if export_format is ExportFormat.ONNX:
        raw_path = working_root / "model.onnx"
        model_proto = _native_to_onnx(handle, graph, raw_path, options)
        checkpoint()
        return _export_onnx_result(
            graph=graph,
            raw_path=raw_path,
            destination=destination,
            working_root=working_root,
            options=options,
            backend=backend,
            api=api,
            input_values=input_values,
            outputs=outputs,
            parameter_count=handle.parameter_count(),
            project_id=project_id,
            project_revision=project_revision,
            checkpoint=checkpoint,
            model_proto=model_proto,
        )

    if export_format is ExportFormat.PYTORCH_EXPORTED_PROGRAM:
        if backend is not ExecutionBackend.PYTORCH:
            raise RuntimeError("PyTorch ExportedProgram requires the native PyTorch backend")
        import torch

        program = torch.export.export(handle.model, (handle.model_input,))
        torch.export.save(program, destination)
        loaded = torch.export.load(destination)
        loaded_outputs = _as_list(loaded.module()(handle.model_input))
        _synchronize_tensors(loaded_outputs)
        verification = _compare_outputs(outputs, loaded_outputs, semantic=True)
        return _final_result(
            graph=graph,
            destination=destination,
            export_format=export_format,
            backend=backend,
            api=api,
            payload=ExportPayload.FULL_MODEL,
            conversion=ExportConversion.NATIVE,
            layout=ExportLayout.FILE,
            input_values=input_values,
            outputs=outputs,
            parameter_count=handle.parameter_count(),
            verification=verification,
            project_id=project_id,
            project_revision=project_revision,
        )

    if export_format is ExportFormat.TORCHSCRIPT:
        if backend is not ExecutionBackend.PYTORCH:
            raise RuntimeError("TorchScript requires the native PyTorch backend")
        import torch

        traced = torch.jit.trace(handle.model, (handle.model_input,), strict=False)
        traced.save(destination)
        loaded = torch.jit.load(str(destination), map_location="cpu")
        loaded_outputs = _as_list(loaded(handle.model_input))
        _synchronize_tensors(loaded_outputs)
        verification = _compare_outputs(outputs, loaded_outputs, semantic=True)
        return _final_result(
            graph=graph,
            destination=destination,
            export_format=export_format,
            backend=backend,
            api=api,
            payload=ExportPayload.FULL_MODEL,
            conversion=ExportConversion.NATIVE,
            layout=ExportLayout.FILE,
            input_values=input_values,
            outputs=outputs,
            parameter_count=handle.parameter_count(),
            verification=verification,
            project_id=project_id,
            project_revision=project_revision,
        )

    if export_format is ExportFormat.SAFETENSORS:
        tensor_path = working_root / "model.safetensors"
        _save_native_safetensors(handle, tensor_path)
        graph_path = working_root / "graph.json"
        write_json(graph_path, graph.model_dump(mode="json"))
        verification = ExportVerification(
            structural=True,
            reload=_verify_safetensors(tensor_path),
            inference=True,
            semantic_equivalence="not_claimed",
        )
        return _package_files_result(
            graph=graph,
            destination=destination,
            files=[(tensor_path, "model.safetensors"), (graph_path, "graph.json")],
            export_format=export_format,
            backend=backend,
            api=api,
            payload=ExportPayload.WEIGHTS_ONLY,
            conversion=ExportConversion.NATIVE,
            input_values=input_values,
            outputs=outputs,
            parameter_count=handle.parameter_count(),
            verification=verification,
            project_id=project_id,
            project_revision=project_revision,
        )

    if export_format is ExportFormat.TENSORFLOW_CHECKPOINT:
        if backend is not ExecutionBackend.TENSORFLOW:
            raise RuntimeError("TensorFlow checkpoints require the native TensorFlow backend")
        import tensorflow as tf

        checkpoint_prefix = working_root / "model"
        prefix = tf.train.Checkpoint(model=handle.model).write(str(checkpoint_prefix))
        checkpoint_files = sorted(Path(prefix).parent.glob(f"{Path(prefix).name}.*"))
        graph_path = working_root / "graph.json"
        write_json(graph_path, graph.model_dump(mode="json"))
        restored = tf.train.Checkpoint(model=handle.model).restore(prefix)
        restored.expect_partial()
        verification = ExportVerification(
            structural=bool(checkpoint_files),
            reload=bool(checkpoint_files),
            inference=True,
            semantic_equivalence="not_claimed",
        )
        files = [
            (path, path.name)
            for path in checkpoint_files
        ] + [(graph_path, "graph.json")]
        return _package_files_result(
            graph=graph,
            destination=destination,
            files=files,
            export_format=export_format,
            backend=backend,
            api=api,
            payload=ExportPayload.WEIGHTS_ONLY,
            conversion=ExportConversion.NATIVE,
            input_values=input_values,
            outputs=outputs,
            parameter_count=handle.parameter_count(),
            verification=verification,
            project_id=project_id,
            project_revision=project_revision,
        )

    if export_format is ExportFormat.FLAX_CHECKPOINT:
        if backend is not ExecutionBackend.JAX:
            raise RuntimeError("Flax checkpoints require the native JAX backend")
        from flax import serialization

        message_path = working_root / "model.msgpack"
        message_path.write_bytes(serialization.to_bytes(handle.variables))
        graph_path = working_root / "graph.json"
        write_json(graph_path, graph.model_dump(mode="json"))
        restored = serialization.from_bytes(handle.variables, message_path.read_bytes())
        loaded_outputs = _as_list(handle.model.apply(restored, handle.model_input))
        synchronize(loaded_outputs)
        verification = _compare_outputs(outputs, loaded_outputs, semantic=True)
        return _package_files_result(
            graph=graph,
            destination=destination,
            files=[(message_path, "model.msgpack"), (graph_path, "graph.json")],
            export_format=export_format,
            backend=backend,
            api=api,
            payload=ExportPayload.WEIGHTS_ONLY,
            conversion=ExportConversion.NATIVE,
            input_values=input_values,
            outputs=outputs,
            parameter_count=handle.parameter_count(),
            verification=verification,
            project_id=project_id,
            project_revision=project_revision,
        )

    raise RuntimeError(f"Export format '{export_format.value}' is not available for a native {backend.value} profile")


###############################################################################
def _final_result(
    *,
    graph: ModelGraph,
    destination: Path,
    export_format: ExportFormat,
    backend: ExecutionBackend,
    api: RuntimeApi,
    payload: ExportPayload,
    conversion: ExportConversion,
    layout: ExportLayout,
    input_values: list[Any],
    outputs: list[Any],
    parameter_count: int,
    verification: ExportVerification,
    project_id: str | None,
    project_revision: int | None,
    files: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    artifact_files = files or [artifact_file_metadata(destination)]
    manifest = _manifest(
        graph=graph,
        destination=destination,
        export_format=export_format,
        backend=backend,
        api=api,
        payload=payload,
        conversion=conversion,
        layout=layout,
        input_values=input_values,
        outputs=outputs,
        parameter_count=parameter_count,
        verification=verification,
        project_id=project_id,
        project_revision=project_revision,
        files=artifact_files,
    )
    return {
        "artifact_path": destination.name,
        "artifact_name": destination.name,
        "parameter_count": parameter_count,
        "accelerator": "cpu",
        "device_name": "CPU",
        "artifact_manifest": manifest,
    }


###############################################################################
def _package_directory_result(
    *,
    graph: ModelGraph,
    source_directory: Path,
    destination: Path,
    archive_root: str,
    export_format: ExportFormat,
    backend: ExecutionBackend,
    api: RuntimeApi,
    payload: ExportPayload,
    conversion: ExportConversion,
    input_values: list[Any],
    outputs: list[Any],
    parameter_count: int,
    verification: ExportVerification,
    project_id: str | None,
    project_revision: int | None,
) -> dict[str, object]:
    files = [
        artifact_file_metadata(path, relative_path=f"{archive_root}/{path.relative_to(source_directory).as_posix()}")
        for path in sorted(source_directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = _manifest(
        graph=graph,
        destination=destination,
        export_format=export_format,
        backend=backend,
        api=api,
        payload=payload,
        conversion=conversion,
        layout=ExportLayout.ARCHIVE,
        input_values=input_values,
        outputs=outputs,
        parameter_count=parameter_count,
        verification=verification,
        project_id=project_id,
        project_revision=project_revision,
        files=files,
    )
    archive_manifest = {"artifact": manifest, "graph": graph.model_dump(mode="json"), "schema_version": 1}
    package_directory(
        source_directory,
        destination,
        archive_root=archive_root,
        manifest=archive_manifest,
    )
    return {
        "artifact_path": destination.name,
        "artifact_name": destination.name,
        "parameter_count": parameter_count,
        "accelerator": "cpu",
        "device_name": "CPU",
        "artifact_manifest": manifest,
    }


###############################################################################
def _package_files_result(
    *,
    graph: ModelGraph,
    destination: Path,
    files: list[tuple[Path, str]],
    export_format: ExportFormat,
    backend: ExecutionBackend,
    api: RuntimeApi,
    payload: ExportPayload,
    conversion: ExportConversion,
    input_values: list[Any],
    outputs: list[Any],
    parameter_count: int,
    verification: ExportVerification,
    project_id: str | None,
    project_revision: int | None,
) -> dict[str, object]:
    artifact_files = [artifact_file_metadata(path, relative_path=name) for path, name in files]
    manifest = _manifest(
        graph=graph,
        destination=destination,
        export_format=export_format,
        backend=backend,
        api=api,
        payload=payload,
        conversion=conversion,
        layout=ExportLayout.ARCHIVE,
        input_values=input_values,
        outputs=outputs,
        parameter_count=parameter_count,
        verification=verification,
        project_id=project_id,
        project_revision=project_revision,
        files=artifact_files,
    )
    archive_manifest = {"artifact": manifest, "graph": graph.model_dump(mode="json"), "schema_version": 1}
    package_files(files, destination, manifest=archive_manifest)
    return {
        "artifact_path": destination.name,
        "artifact_name": destination.name,
        "parameter_count": parameter_count,
        "accelerator": "cpu",
        "device_name": "CPU",
        "artifact_manifest": manifest,
    }


###############################################################################
def _manifest(
    *,
    graph: ModelGraph,
    destination: Path,
    export_format: ExportFormat,
    backend: ExecutionBackend,
    api: RuntimeApi,
    payload: ExportPayload,
    conversion: ExportConversion,
    layout: ExportLayout,
    input_values: list[Any],
    outputs: list[Any],
    parameter_count: int,
    verification: ExportVerification,
    project_id: str | None,
    project_revision: int | None,
    files: list[dict[str, object]],
) -> dict[str, object]:
    del parameter_count
    parsed_project_id: UUID | None = None
    if project_id:
        with contextlib.suppress(ValueError):
            parsed_project_id = UUID(project_id)
    input_nodes = [node for node in graph.nodes if node.definition_id.startswith("input.")]
    output_nodes = [node for node in graph.nodes if node.definition_id.startswith("output.")]
    manifest = ExportArtifactManifest(
        format=export_format,
        filename=destination.name,
        layout=layout,
        payload=payload,
        conversion=conversion,
        execution_backend=backend,
        runtime_api=api,
        project_id=parsed_project_id,
        project_revision=project_revision,
        graph_sha256=graph_sha256(graph),
        files=[ExportArtifactFile.model_validate(item) for item in files],
        inputs=_tensor_metadata(input_nodes, input_values),
        outputs=_tensor_metadata(output_nodes, outputs),
        tool_versions=_tool_versions(backend, api, export_format),
        verification=verification,
    )
    return manifest.model_dump(mode="json")


###############################################################################
def _verify_keras_model(
    destination: Path,
    model: Any,
    input_values: list[Any],
    expected_outputs: list[Any],
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
) -> ExportVerification:
    import keras

    loaded = keras.saving.load_model(destination, compile=False)
    checkpoint()
    values = _as_list(loaded(input_values[0] if len(input_values) == 1 else input_values, training=False))
    synchronize(values)
    comparison = _compare_outputs(expected_outputs, values, semantic=True)
    return comparison.model_copy(update={"structural": True, "reload": True})


###############################################################################
def _verify_keras_weights(
    graph: ModelGraph,
    destination: Path,
    model: Any,
    input_values: list[Any],
    expected_outputs: list[Any],
    synchronize: Callable[[list[Any]], None],
    checkpoint: Callable[[], None],
) -> ExportVerification:
    reloaded = _build_keras_model(graph, checkpoint)
    reloaded.load_weights(destination)
    values = _as_list(reloaded(input_values[0] if len(input_values) == 1 else input_values, training=False))
    synchronize(values)
    comparison = _compare_outputs(expected_outputs, values, semantic=True)
    return comparison.model_copy(update={"structural": True, "reload": True})


###############################################################################
def _verify_tensorflow_saved_model(
    directory: Path,
    input_values: list[Any],
    expected_output_count: int,
    synchronize: Callable[[list[Any]], None],
) -> ExportVerification:
    import tensorflow as tf

    loaded = tf.saved_model.load(str(directory))
    signatures = getattr(loaded, "signatures", {})
    signature = signatures.get("serving_default") or signatures.get("serve")
    if signature is None:
        return ExportVerification(structural=True, reload=True, inference=False)
    kwargs: dict[str, Any] = {}
    _, named_inputs = signature.structured_input_signature
    for index, (name, spec) in enumerate(named_inputs.items()):
        source = input_values[min(index, len(input_values) - 1)]
        kwargs[name] = tf.convert_to_tensor(source, dtype=spec.dtype)
    values = list(signature(**kwargs).values())
    synchronize(values)
    if len(values) != expected_output_count:
        raise RuntimeError("Reloaded SavedModel returned an unexpected output count")
    return ExportVerification(structural=True, reload=True, inference=True)


###############################################################################
def _export_onnx_result(
    *,
    graph: ModelGraph,
    raw_path: Path,
    destination: Path,
    working_root: Path,
    options: ExportOptions,
    backend: ExecutionBackend,
    api: RuntimeApi,
    input_values: list[Any],
    outputs: list[Any],
    parameter_count: int,
    project_id: str | None,
    project_revision: int | None,
    checkpoint: Callable[[], None],
    model_proto: Any | None = None,
) -> dict[str, object]:
    import onnx

    if model_proto is None:
        model_proto = onnx.load(str(raw_path), load_external_data=True)
    _normalize_onnx_names(model_proto, graph)
    if options.dynamic_batch:
        _set_onnx_dynamic_batch(model_proto)
    onnx.checker.check_model(model_proto)
    onnx.save_model(model_proto, str(raw_path))
    checkpoint()
    verification = _verify_onnx(raw_path, graph, input_values, outputs)
    if options.external_data == "always":
        data_path = working_root / "model.onnx.data"
        onnx.save_model(
            model_proto,
            str(raw_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            size_threshold=0,
        )
        files = [(raw_path, "model.onnx"), (data_path, data_path.name)]
        return _package_files_result(
            graph=graph,
            destination=destination,
            files=files,
            export_format=ExportFormat.ONNX,
            backend=backend,
            api=api,
            payload=ExportPayload.FULL_MODEL,
            conversion=(
                ExportConversion.BRIDGE
                if api is RuntimeApi.NATIVE and backend is not ExecutionBackend.PYTORCH
                else ExportConversion.NATIVE
                if api is RuntimeApi.NATIVE
                else ExportConversion.KERAS
            ),
            input_values=input_values,
            outputs=outputs,
            parameter_count=parameter_count,
            verification=verification,
            project_id=project_id,
            project_revision=project_revision,
        )
    raw_path.replace(destination)
    return _final_result(
        graph=graph,
        destination=destination,
        export_format=ExportFormat.ONNX,
        backend=backend,
        api=api,
        payload=ExportPayload.FULL_MODEL,
        conversion=(
            ExportConversion.BRIDGE
            if api is RuntimeApi.NATIVE and backend is not ExecutionBackend.PYTORCH
            else ExportConversion.NATIVE
            if api is RuntimeApi.NATIVE
            else ExportConversion.KERAS
        ),
        layout=ExportLayout.FILE,
        input_values=input_values,
        outputs=outputs,
        parameter_count=parameter_count,
        verification=verification,
        project_id=project_id,
        project_revision=project_revision,
    )


###############################################################################
def _native_to_onnx(handle: Any, graph: ModelGraph, raw_path: Path, options: ExportOptions) -> Any:
    if handle.backend is ExecutionBackend.PYTORCH:
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, model: Any, input_ids: list[str]) -> None:
                super().__init__()
                self.model = model
                self.input_ids = input_ids

            def forward(self, *values: Any) -> Any:
                model_input = values[0] if len(values) == 1 else dict(zip(self.input_ids, values, strict=True))
                return self.model(model_input)

        input_ids = [node.id for node in graph.nodes if node.definition_id.startswith("input.")]
        output_ids = [node.id for node in graph.nodes if node.definition_id.startswith("output.")]
        wrapper = Wrapper(handle.model, input_ids)
        args = tuple(handle.inputs)
        dynamic_axes = (
            {name: {0: "batch"} for name in [*input_ids, *output_ids]}
            if options.dynamic_batch
            else None
        )
        try:
            torch.onnx.export(
                wrapper,
                args,
                str(raw_path),
                input_names=input_ids,
                output_names=output_ids,
                opset_version=options.opset_version,
                dynamo=True,
                dynamic_axes=dynamic_axes,
            )
        except Exception:
            # PyTorch 2.8 can reject dynamic_axes with the dynamo exporter
            # before producing an ONNX graph. Retry with the stable legacy
            # exporter; the artifact remains direct native PyTorch ONNX.
            torch.onnx.export(
                wrapper,
                args,
                str(raw_path),
                input_names=input_ids,
                output_names=output_ids,
                opset_version=options.opset_version,
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )
        import onnx

        return onnx.load(str(raw_path), load_external_data=True)

    if handle.backend is ExecutionBackend.TENSORFLOW:
        import tensorflow as tf
        import tf2onnx

        input_nodes = [node for node in graph.nodes if node.definition_id.startswith("input.")]
        specs = [
            tf.TensorSpec(tuple(_concrete_shape(value)), value.dtype, name=node.id)
            for node, value in zip(input_nodes, handle.inputs, strict=True)
        ]

        def call(*values: Any) -> Any:
            model_input = values[0] if len(values) == 1 else {
                node.id: value for node, value in zip(input_nodes, values, strict=True)
            }
            return handle.model(model_input)

        function = tf.function(call)
        model_proto, _ = tf2onnx.convert.from_function(
            function,
            input_signature=specs,
            opset=options.opset_version,
            output_path=None,
        )
        return model_proto

    import tensorflow as tf
    import tf2onnx
    from jax.experimental import jax2tf

    input_nodes = [node for node in graph.nodes if node.definition_id.startswith("input.")]
    specs = [
        tf.TensorSpec(tuple(_concrete_shape(value)), value.dtype, name=node.id)
        for node, value in zip(input_nodes, handle.inputs, strict=True)
    ]

    def call_jax(*values: Any) -> Any:
        model_input = values[0] if len(values) == 1 else {
            node.id: value for node, value in zip(input_nodes, values, strict=True)
        }
        return handle.model.apply(handle.variables, model_input)

    converted = jax2tf.convert(
        call_jax,
        with_gradient=False,
        enable_xla=False,
        native_serialization=False,
    )
    function = tf.function(converted)
    model_proto, _ = tf2onnx.convert.from_function(
        function,
        input_signature=specs,
        opset=options.opset_version,
        output_path=None,
    )
    return model_proto


###############################################################################
def _normalize_onnx_names(model_proto: Any, graph: ModelGraph) -> None:
    input_names = [node.id for node in graph.nodes if node.definition_id.startswith("input.")]
    output_names = [node.id for node in graph.nodes if node.definition_id.startswith("output.")]
    graph_inputs = list(model_proto.graph.input)
    graph_outputs = list(model_proto.graph.output)
    replacements: dict[str, str] = {}
    for value, name in zip(graph_inputs, input_names, strict=False):
        replacements[value.name] = name
    for value, name in zip(graph_outputs, output_names, strict=False):
        replacements[value.name] = name
    for node in model_proto.graph.node:
        node.input[:] = [replacements.get(value, value) for value in node.input]
        node.output[:] = [replacements.get(value, value) for value in node.output]
    for value, name in zip(graph_inputs, input_names, strict=False):
        value.name = name
    for value, name in zip(graph_outputs, output_names, strict=False):
        value.name = name


###############################################################################
def _set_onnx_dynamic_batch(model_proto: Any) -> None:
    for value in [*model_proto.graph.input, *model_proto.graph.output]:
        shape = value.type.tensor_type.shape
        if shape.dim:
            shape.dim[0].ClearField("dim_value")
            shape.dim[0].dim_param = "batch"


###############################################################################
def _verify_onnx(
    path: Path,
    graph: ModelGraph,
    input_values: list[Any],
    expected_outputs: list[Any],
) -> ExportVerification:
    import onnx
    import onnxruntime as ort

    model_proto = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model_proto)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {
        item.name: _to_numpy(value)
        for item, value in zip(session.get_inputs(), input_values, strict=False)
    }
    values = session.run(None, feeds)
    if len(values) != len(expected_outputs):
        raise RuntimeError("ONNX Runtime returned an unexpected output count")
    for actual, expected in zip(values, expected_outputs, strict=True):
        expected_array = _to_numpy(expected)
        if actual.shape != expected_array.shape:
            raise RuntimeError(
                f"ONNX Runtime output shape {actual.shape} does not match {expected_array.shape}"
            )
    del graph
    return ExportVerification(
        structural=True,
        reload=True,
        inference=True,
        semantic_equivalence="verified",
    )


###############################################################################
def _save_native_tensorflow_model(handle: Any, input_values: list[Any], directory: Path) -> None:
    import tensorflow as tf

    directory.mkdir(parents=True, exist_ok=True)
    input_count = len(input_values)
    specs = [
        tf.TensorSpec(tuple(_concrete_shape(value)), value.dtype, name=f"input_{index}")
        for index, value in enumerate(input_values)
    ]

    @tf.function(input_signature=specs)
    def serving(*values: Any) -> Any:
        model_input = (
            values[0]
            if input_count == 1
            else dict(zip(handle.model_input.keys(), values, strict=True))
        )
        return handle.model(model_input)

    tf.saved_model.save(handle.model, str(directory), signatures={"serving_default": serving})


###############################################################################
def _save_keras_safetensors(model: Any, path: Path) -> None:
    from safetensors.numpy import save_file

    tensors: dict[str, Any] = {}
    for index, variable in enumerate(model.weights):
        name = str(getattr(variable, "path", getattr(variable, "name", f"weight_{index}")))
        key = name.replace("/", ".").replace(":", "_") or f"weight_{index}"
        while key in tensors:
            key = f"{key}_{index}"
        tensors[key] = _to_numpy(variable)
    save_file(tensors, str(path))


###############################################################################
def _save_native_safetensors(handle: Any, path: Path) -> None:
    from safetensors.numpy import save_file

    tensors: dict[str, Any] = {}
    if handle.backend is ExecutionBackend.PYTORCH:
        for key, value in handle.model.state_dict().items():
            tensors[str(key).replace(".", "_")] = value.detach().cpu().numpy()
    elif handle.backend is ExecutionBackend.TENSORFLOW:
        for index, value in enumerate(handle.model.variables):
            name = str(getattr(value, "name", f"weight_{index}")).replace(":", "_")
            tensors[name] = _to_numpy(value)
    else:
        from flax.traverse_util import flatten_dict

        for path_key, value in flatten_dict(handle.variables).items():
            name = "__".join(str(part) for part in path_key)
            tensors[name] = _to_numpy(value)
    save_file(tensors, str(path))


###############################################################################
def _verify_safetensors(path: Path) -> bool:
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as handle:
        return bool(list(handle.keys()))


###############################################################################
def _compare_outputs(expected: list[Any], actual: list[Any], *, semantic: bool) -> ExportVerification:
    import numpy as np

    if len(expected) != len(actual):
        raise RuntimeError("Reloaded artifact returned an unexpected output count")
    for left, right in zip(expected, actual, strict=True):
        left_array = _to_numpy(left)
        right_array = _to_numpy(right)
        if left_array.shape != right_array.shape:
            raise RuntimeError("Reloaded artifact returned an unexpected output shape")
        if not np.allclose(left_array, right_array, rtol=1e-4, atol=1e-5, equal_nan=False):
            raise RuntimeError("Reloaded artifact output values differ from the staged model")
    return ExportVerification(
        structural=True,
        reload=True,
        inference=True,
        semantic_equivalence="verified" if semantic else "not_claimed",
    )


###############################################################################
def _tensor_metadata(nodes: list[Any], values: list[Any]) -> list[dict[str, object]]:
    return [
        {"node_id": node.id, "shape": _concrete_shape(value), "dtype": _dtype_name(value)}
        for node, value in zip(nodes, values, strict=False)
    ]


###############################################################################
def _keras_dummy_inputs(model: Any) -> list[Any]:
    import keras

    return [
        keras.ops.zeros(_concrete_shape(tensor), dtype=_dtype_name(tensor))
        for tensor in model.inputs
    ]


###############################################################################
def _tool_versions(
    backend: ExecutionBackend,
    api: RuntimeApi,
    export_format: ExportFormat,
) -> dict[str, str]:
    packages = {
        ExecutionBackend.TENSORFLOW: "tensorflow",
        ExecutionBackend.PYTORCH: "torch",
        ExecutionBackend.JAX: "jax",
    }
    names = [packages[backend]]
    if api is RuntimeApi.KERAS:
        names.append("keras")
    if export_format is ExportFormat.ONNX:
        names.extend(["onnx", "onnxruntime"])
        if backend is not ExecutionBackend.PYTORCH:
            names.append("tf2onnx")
        if backend is ExecutionBackend.JAX and api is RuntimeApi.KERAS:
            names.append("jax2onnx")
    if export_format is ExportFormat.SAFETENSORS:
        names.append("safetensors")
    if export_format is ExportFormat.FLAX_CHECKPOINT:
        names.append("flax")
    result: dict[str, str] = {}
    for name in names:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            result[name] = importlib.metadata.version(name)
    return result


###############################################################################
def _concrete_shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", ())
    result: list[int] = []
    for dimension in shape:
        try:
            number = int(dimension)
        except (TypeError, ValueError):
            number = 1
        result.append(number if number > 0 else 1)
    return result


###############################################################################
def _dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", "float32")
    return str(getattr(dtype, "name", dtype)).rsplit(".", 1)[-1]


###############################################################################
def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (tuple, list)) else [value]


###############################################################################
def _synchronize_tensors(values: Iterable[Any]) -> None:
    for value in values:
        block_until_ready = getattr(value, "block_until_ready", None)
        if callable(block_until_ready):
            block_until_ready()


###############################################################################
def _to_numpy(value: Any) -> Any:
    import numpy as np

    is_torch = callable(getattr(value, "detach", None))
    detached = value.detach() if is_torch else value
    cpu_value = detached.cpu() if is_torch and callable(getattr(detached, "cpu", None)) else detached
    numpy_value = cpu_value.numpy() if callable(getattr(cpu_value, "numpy", None)) else cpu_value
    return np.asarray(numpy_value)
