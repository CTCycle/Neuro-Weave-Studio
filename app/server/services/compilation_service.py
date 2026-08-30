from __future__ import annotations

from common.catalogs.loader import NodeCatalog
from compilation.compiler import compile_to_ir
from domain.compilation import CompilationResult
from domain.enums import ExecutionBackend
from domain.graph import ModelGraph
from exporters.keras_exporter import KerasSourceExporter
from exporters.native_jax_exporter import NativeJaxSourceExporter
from exporters.native_tensorflow_exporter import NativeTensorFlowSourceExporter
from exporters.pytorch_exporter import PyTorchSourceExporter
from services.validation_service import ValidationService


###############################################################################
class CompilationService:
    # -------------------------------------------------------------------------
    def __init__(self, catalog: NodeCatalog, validator: ValidationService) -> None:
        self._catalog = catalog
        self._validator = validator
        self._exporters = {
            (True, ExecutionBackend.TENSORFLOW): KerasSourceExporter(),
            (True, ExecutionBackend.PYTORCH): KerasSourceExporter(),
            (True, ExecutionBackend.JAX): KerasSourceExporter(),
            (False, ExecutionBackend.TENSORFLOW): NativeTensorFlowSourceExporter(),
            (False, ExecutionBackend.PYTORCH): PyTorchSourceExporter(),
            (False, ExecutionBackend.JAX): NativeJaxSourceExporter(),
        }

    # -------------------------------------------------------------------------
    def compile(
        self,
        graph: ModelGraph,
        execution_backend: ExecutionBackend,
        keras_enabled: bool,
    ) -> CompilationResult:
        validation = self._validator.validate(graph, execution_backend, keras_enabled)
        if not validation.valid:
            return CompilationResult(
                valid=False,
                execution_backend=execution_backend,
                keras_enabled=keras_enabled,
                diagnostics=validation.diagnostics,
            )
        ir = compile_to_ir(
            graph,
            execution_backend,
            keras_enabled,
            self._catalog,
            shape_summaries=validation.shape_summaries,
        )
        code = self._exporters[(keras_enabled, execution_backend)].generate(
            graph, ir, self._catalog
        )
        return CompilationResult(
            valid=True,
            execution_backend=execution_backend,
            keras_enabled=keras_enabled,
            diagnostics=validation.diagnostics,
            code=code,
            ir=ir,
        )
