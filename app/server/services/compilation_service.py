from __future__ import annotations

from common.catalogs.loader import NodeCatalog
from compilation.compiler import compile_to_ir
from domain.compilation import CompilationResult
from domain.enums import CodeTarget
from domain.graph import ModelGraph
from exporters.jax_exporter import JaxSourceExporter
from exporters.keras_exporter import KerasSourceExporter
from exporters.pytorch_exporter import PyTorchSourceExporter
from exporters.tensorflow_exporter import TensorFlowSourceExporter
from services.validation_service import ValidationService

###############################################################################
class CompilationService:

    # -------------------------------------------------------------------------
    def __init__(self, catalog: NodeCatalog, validator: ValidationService) -> None:
        self._catalog = catalog
        self._validator = validator
        self._exporters = {
            CodeTarget.KERAS: KerasSourceExporter(),
            CodeTarget.TENSORFLOW: TensorFlowSourceExporter(),
            CodeTarget.PYTORCH: PyTorchSourceExporter(),
            CodeTarget.JAX: JaxSourceExporter(),
        }

    # -------------------------------------------------------------------------
    def compile(self, graph: ModelGraph, code_target: CodeTarget) -> CompilationResult:
        validation = self._validator.validate(graph, code_target)
        if not validation.valid:
            return CompilationResult(
                valid=False,
                code_target=code_target,
                diagnostics=validation.diagnostics,
            )
        ir = compile_to_ir(
            graph,
            code_target,
            self._catalog,
            shape_summaries=validation.shape_summaries,
        )
        code = self._exporters[code_target].generate(graph, ir)
        return CompilationResult(
            valid=True,
            code_target=code_target,
            diagnostics=validation.diagnostics,
            code=code,
            ir=ir,
        )
