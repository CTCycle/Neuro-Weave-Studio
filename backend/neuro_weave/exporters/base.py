from __future__ import annotations

from abc import ABC, abstractmethod

from neuro_weave.domain.compilation import CanonicalModelIR
from neuro_weave.domain.graph import ModelGraph


class SourceExporter(ABC):
    @abstractmethod
    def generate(self, graph: ModelGraph, ir: CanonicalModelIR) -> str:
        raise NotImplementedError
