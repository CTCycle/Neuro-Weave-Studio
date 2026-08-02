from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from neuro_weave.domain.enums import DataType


class SymbolicDimension(BaseModel):
    name: str = Field(min_length=1, max_length=64)


Dimension = int | SymbolicDimension | None


class TensorSpec(BaseModel):
    shape: list[Dimension] = Field(default_factory=list)
    dtype: DataType = DataType.FLOAT32
    name: str | None = None

    @model_validator(mode="after")
    def validate_dimensions(self) -> "TensorSpec":
        for dimension in self.shape:
            if isinstance(dimension, int) and dimension <= 0:
                raise ValueError("Concrete tensor dimensions must be positive")
        return self

    @property
    def rank(self) -> int:
        return len(self.shape)

    def compatible_with(self, other: "TensorSpec") -> bool:
        if self.rank != other.rank:
            return False
        return all(
            left is None or right is None or left == right
            for left, right in zip(self.shape, other.shape, strict=True)
        )
