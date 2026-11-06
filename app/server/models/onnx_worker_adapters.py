# Copyright © 2026 Thomas Virdis
# Licensed under the MIT License.

"""PyTorch export adapter loaded only inside the selected worker process."""

from __future__ import annotations

from typing import Any

import torch


class NativeOnnxWrapper(torch.nn.Module):
    def __init__(self, model: Any, input_ids: list[str]) -> None:
        super().__init__()
        self.model = model
        self.input_ids = input_ids

    def forward(self, *values: Any) -> Any:
        model_input = (
            values[0]
            if len(values) == 1
            else dict(zip(self.input_ids, values, strict=True))
        )
        return self.model(model_input)
