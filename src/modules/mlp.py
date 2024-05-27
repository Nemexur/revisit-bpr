from typing import Callable

import torch
import torch.nn.functional as F

from src.modules.activation import Activation

Act = Activation | Callable[[torch.Tensor], torch.Tensor]


class MLP(torch.nn.Module):
    def __init__(
        self,
        linears: list[torch.nn.Linear],
        activations: Act | list[Act] = F.relu,
        dropouts: float | list[float] = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(activations, list):
            activations = [activations] * len(linears)
        if not isinstance(dropouts, list):
            dropouts = [dropouts] * len(linears)
        self._input_size = linears[0].in_features
        self._output_size = linears[-1].out_features
        self._linears = torch.nn.ModuleList(linears)
        self._dropouts = torch.nn.ModuleList(
            [torch.nn.Dropout(p) if p > 0 else torch.nn.Identity() for p in dropouts]
        )
        self._activations = activations

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        for linear, act, dropout in zip(
            self._linears, self._activations, self._dropouts, strict=True
        ):
            t = dropout(act(linear(t)))
        return t

    def input_size(self) -> int:
        return self._input_size

    def output_size(self) -> int:
        return self._output_size
