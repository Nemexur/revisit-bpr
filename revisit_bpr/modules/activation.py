# pyright: reportPrivateImportUsage=false

from typing import Callable
from functools import partial
import math

import torch
import torch.nn.functional as F

_ActivationFunc = Callable[[torch.Tensor], torch.Tensor]


@torch.jit.script
def gelu(t: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return t * 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))


@torch.jit.script
def swish(t: torch.Tensor) -> torch.Tensor:
    return t * torch.sigmoid(t)


class Activation:
    _registry: dict[str, Callable] = {}

    def __init__(self, name: str, *args, **kwargs) -> None:
        if name not in self._registry:
            raise ValueError(
                f"Unrecognized activation `{name}`. "
                f"Supported options: {', '.join(self._registry)}."
            )
        self._name = name
        self._act = partial(self._registry[name], *args, **kwargs)

    def __repr__(self) -> str:
        return f"Activation({self._name})"

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self._act(t)

    @staticmethod
    def register(name: str, act: _ActivationFunc) -> None:
        Activation._registry[name] = act


_activations = {
    "identity": lambda x: x,
    "gelu": gelu,
    "swish": swish,
    "tanh": F.tanh,
    "relu": F.relu,
    "relu6": F.relu6,
    "elu": F.elu,
    "prelu": F.prelu,
    "leaky_relu": F.leaky_relu,
    "threshold": F.threshold,
    "hardtanh": F.hardtanh,
    "sigmoid": F.sigmoid,
    "logsigmoid": F.logsigmoid,
    "softplus": F.softplus,
    "softshrink": F.softshrink,
    "softsign": F.softsign,
    "tanhshrink": F.tanhshrink,
}
for name, act_func in _activations.items():
    Activation.register(name, act_func)
