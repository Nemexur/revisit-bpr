from typing import Any
from abc import ABC, abstractmethod


class BaseScheduler(ABC):
    @abstractmethod
    def weight(self) -> float:
        return

    @abstractmethod
    def step(self) -> None:
        return

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


class Constant(BaseScheduler):
    def __init__(self, weight: float = 1.0) -> None:
        self._weight = weight

    def weight(self) -> float:
        return self._weight

    def step(self) -> None:
        return

    def state_dict(self) -> dict[str, Any]:
        return {"weight": self._weight}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._weight = state_dict["weight"]


class Linear(BaseScheduler):
    def __init__(
        self,
        annealing_steps: int,
        zero_weight_steps: int = 0,
        max_weight: float = 1.0,
    ) -> None:
        self._step = 0
        self._weight = 0.0
        self._max_weight = max_weight
        self._zero_weight_steps = zero_weight_steps
        self._annealing_steps = annealing_steps

    def weight(self) -> float:
        return self._weight

    def step(self) -> None:
        self._step += 1
        if self._zero_weight_steps > 0 and self._step <= self._zero_weight_steps:
            return
        self._weight = min(
            self._max_weight, (self._step - self._zero_weight_steps) / self._annealing_steps
        )

    def state_dict(self) -> dict[str, Any]:
        return {"weight": self._weight, "step": self._step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._weight, self._step = state_dict["weight"], state_dict["step"]
