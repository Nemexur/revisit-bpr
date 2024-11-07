from typing import Any, Callable
from abc import ABC, abstractmethod
from functools import wraps

from accelerate import Accelerator
import torch


class Metric(ABC):
    """Base class for all metrics."""

    @property
    def accelerator(self) -> Accelerator | None:
        if not hasattr(self, "_accelerator"):
            return None
        return self._accelerator

    def set_accelerator(self, value: Accelerator) -> None:
        self._accelerator = value

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> None:
        pass

    @abstractmethod
    def compute(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_metric(self, reset: bool = False) -> torch.Tensor:
        """Compute and return averaged metric. Optionally reset internal state if needed."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""
        pass


class MaskedMetric(Metric):
    """Interface for metrics with masked computation."""

    @abstractmethod
    def __call__(
        self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> None:
        pass

    @abstractmethod
    def compute(
        self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        pass


def sync_all_reduce(*attrs: Any) -> Callable:
    def wrapper(func: Callable) -> Callable:
        @wraps(func)
        def another_wrapper(self: Metric, *args: Any, **kwargs: Any) -> Callable:
            if not isinstance(self, Metric):
                raise RuntimeError(
                    "Decorator sync_all_reduce should be used on Metric class methods only"
                )
            if self.accelerator is None:
                raise RuntimeError("Decorator sync_all_reduce requires an instance of Acceleartor")
            sync = getattr(self, "_sync", False)
            for attr in attrs:
                if not sync:
                    continue
                op_kwargs = {}
                if ":" in attr:
                    attr, op = attr.split(":")
                    valid_ops = ("mean", "sum", "none")
                    if op not in valid_ops:
                        raise ValueError(
                            f"Reduction operation is not valid (expected : {valid_ops}, got: {op})"
                        )
                    op_kwargs["reduction"] = op
                t = getattr(self, attr, None)
                if t is None:
                    continue
                t = self.accelerator.reduce(t, **op_kwargs)
                setattr(self, attr, t)

            return func(self, *args, **kwargs)

        return another_wrapper

    return wrapper


def validate_metric_inputs(output: torch.Tensor, target: torch.Tensor) -> None:
    if output.size() != target.size():
        raise IndexError(
            "Different sizes in output and target tensors: "
            f"output - {output.size()}, target - {target.size()}."
        )
    if not (target.eq(0) | target.eq(1)).all():
        raise ValueError("Target contains values outside of 0 and 1.\nTarget:\n{target}")


def prepare_target(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    indices = torch.argsort(-output, dim=-1)
    sorted_target = torch.gather(target, index=indices, dim=-1)
    return sorted_target
