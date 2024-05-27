from typing import Any

import torch

from src.metrics.metric import Metric
from src.metrics.precision import Precision
from src.metrics.recall import Recall


class FBeta(Metric):
    """
    Implementation of `F-Beta` Metric.
    Basic RecSys metric that calculates precision of `Top-K` elements.

    Parameters
    ----------
    topk: `int`, required
        Top-K elements to take into account.
    beta: `float`, optional (default = 1.0)
        Beta value for F-Measure.
    """

    def __init__(self, topk: int, beta: float = 1.0) -> None:
        assert topk > 0, f"Invalid topk value: {topk}"
        self._topk = topk
        self._beta = beta
        self._precision = Precision(self._topk)
        self._recall = Recall(self._topk)
        self._total_f = self._total_count = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_f": self._total_f,
            "total_count": self._total_count,
            "precision": self._precision.state_dict(),
            "recall": self._recall.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._total_f, self._total_count = state_dict["total_f"], state_dict["total_count"]
        self._precision.load_state_dict(state_dict["precision"])
        self._recall.load_state_dict(state_dict["recall"])
        if self.accelerator is None:
            return
        self._total_f = self._total_f.to(self.accelerator.device)
        self._total_count = self._total_count.to(self.accelerator.device)

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> None:
        self._total_count += torch.tensor(target.size(0), device=output.device)
        self._total_f += self.compute(output, target).sum()

    def compute(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        precision = self._precision.compute(output, target)
        recall = self._recall.compute(output, target)
        return (
            (1.0 + self._beta**2)
            * precision
            * recall
            / (self._beta**2 * precision + recall + 1e-13)
        )

    def get_metric(self, reset: bool = False) -> torch.Tensor:
        metric = self._total_f / self._total_count
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        device = torch.device("cpu")
        if self.accelerator is not None:
            device = self.accelerator.device
        self._total_f = torch.tensor(0.0, device=device)
        self._total_count = torch.tensor(0.0, device=device)
