from typing import Any

import torch

from src.metrics.metric import Metric, prepare_target, validate_metric_inputs


class Precision(Metric):
    """
    Implementation of `Precision` Metric.
    Basic RecSys metric that calculates precision of `Top-K` elements.

    Parameters
    ----------
    topk : `int`, required
        Top-K elements to take into account.
    """

    def __init__(self, topk: int) -> None:
        assert topk > 0, f"Invalid topk value: {topk}"
        self._topk = topk
        self._total_precision = self._total_count = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_precision": self._total_precision,
            "total_count": self._total_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._total_precision, self._total_count = (
            state_dict["total_precision"],
            state_dict["total_count"],
        )
        if self.accelerator is None:
            return
        self._total_precision = self._total_precision.to(self.accelerator.device)
        self._total_count = self._total_count.to(self.accelerator.device)

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> None:
        self._total_count += torch.tensor(target.size(0), device=output.device)
        self._total_precision += self.compute(output, target).sum()

    def compute(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        validate_metric_inputs(output, target)
        topk = min(output.size(-1), self._topk)
        # target_sorted_by_output ~ (users, topk)
        target_sorted_by_output = prepare_target(output, target)[:, :topk]
        # precision_score ~ (users)
        precision_score = target_sorted_by_output.sum(dim=-1) / topk
        return precision_score

    def get_metric(self, reset: bool = False) -> torch.Tensor:
        metric = self._total_precision / self._total_count
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        device = torch.device("cpu")
        if self.accelerator is not None:
            device = self.accelerator.device
        self._total_precision = torch.tensor(0.0, device=device)
        self._total_count = torch.tensor(0.0, device=device)
