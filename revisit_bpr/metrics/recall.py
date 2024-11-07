from typing import Any

import torch

from revisit_bpr.metrics.metric import Metric, prepare_target, validate_metric_inputs


class Recall(Metric):
    """
    Implementation of `Recall` Metric.
    Basic RecSys metric that calculates recall of `Top-K` elements.

    Parameters
    ----------
    topk : `int`, required
        Top-K elements to take into account.
    """

    def __init__(self, topk: int) -> None:
        assert topk > 0, f"Invalid topk value: {topk}"
        self._topk = topk
        self._total_recall = self._total_count = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_recall": self._total_recall,
            "total_count": self._total_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._total_recall, self._total_count = (
            state_dict["total_recall"],
            state_dict["total_count"],
        )
        if self.accelerator is None:
            return
        self._total_recall = self._total_recall.to(self.accelerator.device)
        self._total_count = self._total_count.to(self.accelerator.device)

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> None:
        self._total_count += torch.tensor(target.size(0), device=output.device)
        self._total_recall += self.compute(output, target).sum()

    def compute(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        validate_metric_inputs(output, target)
        topk = min(output.size(-1), self._topk)
        # target_sorted_by_output ~ (users, topk)
        target_sorted_by_output = prepare_target(output, target)[:, :topk]
        # recall_score ~ (users)
        recall_score = torch.nan_to_num(target_sorted_by_output.sum(dim=-1) / target.sum(dim=-1))
        return recall_score

    def get_metric(self, reset: bool = False) -> torch.Tensor:
        metric = self._total_recall / self._total_count
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        device = torch.device("cpu")
        if self.accelerator is not None:
            device = self.accelerator.device
        self._total_recall = torch.tensor(0.0, device=device)
        self._total_count = torch.tensor(0.0, device=device)
