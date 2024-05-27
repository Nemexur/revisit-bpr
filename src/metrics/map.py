from typing import Any

import torch

from src.metrics.metric import Metric, prepare_target, validate_metric_inputs


class MAP(Metric):
    """
    Implementation of `Mean Average Precision` Metric.
    The precision metric summarizes the fraction of relevant items
    out of the whole the recommendation list.

    Parameters
    ----------
    topk : `int`, required
        Top-K elements to take into account.
    normalized : `bool`, optional (default = `True`)
        Whether to pick normalized over users implementation or not.
    """

    def __init__(self, topk: int, normalized: bool = True) -> None:
        assert topk > 0, f"Invalid topk value: {topk}"
        self._topk = topk
        self._total_map = self._total_count = 0
        self._normalized = normalized

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_map": self._total_map,
            "total_count": self._total_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._total_map, self._total_count = state_dict["total_map"], state_dict["total_count"]
        if self.accelerator is None:
            return
        self._total_map = self._total_map.to(self.accelerator.device)
        self._total_count = self._total_count.to(self.accelerator.device)

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> None:
        self._total_count += torch.tensor(target.size(0), device=output.device)
        self._total_map += self.compute(output, target).sum()

    def compute(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        validate_metric_inputs(output, target)
        topk = min(output.size(-1), self._topk)
        # target_sorted_by_output ~ (users, topk)
        target_sorted_by_output = prepare_target(output, target)[:, :topk]
        target_cumsum = target_sorted_by_output.cumsum(dim=-1)
        # topk_tensor ~ (topk)
        topk_tensor = torch.arange(0, topk, dtype=torch.long, device=output.device)
        # precisions ~ (users, topk)
        precisions = target_cumsum / (topk_tensor + 1.0)
        relevant_precisions = precisions * target_sorted_by_output
        # Add  normalization if needed
        denominator = (
            target.sum(dim=-1).clamp(max=topk)
            if self._normalized
            else target_sorted_by_output.sum(dim=-1)
        )
        # ap_score ~ (users)
        ap_score = torch.nan_to_num(relevant_precisions.sum(dim=-1) / denominator)
        return ap_score

    def get_metric(self, reset: bool = False) -> torch.Tensor:
        metric = self._total_map / self._total_count
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        device = torch.device("cpu")
        if self.accelerator is not None:
            device = self.accelerator.device
        self._total_map = torch.tensor(0.0, device=device)
        self._total_count = torch.tensor(0.0, device=device)
