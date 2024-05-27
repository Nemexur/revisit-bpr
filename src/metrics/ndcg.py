from typing import Any

import torch

from src.metrics.metric import Metric, prepare_target


def exp_dcg(tensor: torch.Tensor) -> torch.Tensor:
    """Calculate `Exponential` gain function for `NDCG` Metric."""
    gains = (2**tensor) - 1
    return gains / torch.log2(
        torch.arange(0, tensor.size(-1), dtype=torch.float, device=tensor.device) + 2.0
    )


def linear_dcg(tensor: torch.Tensor) -> torch.Tensor:
    """Calculate `Linear` gain function for `NDCG` Metric."""
    discounts = 1 / (
        torch.arange(0, tensor.size(-1), dtype=torch.float, device=tensor.device) + 1.0
    )
    discounts[0] = 1.0
    return tensor * discounts


class NDCG(Metric):
    """
    Implementation of `Normalized Discounter Cumulative Gain` Metric.
    Graded relevance as a measure of  usefulness, or gain, from examining a set of items.
    Gain may be reduced at lower ranks.

    Parameters
    ----------
    topk : `int`, required
        Top-K elements to take into account.
    gain_function : `str`, optional (default = `"exp"`)
        Pick the gain function for the ground truth labels.
        Two options:
        - exp
        - linear
    """

    def __init__(self, topk: int, gain_function: str = "exp") -> None:
        assert topk > 0, f"Invalid topk value: {topk}"
        assert gain_function in (
            "exp",
            "linear",
        ), f"Invalid gain_function value: {gain_function}"
        self._topk = topk
        self._total_ndcg = self._total_count = 0
        self._dcg = exp_dcg if gain_function == "exp" else linear_dcg

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_ndcg": self._total_ndcg,
            "total_count": self._total_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._total_ndcg, self._total_count = state_dict["total_ndcg"], state_dict["total_count"]
        if self.accelerator is None:
            return
        self._total_ndcg = self._total_ndcg.to(self.accelerator.device)
        self._total_count = self._total_count.to(self.accelerator.device)

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> None:
        self._total_count += torch.tensor(target.size(0), device=output.device)
        self._total_ndcg += self.compute(output, target).sum()

    def compute(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        topk = min(output.size(-1), self._topk)
        # target_sorted_by_output, ideal_target ~ (users, topk)
        target_sorted_by_output = prepare_target(output, target)[:, :topk]
        ideal_target = prepare_target(target, target)[:, :topk]
        prediction_dcg = self._dcg(target_sorted_by_output).sum(dim=-1)
        ideal_dcg = self._dcg(ideal_target).sum(dim=-1)
        # ndcg_score ~ (users)
        ndcg_score = torch.nan_to_num(prediction_dcg / ideal_dcg)
        return ndcg_score

    def get_metric(self, reset: bool = False) -> torch.Tensor:
        metric = self._total_ndcg / self._total_count
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        device = torch.device("cpu")
        if self.accelerator is not None:
            device = self.accelerator.device
        self._total_ndcg = torch.tensor(0.0, device=device)
        self._total_count = torch.tensor(0.0, device=device)
