from typing import Any
from functools import reduce
from operator import mul

from einops import rearrange
import torch

from revisit_bpr.metrics.metric import MaskedMetric


class RocAucOne(MaskedMetric):
    def __init__(self) -> None:
        self._total_auc = self._total_count = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_auc": self._total_auc,
            "total_count": self._total_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._total_auc, self._total_count = state_dict["total_auc"], state_dict["total_count"]
        if self.accelerator is None:
            return
        self._total_auc = self._total_auc.to(self.accelerator.device)
        self._total_count = self._total_count.to(self.accelerator.device)

    def __call__(
        self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> None:
        if mask is None:
            mask = torch.ones_like(target)
        self._total_count += torch.tensor(target.size(0), device=output.device)
        self._total_auc += self.compute(output, target, mask).sum()

    def compute(
        self, output: torch.Tensor, _: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(output)
        # It is implied that the first item in each sample is related to positive item
        mask = mask[:, 1:]
        true_output, neg_output = output[:, 0], output[:, 1:]
        # score ~ (batch size, num neg items)
        score = (true_output.unsqueeze(-1) > neg_output).float()
        score[mask.eq(0)] = 0.0
        return score.sum(dim=-1) / mask.sum(dim=-1)

    def get_metric(self, reset: bool = False) -> torch.Tensor:
        metric = self._total_auc / self._total_count
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        device = torch.device("cpu")
        if self.accelerator is not None:
            device = self.accelerator.device
        self._total_auc = torch.tensor(0.0, device=device)
        self._total_count = torch.tensor(0.0, device=device)


class RocAucMany(MaskedMetric):
    def __init__(self) -> None:
        self._total_auc = self._total_count = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_auc": self._total_auc,
            "total_count": self._total_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._total_auc, self._total_count = state_dict["total_auc"], state_dict["total_count"]
        if self.accelerator is None:
            return
        self._total_auc = self._total_auc.to(self.accelerator.device)
        self._total_count = self._total_count.to(self.accelerator.device)

    def __call__(
        self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> None:
        if mask is None:
            mask = torch.ones_like(target)
        self._total_count += torch.tensor(target.size(0), device=output.device)
        self._total_auc += self.compute(output, target, mask).sum()

    def compute(
        self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(output)
        # score ~ (batch size, num items, num items)
        score = (
            rearrange(output, "batch items -> batch items ()")
            > rearrange(output, "batch items -> batch () items")
        ).float()
        # Discard nonpositive items from comparison
        score.masked_fill_(target.eq(0).unsqueeze(-1), 0.0)
        # Discard positive items and padding from comparison
        score.masked_fill_(
            rearrange(torch.logical_or(target.ne(0), mask.eq(0)), "batch items -> batch () items"),
            0.0,
        )
        # Discard padded items from valid negatives
        neg_items_mask = torch.logical_and(target.eq(0), mask.ne(0)).float()
        num_pairs = target.sum(dim=-1) * neg_items_mask.sum(dim=-1)
        return torch.einsum("b...->b", score) / num_pairs

    def get_metric(self, reset: bool = False) -> torch.Tensor:
        metric = self._total_auc / self._total_count
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        device = torch.device("cpu")
        if self.accelerator is not None:
            device = self.accelerator.device
        self._total_auc = torch.tensor(0.0, device=device)
        self._total_count = torch.tensor(0.0, device=device)


class RocAucManySlow(MaskedMetric):
    def __init__(self) -> None:
        self._total_auc = self._total_count = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_auc": self._total_auc,
            "total_count": self._total_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._total_auc, self._total_count = state_dict["total_auc"], state_dict["total_count"]
        if self.accelerator is None:
            return
        self._total_auc = self._total_auc.to(self.accelerator.device)
        self._total_count = self._total_count.to(self.accelerator.device)

    def __call__(
        self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> None:
        if mask is None:
            mask = torch.ones_like(target)
        self._total_count += torch.tensor(target.size(0), device=output.device)
        self._total_auc += self.compute(output, target, mask).sum()

    def compute(
        self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(output)
        batch_size = output.size(0)
        result = torch.empty(batch_size, dtype=torch.float, device=output.device)
        for idx in range(batch_size):
            output_idx, target_idx, mask_idx = output[idx], target[idx], mask[idx]
            score = (
                rearrange(output_idx[target_idx.ne(0)], "pos -> pos ()")
                > rearrange(
                    output_idx[torch.logical_and(target_idx.eq(0), mask_idx.ne(0))],
                    "neg -> () neg",
                )
            ).float()
            result[idx] = score.sum() / reduce(mul, score.size())
        return result

    def get_metric(self, reset: bool = False) -> torch.Tensor:
        metric = self._total_auc / self._total_count
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        device = torch.device("cpu")
        if self.accelerator is not None:
            device = self.accelerator.device
        self._total_auc = torch.tensor(0.0, device=device)
        self._total_count = torch.tensor(0.0, device=device)
