import torch
from einops import repeat
from revisit_bpr.models import BPR
from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def sample(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pass


class UniformSampler(Sampler):
    def __init__(self, num_items: int, neg_gen: torch.Generator) -> None:
        self._neg_gen = neg_gen
        self._item_weights = torch.ones(num_items, dtype=torch.float32, device=self._neg_gen.device)

    def sample(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        num = batch["item"].size(-1)
        return torch.multinomial(
            _sampling_weights(self._item_weights, batch["seen_items"]),
            num_samples=num,
            generator=self._neg_gen,
        )


class AdaptiveSampler(Sampler):
    def __init__(
        self,
        model: BPR,
        num_items: int,
        sampling_prob: float,
        neg_gen: torch.Generator,
        every: int,
    ) -> None:
        self._model = model
        self._sampling_prob = sampling_prob
        self._neg_gen = neg_gen
        self._every = every
        self._iteration_cnt = 0
        self._item_weights = torch.ones(num_items, dtype=torch.float32, device=neg_gen.device)

    def sample(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self._iteration_cnt += 1
        num = batch["item"].size(-1)
        user, seen_items = batch["user"], batch["seen_items"]
        features = self._model.logits_model.get_features()
        # num_notseen_items ~ (batch size, 1)
        num_notseen_items = (
            _sampling_weights(self._item_weights, seen_items).gt(0).sum(dim=-1, keepdim=True)
        )
        # factor, rank ~ (batch size, num samples)
        factor = torch.multinomial(
            features["user"].abs()[user] * self._factor_std,
            num_samples=num,
            generator=self._neg_gen,
        )
        # In this case rank is not greater than the number of not seen items
        rank = (
            torch.empty_like(factor)
            .geometric_(self._sampling_prob, generator=self._neg_gen)
            .clamp_(max=num_notseen_items)
        )
        # Consider not seen items only
        rank = torch.where(
            features["user"][user].gather(dim=-1, index=factor).gt(0),
            rank - 1,
            num_notseen_items - rank,
        )
        if (rank < 0).sum() > 0 or (rank > num_notseen_items.sub(1)).sum() > 0:
            print(
                "Detected out of bounds. Force clamp. "
                f"Lower 0: {(rank < 0).sum().item()}. "
                f"More than not seen: {(rank > num_notseen_items.sub(1)).sum().item()}"
            )
            rank = rank.clamp(min=torch.zeros_like(num_notseen_items), max=num_notseen_items - 1)
        # Just in case manually discard padding item
        seen_items = repeat(
            torch.hstack((seen_items, torch.zeros_like(rank))),
            "batch items -> batch num items",
            num=num,
        )
        output = (
            torch.argsort(
                -self._factor_to_items[factor].scatter(-1, index=seen_items, value=-1e13),
                dim=-1,
            )
            .gather(dim=-1, index=rank.unsqueeze(-1))
            .squeeze(-1)
        )
        if self._iteration_cnt % self._every == 0:
            self.update_stats()
        return output

    @torch.no_grad()
    def update_stats(self) -> None:
        features = self._model.logits_model.get_features()
        # item_by_factor ~ (factors, num items)
        self._factor_to_items = torch.einsum("if->fi", features["item"]).detach().clone()
        # factor_var ~ (1, factors)
        self._factor_std = features["item"][1:].detach().std(dim=0, keepdim=True)


def _sampling_weights(item_weights: torch.Tensor, seen_items: torch.Tensor) -> torch.Tensor:
    weights = repeat(item_weights, "items -> batch items", batch=seen_items.size(0)).scatter(
        dim=-1, index=seen_items, value=0.0
    )
    weights[:, 0] = 0.0  # Discard padding
    weights *= weights.sum(dim=-1, keepdim=True).reciprocal()
    return weights
