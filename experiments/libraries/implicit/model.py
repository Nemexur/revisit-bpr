from copy import deepcopy

from implicit.recommender_base import RecommenderBase
import numpy as np
import scipy.sparse as sps
import torch


class Model(torch.nn.Module):
    def __init__(self, inner_model: RecommenderBase, num_users: int, num_items: int) -> None:
        super().__init__()
        self._inner_model = inner_model
        self.register_buffer(
            "user_factors", torch.empty(num_users, inner_model.factors + 1, dtype=torch.float)
        )
        self.register_buffer(
            "item_factors", torch.empty(num_items, inner_model.factors + 1, dtype=torch.float)
        )

    def forward(self, inputs: sps.csr_matrix | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.training:
            logits = torch.einsum(
                "bh,b...h->b...",
                self.user_factors[inputs["user"]],
                self.item_factors[inputs["item"]],
            )
            if (mask := inputs.get("mask")) is not None:
                logits.masked_fill_(mask.eq(0), -1e13)
            return {"logits": logits}
        self._inner_model.fit(deepcopy(inputs), show_progress=True)
        user_factors = (
            self._inner_model.user_factors.to_numpy()
            if not isinstance(self._inner_model.user_factors, np.ndarray)
            else self._inner_model.user_factors
        )
        item_factors = (
            self._inner_model.item_factors.to_numpy()
            if not isinstance(self._inner_model.item_factors, np.ndarray)
            else self._inner_model.item_factors
        )
        self.user_factors = torch.from_numpy(user_factors).float().to(self.user_factors.device)
        self.item_factors = torch.from_numpy(item_factors).float().to(self.item_factors.device)
        return {}
