from copy import deepcopy

from lightfm import LightFM
import scipy.sparse as sps
import torch


class Model(torch.nn.Module):
    def __init__(
        self,
        inner_model: LightFM,
        num_users: int,
        num_items: int,
        epochs: int,
        num_threads: int = 1,
    ) -> None:
        super().__init__()
        self._inner_model = inner_model
        self._epochs = epochs
        self._num_threads = num_threads
        self.register_buffer(
            "user_factors", torch.empty(num_users, inner_model.no_components, dtype=torch.float)
        )
        self.register_buffer(
            "item_factors", torch.empty(num_items, inner_model.no_components, dtype=torch.float)
        )
        self.register_buffer("user_biases", torch.empty(num_users, dtype=torch.float))
        self.register_buffer("item_biases", torch.empty(num_items, dtype=torch.float))

    def forward(self, inputs: sps.coo_matrix | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.training:
            logits = (
                torch.einsum(
                    "bh,b...h->b...",
                    self.user_factors[inputs["user"]],
                    self.item_factors[inputs["item"]],
                )
                + self.item_biases[inputs["item"]]
                + self.user_biases[inputs["user"]].unsqueeze(-1)
            )
            if (mask := inputs.get("mask")) is not None:
                logits.masked_fill_(mask.eq(0), -1e13)
            return {"logits": logits}
        self._inner_model.fit(
            deepcopy(inputs), epochs=self._epochs, num_threads=self._num_threads, verbose=True
        )
        self.user_factors = (
            torch.from_numpy(self._inner_model.user_embeddings).float().to(self.user_factors.device)
        )
        self.user_biases = (
            torch.from_numpy(self._inner_model.user_biases).float().to(self.user_biases.device)
        )
        self.item_factors = (
            torch.from_numpy(self._inner_model.item_embeddings).float().to(self.item_factors.device)
        )
        self.item_biases = (
            torch.from_numpy(self._inner_model.item_biases).float().to(self.item_biases.device)
        )
        return {}
