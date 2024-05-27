import torch


class Model(torch.nn.Module):
    def __init__(self, num_items: int) -> None:
        super().__init__()
        self.register_buffer("_item_counters", torch.zeros(num_items, dtype=torch.float))

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.training:
            return {"logits": self._item_counters[inputs["item"]]}
        items = inputs["item"].view(-1)
        mask = items.gt(0)
        if "mask" in inputs:
            mask = inputs["mask"].ne(0).view(-1)
        items = items[mask]
        self._item_counters.scatter_add_(
            dim=-1, index=items, src=torch.ones_like(items, dtype=torch.float)
        )
        return {"logits": self._item_counters[inputs["item"]]}
