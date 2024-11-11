from einops import rearrange, reduce, repeat
import torch
from torch.nn import init

from revisit_bpr.models.bpr.loss import Loss


class BaseLogitModel(torch.nn.Module):
    def get_features(self) -> dict[str, torch.Tensor]:
        return {}


class Model(torch.nn.Module):
    """
    The BPR model.

    Parameters
    ----------
    logits_model : `BaseLogitModel`, required
        The model that produces logits for a user-item pair.
    reg_alphas : `dict[str, float] | None`, optional (default = None)
        Regularization alphas. It supports these keys: user, item, neg, all.
        User, item, and neg override all if present.
    fuse_forward : `bool`, optional (default = False)
        Simple optimization in the BPR model. It fuses the computation of user-item and user-neg logits.
    """

    def __init__(
        self,
        logits_model: BaseLogitModel,
        reg_alphas: dict[str, float] | None = None,
        fuse_forward: bool = False,
    ) -> None:
        super().__init__()
        self.logits_model = logits_model
        self._reg_alphas = reg_alphas or {}
        self._fuse_forward = fuse_forward
        self._loss = Loss(size_average=False)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # inputs.user ~ (batch size,)
        # inputs.item, inputs.neg ~ (batch size, num items)
        if not self.training:
            logits = self.logits_model(inputs["user"], inputs["item"], inputs)
            if (mask := inputs.get("mask")) is not None:
                logits.masked_fill_(mask.eq(0), -1e13)
            return {"logits": logits}
        if self._fuse_forward:
            items = torch.hstack((inputs["item"], inputs["neg"]))
            logits = self.logits_model(inputs["user"], items, inputs)
            logits_pos, logits_neg = (
                logits[:, : inputs["item"].size(-1)],
                logits[:, inputs["item"].size(-1) :],
            )
        else:
            logits_pos, logits_neg = (
                self.logits_model(inputs["user"], inputs["item"], inputs),
                self.logits_model(inputs["user"], inputs["neg"], inputs),
            )
        output_dict = {
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logits": logits_pos - logits_neg,
        }
        bpr_loss = output_dict["bpr_loss"] = self._loss(output_dict["logits"]).sum()
        l2_reg = output_dict["l2_reg"] = self.regularization(inputs).sum()
        output_dict["loss"] = bpr_loss + l2_reg
        return output_dict

    def regularization(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.logits_model.get_features()
        if len(features) == 0:
            return torch.tensor(0)
        all_reg, user_reg, item_reg, neg_reg = (
            self._reg_alphas.get("all"),
            self._reg_alphas.get("user"),
            self._reg_alphas.get("item"),
            self._reg_alphas.get("neg"),
        )
        if all(r is None for r in (all_reg, user_reg, item_reg, neg_reg)):
            return torch.tensor(0)
        if all_reg is not None:
            user_reg = item_reg = neg_reg = all_reg
        user_reg = user_reg or 0
        item_reg = item_reg or 0
        neg_reg = neg_reg or item_reg
        reg_term = (
            item_reg * torch.einsum("b...->b", features["item"][inputs["item"]].pow(2))
            + neg_reg * torch.einsum("b...->b", features["item"][inputs["neg"]].pow(2))
        )  # fmt: skip
        if (user_features := features.get("user")) is not None:
            reg_term += user_reg * torch.einsum("b...->b", user_features[inputs["user"]].pow(2))
        return reg_term.mean() / 2 if self._loss.size_average else reg_term / 2


class MF(BaseLogitModel):
    def __init__(
        self,
        user_emb: torch.nn.Embedding,
        item_emb: torch.nn.Embedding,
        item_bias: bool = False,
        user_bias: bool = False,
    ) -> None:
        super().__init__()
        self._user_emb = user_emb
        self._item_emb = item_emb
        if item_bias:
            self._item_bias = torch.nn.Parameter(torch.empty(item_emb.num_embeddings))
        else:
            self.register_parameter("_item_bias", None)
        if user_bias:
            self._user_bias = torch.nn.Parameter(torch.empty(user_emb.num_embeddings))
        else:
            self.register_parameter("_user_bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self._user_emb.weight.uniform_().add_(-0.5).div_(self._user_emb.embedding_dim)
            if self._user_emb.padding_idx is not None:
                self._user_emb.weight[self._user_emb.padding_idx].fill_(0)
        with torch.no_grad():
            self._item_emb.weight.uniform_().add_(-0.5).div_(self._item_emb.embedding_dim)
            if self._item_emb.padding_idx is not None:
                self._item_emb.weight[self._item_emb.padding_idx].fill_(0)
        if self._item_bias is not None:
            init.zeros_(self._item_bias)
        if self._user_bias is not None:
            init.zeros_(self._user_bias)

    def forward(
        self, user: torch.Tensor, item: torch.Tensor, _: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # user ~ (batch size,)
        # item ~ (batch size, ...)
        user_logits, item_logits = self._user_emb(user), self._item_emb(item)
        logits = torch.einsum("bh,b...h->b...", user_logits, item_logits)
        if self._item_bias is not None:
            logits += self._item_bias[item]
        if self._user_bias is not None:
            user_bias = self._user_bias[user]
            while user_bias.dim() < logits.dim():
                user_bias = user_bias.unsqueeze(-1)
            logits += user_bias
        return logits

    def get_features(self) -> dict[str, torch.Tensor]:
        return {
            "user": self._user_emb.weight,
            "item": self._item_emb.weight,
            "user_bias": self._user_bias,
            "item_bias": self._item_bias,
        }


class ItemKNN(BaseLogitModel):
    def __init__(
        self, num_items: int, hidden_dim: int, padding_idx: int = 0, bias: bool = False
    ) -> None:
        super().__init__()
        self._padding_idx = padding_idx
        self._weights = torch.nn.Parameter(torch.empty(num_items, hidden_dim))
        if bias:
            self._bias = torch.nn.Parameter(torch.empty(num_items))
        else:
            self.register_parameter("_bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.uniform_(self._weights)
        if self._bias is not None:
            init.zeros_(self._bias)
        with torch.no_grad():
            self._weights[self._padding_idx].fill_(0)

    def forward(
        self, _: torch.Tensor, item: torch.Tensor, other: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # user ~ (batch size,)
        # item ~ (batch size, num items)
        # seen_items ~ (batch size, seen items)
        seen_items = other["seen_items"]
        item_weights, seen_items_weights = self._weights[item], self._weights[seen_items]
        # Discard current items if present
        mask = reduce(
            rearrange(seen_items, "batch items -> batch () items").eq(item.unsqueeze(-1)),
            "batch items seen -> batch seen",
            reduction="max",
        )
        seen_items_weights[mask] = 0.0
        # logits ~ (batch size, num items)
        logits = torch.einsum("bih,bsh->bi", item_weights, seen_items_weights)
        if self._bias is not None:
            logits += self._bias[item]
        return logits

    def get_features(self) -> dict[str, torch.Tensor]:
        return {"item": self._weights, "bias": self._bias}


class FreeItemKNN(BaseLogitModel):
    def __init__(self, num_items: int, padding_idx: int = 0, bias: bool = False) -> None:
        super().__init__()
        self._padding_idx = padding_idx
        self._weights = torch.nn.Parameter(torch.empty(num_items, num_items))
        if bias:
            self._bias = torch.nn.Parameter(torch.empty(num_items))
        else:
            self.register_parameter("_bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.uniform_(self._weights)
        if self._bias is not None:
            init.zeros_(self._bias)
        with torch.no_grad():
            self._weights[self._padding_idx].fill_(0)
            if self._bias is not None:
                self._bias[self._padding_idx].fill_(0)

    def forward(
        self, _: torch.Tensor, item: torch.Tensor, other: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # item ~ (batch size, num items)
        other = other or {}
        if "seen_items" not in other:
            raise ValueError("seen_items should be present")
        # seen_items ~ (batch size, num items)
        seen_items = other["seen_items"]
        # Discard current items if present
        mask = reduce(
            rearrange(seen_items, "batch items -> batch () items").eq(item.unsqueeze(-1)),
            "batch items seen -> batch seen",
            reduction="max",
        )
        # Repeat is required because gather do not broadcast tensors
        # sim_matrix ~ (batch size, num items, seen items)
        sim_matrix = torch.gather(
            self._weights[item],
            index=repeat(seen_items, "batch seen -> batch items seen", items=item.size(-1)),
            dim=-1,
        )
        sim_matrix.masked_fill_(rearrange(mask, "batch seen -> batch () seen"), 0.0)
        # logits ~ (batch size, num items)
        logits = sim_matrix.sum(dim=-1)
        if self._bias is not None:
            logits += self._bias[item]
        return logits

    def get_features(self) -> dict[str, torch.Tensor]:
        return {"item": self._weights, "bias": self._bias}
