import scipy.sparse as sps
import torch


class Model(torch.nn.Module):
    def __init__(
        self,
        num_items: int,
        lambda_weight: float = 100.0,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self._num_items = num_items
        self._lambda_weight = lambda_weight
        self._threshold = threshold
        self.register_buffer("_item_matrix", torch.empty(num_items, num_items, dtype=torch.float))

    def forward(self, inputs: sps.csr_matrix | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.training:
            return {"logits": torch.einsum("bi,ij->bj", inputs["source"], self._item_matrix)}
        device = self._item_matrix.device
        X = (
            torch.sparse_csr_tensor(
                torch.from_numpy(inputs.indptr).long(),
                torch.from_numpy(inputs.indices).long(),
                torch.from_numpy(inputs.data),
                size=inputs.shape,
                dtype=torch.float,
                device=device,
            )
            .to_sparse_coo()
            .coalesce()
        )
        gram_matrix = torch.sparse.mm(X.transpose(0, 1), X)
        if self._threshold > 0:
            values = gram_matrix.values()
            values *= values.gt(self._threshold).float()
            gram_matrix = torch.sparse_coo_tensor(
                gram_matrix.indices(),
                gram_matrix.values(),
                gram_matrix.size(),
                dtype=torch.float,
                device=device,
            )
        gram_matrix += self._lambda_weight * torch.eye(gram_matrix.size(0)).to_sparse()
        gram_matrix = gram_matrix.to_dense()
        precision_matrix = torch.linalg.inv(gram_matrix)
        item_matrix = precision_matrix / (-torch.diag(precision_matrix))
        item_matrix.fill_diagonal_(0.0)
        self._item_matrix = item_matrix.float()
        return {}
