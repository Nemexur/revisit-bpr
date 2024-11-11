import torch


class MultinomialLoss(torch.nn.Module):
    def __init__(self, size_average: bool = False) -> None:
        super().__init__()
        self.size_average = size_average

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        # logits, target, weights ~ (batch size, num classes)
        if weights is None:
            weights = torch.ones_like(logits)
        # Multiply with weights to discard padding
        log_probs = torch.log_softmax(logits, dim=-1) * weights
        mult_loss = -torch.einsum("bc,bc->b", log_probs, target)
        return mult_loss.mean() if self.size_average else mult_loss
