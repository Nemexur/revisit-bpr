import torch
import torch.nn.functional as F


class Loss(torch.nn.Module):
    def __init__(self, size_average: bool = True) -> None:
        super().__init__()
        self.size_average = size_average

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        scores = -F.logsigmoid(logits)
        return scores.mean() if self.size_average else scores
