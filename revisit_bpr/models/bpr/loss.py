import torch
import torch.nn.functional as F


class Loss(torch.nn.Module):
    """
    The BPR pairwise loss.

    Parameters
    ----------
    size_average : `bool`, optional (default = True)
        By default, the losses are averaged over each loss element in the batch. If the field is set to ``False``, the losses are returned for each instance in the batch instead.
    """

    def __init__(self, size_average: bool = True) -> None:
        super().__init__()
        self.size_average = size_average

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        scores = -F.logsigmoid(logits)
        return scores.mean() if self.size_average else scores
