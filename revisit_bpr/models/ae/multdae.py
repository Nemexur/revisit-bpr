import torch
import torch.nn.functional as F

from revisit_bpr.modules import MLP
from revisit_bpr.models.ae.loss import MultinomialLoss


class Model(torch.nn.Module):
    def __init__(
        self, encoder: MLP, decoder: MLP, normalize: bool = True, dropout_prob: float = 0.0
    ) -> None:
        super().__init__()
        assert dropout_prob >= 0.0, "dropout should be greater or equal 0."
        self.variational = False
        self._encoder = encoder
        self._decoder = decoder
        self._normalize = normalize
        self._dropout = torch.nn.Dropout(dropout_prob) if dropout_prob > 0 else torch.nn.Identity()
        self._loss = MultinomialLoss(size_average=True)

    def _encode(self, source: torch.Tensor) -> torch.Tensor:
        # source ~ (batch size, num items)
        source = F.normalize(source, dim=-1, p=2) if self._normalize else source
        source = self._dropout(source)
        # output ~ (batch size, hidden size)
        return self._encoder(source)

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        # latent ~ (batch size, hidden size)
        # output ~ (batch size, num items)
        return self._decoder(latent)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # source, target ~ (batch size, num items)
        encoded = self._encode(inputs["source"])
        # logits, probs ~ (batch size, num items)
        logits = self._decode(encoded)
        output_dict = {
            "logits": logits,
            "probs": logits.softmax(dim=-1),
        }
        if (target := inputs.get("target")) is not None:
            output_dict["loss"] = self._loss(logits, target)
        return output_dict
