from typing import NamedTuple

import torch
import torch.nn.functional as F

from src.models.ae.kl_scheduler import BaseScheduler, Constant
from src.models.ae.loss import MultinomialLoss
from src.modules import MLP


class EncoderOut(NamedTuple):
    sample: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor


class Model(torch.nn.Module):
    def __init__(
        self,
        encoder: MLP,
        decoder: MLP,
        latent_dim: int,
        normalize: bool = True,
        kl_scheduler: BaseScheduler | None = None,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        assert dropout_prob >= 0.0, "dropout should be greater or equal 0."
        assert (
            latent_dim == encoder.output_size() // 2
        ), "latent_dim should be encoder.output_size() / 2"
        self.variational = True
        self.kl_scheduler = kl_scheduler or Constant(weight=1.0)
        self._encoder = encoder
        self._decoder = decoder
        self._latent_dim = latent_dim
        self._normalize = normalize
        self._dropout = torch.nn.Dropout(dropout_prob) if dropout_prob > 0 else torch.nn.Identity()
        self._recon_loss = MultinomialLoss(size_average=True)

    def _sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_var / 2) * eps

    def _encode(self, source: torch.Tensor) -> EncoderOut:
        # source ~ (batch size, num items)
        source = F.normalize(source, dim=-1, p=2) if self._normalize else source
        source = self._dropout(source)
        # out ~ (batch size, 2 * latent dim)
        out = self._encoder(source)
        mu, log_var = out[:, : self._latent_dim], out[:, self._latent_dim :]
        return EncoderOut(self._sample(mu, log_var), mu=mu, log_var=log_var)

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        # latent ~ (batch size, hidden size)
        # output ~ (batch size, num items)
        return self._decoder(latent)

    def _kl_loss(self, out: EncoderOut) -> torch.Tensor:
        # mu, log_var ~ (batch size, latent dim)
        loss = -0.5 * torch.sum(1 + out.log_var - out.mu**2 - out.log_var.exp(), dim=-1)
        if self._recon_loss.size_average:
            return loss.mean()
        return loss

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # source, target ~ (batch size, num items)
        encode_out = self._encode(inputs["source"])
        # logits, probs ~ (batch size, num items)
        logits = self._decode(encode_out.sample)
        output_dict = {
            "logits": logits,
            "probs": logits.softmax(dim=-1),
        }
        if (target := inputs.get("target")) is not None:
            recon_loss = output_dict["recon_loss"] = self._recon_loss(logits, target)
            kl_loss = output_dict["kl_loss"] = self._kl_loss(encode_out)
            output_dict["loss"] = recon_loss + self.kl_scheduler.weight() * kl_loss
        if self.training:
            self.kl_scheduler.step()
        return output_dict
