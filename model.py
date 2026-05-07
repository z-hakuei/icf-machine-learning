"""
Any-order Autoregressive Transformer + GMM model for ICF surrogate / inverse / posterior tasks.

Architecture follows the design in 机器学习横版.pdf:

  Per-token features = [value, mask, onehot_id]      (1 + 1 + D)
  Linear[2+D, d_model]
  -> Stack of self-attention encoder layers (full bi-directional attention)
  -> Linear[d_model, 3*K]
  -> Gaussian Mixture Model head (K components) per physical quantity

Training objective: -log p(x_masked | x_visible) averaged over masked tokens,
with the masking ratio r ~ U(0,1) sampled fresh for every batch.

Because the Transformer is permutation-equivariant w.r.t. tokens (positional
information is carried entirely by the per-token onehot ID), the same network
implements ALL conditional distributions p(x_S | x_{S^c}) for any subset S.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_dim: int = 20          # number of physical quantities (5 inputs + 15 outputs)
    d_model: int = 64        # hidden dimension
    n_layers: int = 4        # number of self-attention layers
    n_heads: int = 4         # attention heads
    d_ff: int = 256          # feed-forward dim
    n_gmm: int = 10          # number of Gaussian mixture components per dim
    dropout: float = 0.0
    sigma_min: float = 1e-3  # numerical floor on GMM std


class GMMHead(nn.Module):
    """Maps a token vector of dim d_model to GMM parameters (w, mu, sigma)."""

    def __init__(self, d_model: int, k: int, sigma_min: float = 1e-3):
        super().__init__()
        self.k = k
        self.sigma_min = sigma_min
        self.proj = nn.Linear(d_model, 3 * k)

    def forward(self, h: torch.Tensor):
        """h: (..., d_model) -> (w, mu, sigma) each of shape (..., k)"""
        out = self.proj(h)
        w_logits, mu, log_sigma = out.chunk(3, dim=-1)
        log_w = F.log_softmax(w_logits, dim=-1)
        sigma = F.softplus(log_sigma) + self.sigma_min
        return log_w, mu, sigma

    @staticmethod
    def log_prob(x: torch.Tensor, log_w: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """x: (...,), log_w/mu/sigma: (..., k) -> log p(x) with shape (...,)"""
        x = x.unsqueeze(-1)
        log_norm = -0.5 * math.log(2 * math.pi) - torch.log(sigma)
        log_pdf = log_norm - 0.5 * ((x - mu) / sigma) ** 2
        return torch.logsumexp(log_w + log_pdf, dim=-1)

    @staticmethod
    def sample(log_w: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Draw one sample per (...,) location from the predicted GMM."""
        w = log_w.exp()
        # categorical pick of mixture component
        flat_w = w.reshape(-1, w.shape[-1])
        idx = torch.multinomial(flat_w, num_samples=1).reshape(w.shape[:-1])
        gather_idx = idx.unsqueeze(-1)
        chosen_mu = mu.gather(-1, gather_idx).squeeze(-1)
        chosen_sigma = sigma.gather(-1, gather_idx).squeeze(-1)
        return chosen_mu + chosen_sigma * torch.randn_like(chosen_mu)

    @staticmethod
    def pdf_grid(log_w: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """Evaluate p(x) on a grid xs (shape (g,)) for each (...,) location.
        Returns shape (..., g)."""
        # broadcast: (..., 1, k) vs (1, ..., g, 1) -> (..., g, k)
        log_w_e = log_w.unsqueeze(-2)
        mu_e = mu.unsqueeze(-2)
        sigma_e = sigma.unsqueeze(-2)
        xs_e = xs.reshape(*([1] * (mu.ndim - 1)), -1, 1)
        log_norm = -0.5 * math.log(2 * math.pi) - torch.log(sigma_e)
        log_pdf = log_norm - 0.5 * ((xs_e - mu_e) / sigma_e) ** 2
        return torch.logsumexp(log_w_e + log_pdf, dim=-1).exp()


class AnyOrderARTransformer(nn.Module):
    """Bidirectional Transformer encoder that, in combination with a random
    masking schedule and a GMM output head, learns every conditional density
    p(x_S | x_{S^c}) of the joint distribution over n_dim physical quantities.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Per-token feature: [value, mask_flag, onehot_id (n_dim)]
        in_dim = 2 + cfg.n_dim
        self.embed = nn.Linear(in_dim, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.head = GMMHead(cfg.d_model, cfg.n_gmm, cfg.sigma_min)

        # Pre-built onehot id matrix for n_dim tokens
        self.register_buffer("id_onehot", torch.eye(cfg.n_dim), persistent=False)

    def encode(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """values, mask : (B, n_dim) float. mask=1 means TOKEN IS MASKED (hidden)."""
        B, N = values.shape
        masked_values = values * (1.0 - mask)  # zero-out the hidden positions
        ids = self.id_onehot.unsqueeze(0).expand(B, N, N)
        feats = torch.cat([masked_values.unsqueeze(-1), mask.unsqueeze(-1), ids], dim=-1)
        h = self.embed(feats)
        h = self.encoder(h)
        return h

    def forward(self, values: torch.Tensor, mask: torch.Tensor):
        """Returns GMM params (log_w, mu, sigma), each (B, n_dim, k)."""
        h = self.encode(values, mask)
        return self.head(h)

    # ---- losses & sampling utilities -------------------------------------

    def loss(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood, averaged over MASKED positions only.

        The any-order autoregressive identity (Uria 2014, ARDM)
            log p(x_S | x_{S^c}) = sum_{i in S} log p(x_i | x_{S^c})
        means that, when the per-batch masking ratio r is sampled uniformly
        on (0,1), minimising this objective converges to the joint negative
        log-likelihood of the data distribution (KL minimiser, Goodfellow 2016).
        """
        log_w, mu, sigma = self.forward(values, mask)
        lp = GMMHead.log_prob(values, log_w, mu, sigma)  # (B, n_dim)
        denom = mask.sum().clamp_min(1.0)
        return -(lp * mask).sum() / denom

    @torch.no_grad()
    def predict_marginals(self, condition_values: torch.Tensor, condition_mask: torch.Tensor):
        """Predict GMM parameters for every dimension under a given conditioning.

        condition_mask = 1 where dim is HIDDEN (we predict it),
                       = 0 where dim is OBSERVED (its value is in condition_values).
        """
        return self.forward(condition_values, condition_mask)

    @torch.no_grad()
    def sample_joint(self, condition_values: torch.Tensor, condition_mask: torch.Tensor,
                     order: list[int] | None = None) -> torch.Tensor:
        """Auto-regressively sample the masked dimensions, one at a time, in `order`.
        If order is None a random permutation of the masked indices is used.
        Returns a fully-filled (B, n_dim) tensor.
        """
        B, N = condition_values.shape
        values = condition_values.clone()
        mask = condition_mask.clone()
        if order is None:
            order = torch.randperm(N).tolist()
            order = [i for i in order if mask[0, i] > 0.5]
        for i in order:
            log_w, mu, sigma = self.forward(values, mask)
            sample = GMMHead.sample(log_w[:, i], mu[:, i], sigma[:, i])
            values[:, i] = sample
            mask[:, i] = 0.0
        return values

    @torch.no_grad()
    def predict_mean(self, condition_values: torch.Tensor, condition_mask: torch.Tensor) -> torch.Tensor:
        """Mean of the predicted GMM at every dimension (used as point estimate
        for surrogate-model R²/RMSE evaluation)."""
        log_w, mu, _ = self.forward(condition_values, condition_mask)
        return (log_w.exp() * mu).sum(dim=-1)
