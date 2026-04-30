"""
tessera_diffusion_llm/kernels/time_embed.py

Time-step / noise-level embedding modules for diffusion transformers.

Provides:
  • SinusoidalTimeEmbed  — sinusoidal (fixed) encoding of integer timesteps
  • FourierTimeEmbed     — random-Fourier-feature encoding for continuous t ∈ [0,1]
  • TimeEmbedMLP         — projects the raw embedding to model hidden_size
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal encoding for discrete integer timesteps.

    Maps t ∈ {0, …, T-1} → (B, dim) via sin/cos features.
    Matches the Denoising Diffusion PM encoding (Ho et al., 2020).
    """

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even for sinusoidal embedding"
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer or float timestep tensor.
        Returns:
            (B, dim) sinusoidal embedding.
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        # t: (B,) → (B, 1); freqs: (half,) → (1, half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)   # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
        return emb


class FourierTimeEmbed(nn.Module):
    """Random Fourier Features for continuous t ∈ [0, 1].

    Better suited for flow-matching and continuous-time diffusion where t
    is sampled from [0,1] rather than integers.

    Reference: Karras et al. (2022) "Elucidating the Design Space of
    Diffusion-Based Generative Models".
    """

    def __init__(self, dim: int, bandwidth: float = 1.0) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.bandwidth = bandwidth
        # Fixed random frequencies — not trained
        W = torch.randn(dim // 2) * bandwidth
        self.register_buffer("W", W)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) float tensor in [0, 1].
        Returns:
            (B, dim)
        """
        args = 2.0 * math.pi * t.float().unsqueeze(-1) * self.W.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeEmbedMLP(nn.Module):
    """Two-layer MLP that projects time embeddings to model hidden size.

    Architecture:
        Linear(raw_dim → 4*hidden_size) → SiLU → Linear(4*hidden_size → hidden_size)
    """

    def __init__(self, raw_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(raw_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_emb: (B, raw_dim)
        Returns:
            (B, hidden_size)
        """
        return self.net(t_emb)


class CombinedTimeEmbed(nn.Module):
    """Full time-embedding pipeline: raw encoding → MLP projection.

    Usage::

        t_emb_module = CombinedTimeEmbed(
            hidden_size=768, embed_type="sinusoidal"
        )
        cond = t_emb_module(t)   # (B, 768)
    """

    def __init__(
        self,
        hidden_size: int,
        embed_type: str = "sinusoidal",
        max_period: float = 10_000.0,
    ) -> None:
        super().__init__()
        if embed_type == "sinusoidal":
            self.raw = SinusoidalTimeEmbed(hidden_size, max_period=max_period)
        elif embed_type == "fourier":
            self.raw = FourierTimeEmbed(hidden_size)
        else:
            raise ValueError(f"Unknown embed_type '{embed_type}'")
        self.mlp = TimeEmbedMLP(hidden_size, hidden_size)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.raw(t))
