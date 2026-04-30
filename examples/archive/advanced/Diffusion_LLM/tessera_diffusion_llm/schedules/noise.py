"""
tessera_diffusion_llm/schedules/noise.py

Noise / masking schedules for all three diffusion variants.

Provides:
  • cosine_beta_schedule   — β_t for continuous diffusion (Nichol & Dhariwal, 2021)
  • linear_beta_schedule   — original Ho et al. schedule
  • sqrt_beta_schedule     — Hoogeboom et al. 2023
  • cosine_mask_schedule   — masking rate m(t) for MDLM
  • linear_mask_schedule   — linear masking rate
  • NoiseSchedule          — pre-computed buffers for continuous diffusion
  • MaskSchedule           — masking probabilities for MDLM
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import torch


# ---------------------------------------------------------------------------
# Beta schedules (continuous diffusion)
# ---------------------------------------------------------------------------

def cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008,
    device: torch.device = None,
) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal, 2021).  Returns β_t of shape (T,)."""
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64, device=device)
    alphas_cumprod = torch.cos(
        ((steps / num_timesteps) + s) / (1.0 + s) * math.pi * 0.5
    ).pow(2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(1e-4, 0.9999).float()


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: torch.device = None,
) -> torch.Tensor:
    """Linear schedule (Ho et al., 2020).  Returns β_t of shape (T,)."""
    return torch.linspace(beta_start, beta_end, num_timesteps,
                          dtype=torch.float32, device=device)


def sqrt_beta_schedule(
    num_timesteps: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Sqrt schedule (Hoogeboom et al., 2023).  Slower noise accumulation."""
    t = torch.linspace(0, 1, num_timesteps + 1, dtype=torch.float64, device=device)
    alphas_cumprod = 1.0 - torch.sqrt(t + 1e-8)
    alphas_cumprod = (alphas_cumprod / alphas_cumprod[0]).clamp(0, 1)
    betas = 1.0 - alphas_cumprod[1:] / (alphas_cumprod[:-1] + 1e-8)
    return betas.clamp(1e-4, 0.9999).float()


# ---------------------------------------------------------------------------
# Masking schedules (MDLM / discrete diffusion)
# ---------------------------------------------------------------------------

def cosine_mask_schedule(
    num_timesteps: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Cosine masking rate m(t) ∈ [0, 1], shape (T+1,).

    m(0) ≈ 0  (no masking at t=0)
    m(T) = 1  (fully masked at t=T)
    """
    t = torch.arange(num_timesteps + 1, dtype=torch.float32, device=device)
    m = 1.0 - torch.cos(0.5 * math.pi * t / num_timesteps).pow(2)
    return m.clamp(0.0, 1.0)


def linear_mask_schedule(
    num_timesteps: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Linear masking rate m(t) = t/T, shape (T+1,)."""
    return torch.linspace(0.0, 1.0, num_timesteps + 1,
                          dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Pre-computed noise schedule buffers (for continuous diffusion)
# ---------------------------------------------------------------------------

class NoiseSchedule(torch.nn.Module):
    """Registers all pre-computed diffusion coefficients as buffers.

    Computes:
        β_t, ᾱ_t, σ_t = √(1 - ᾱ_t), √ᾱ_t
    and their posterior quantities:
        μ̃_coeff1, μ̃_coeff2, β̃_t, log β̃_clipped
    """

    def __init__(
        self,
        num_timesteps: int,
        schedule: Literal["cosine", "linear", "sqrt"] = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        # Build β
        if schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif schedule == "sqrt":
            betas = sqrt_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # Forward process coefficients
        self.register_buffer("betas",                           betas)
        self.register_buffer("alphas",                          alphas)
        self.register_buffer("alphas_cumprod",                  alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",             alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod",   (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("log_one_minus_alphas_cumprod",    (1.0 - alphas_cumprod).log())
        self.register_buffer("sqrt_recip_alphas_cumprod",       (1.0 / alphas_cumprod).sqrt())
        self.register_buffer("sqrt_recipm1_alphas_cumprod",     (1.0 / alphas_cumprod - 1.0).sqrt())

        # Posterior q(x_{t-1} | x_t, x_0) coefficients
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance",              posterior_variance)
        self.register_buffer("posterior_log_var_clipped",
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                             betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod))

        self.num_timesteps = num_timesteps

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε."""
        s = self._extract(self.sqrt_alphas_cumprod, t, x_start)
        sm = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start)
        return s * x_start + sm * noise

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Recover x_0 from (x_t, ε_pred)."""
        s1 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t)
        s2 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return s1 * x_t - s2 * noise

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Posterior mean and variance: q(x_{t-1} | x_t, x_0)."""
        c1  = self._extract(self.posterior_mean_coef1, t, x_t)
        c2  = self._extract(self.posterior_mean_coef2, t, x_t)
        mean = c1 * x_start + c2 * x_t
        var  = self._extract(self.posterior_variance, t, x_t)
        logv = self._extract(self.posterior_log_var_clipped, t, x_t)
        return mean, var, logv

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Gather schedule values at timestep t, broadcast to x's shape."""
        out = a.gather(0, t.long())
        return out.reshape(t.shape[0], *((1,) * (x.ndim - 1)))


# ---------------------------------------------------------------------------
# Masking schedule (MDLM)
# ---------------------------------------------------------------------------

class MaskSchedule(torch.nn.Module):
    """Pre-computed masking rates and marginal transition probabilities for MDLM."""

    def __init__(
        self,
        num_timesteps: int,
        schedule: Literal["cosine", "linear"] = "cosine",
    ) -> None:
        super().__init__()
        if schedule == "cosine":
            m = cosine_mask_schedule(num_timesteps)
        else:
            m = linear_mask_schedule(num_timesteps)

        self.num_timesteps = num_timesteps
        self.register_buffer("mask_rate", m)           # shape (T+1,)

    def get_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Return masking probability for each timestep t ∈ {0,…,T}.
        t: (B,) LongTensor
        Returns: (B,)
        """
        return self.mask_rate[t.long()]

    def get_transition_prob(
        self,
        t: torch.Tensor,
        t_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Transition prob of unmasking from t_prev to t: m(t_prev→t).
        = (m(t) - m(t_prev)) / (1 - m(t_prev) + ε)
        Returns: (B,)
        """
        m_t    = self.mask_rate[t.long()]
        m_tprev = self.mask_rate[t_prev.long()]
        return ((m_t - m_tprev) / (1.0 - m_tprev + 1e-8)).clamp(0.0, 1.0)
