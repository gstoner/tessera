"""
tessera_diffusion_llm/schedules/sampling.py

Sampling algorithms for all three diffusion variants.

  • ddpm_step         — single DDPM denoising step (stochastic)
  • ddim_step         — single DDIM denoising step (deterministic)
  • ddpm_sample       — full DDPM generation loop
  • ddim_sample       — full DDIM generation loop
  • ode_euler_step    — Euler step for flow matching ODE
  • ode_midpoint_step — Midpoint step for flow matching ODE
  • flow_ode_sample   — full ODE integration loop
  • mdlm_step         — single MDLM unmasking step
  • mdlm_sample       — full MDLM generation loop (confidence-ordered)
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DDPM
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddpm_step(
    x_t: torch.Tensor,
    t: torch.Tensor,
    pred_noise: torch.Tensor,
    schedule,                    # NoiseSchedule instance
    pred_log_var: Optional[torch.Tensor] = None,
    clip_x0: bool = True,
) -> torch.Tensor:
    """One DDPM reverse step: x_{t-1} ~ p(x_{t-1} | x_t).

    Args:
        x_t:          (B, T, D) noisy embeddings at timestep t.
        t:            (B,) integer timestep.
        pred_noise:   (B, T, D) model's ε prediction.
        schedule:     NoiseSchedule with pre-computed buffers.
        pred_log_var: Optional (B, T, D) learned log variance.
        clip_x0:      Whether to clip predicted x₀ to [-1, 1].

    Returns:
        x_{t-1}: (B, T, D)
    """
    x0_pred = schedule.predict_start_from_noise(x_t, t, pred_noise)
    if clip_x0:
        x0_pred = x0_pred.clamp(-1.0, 1.0)

    post_mean, _, post_logv = schedule.q_posterior(x0_pred, x_t, t)

    if pred_log_var is not None:
        # Interpolate between fixed and learned variance
        min_log = schedule._extract(schedule.posterior_log_var_clipped, t, x_t)
        max_log = schedule._extract(torch.log(schedule.betas), t, x_t)
        frac    = (pred_log_var + 1.0) / 2.0
        post_logv = frac * max_log + (1.0 - frac) * min_log

    noise    = torch.randn_like(x_t)
    nonzero  = (t > 0).float().reshape(-1, *([1] * (x_t.ndim - 1)))
    return post_mean + nonzero * (0.5 * post_logv).exp() * noise


@torch.no_grad()
def ddpm_sample(
    model_fn: Callable,           # (x_t, t) → pred_noise [, pred_log_var]
    schedule,
    shape: Tuple[int, ...],
    device: torch.device,
    clip_x0: bool = True,
    learned_variance: bool = False,
) -> torch.Tensor:
    """Full DDPM generation.  Returns final sample (B, T, D)."""
    x = torch.randn(shape, device=device)
    T = schedule.num_timesteps
    for t_int in reversed(range(T)):
        t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
        out = model_fn(x, t)
        pred_noise  = out[0] if isinstance(out, (tuple, list)) else out
        pred_logv   = out[1] if (isinstance(out, (tuple, list)) and learned_variance) else None
        x = ddpm_step(x, t, pred_noise, schedule, pred_logv, clip_x0)
    return x


# ---------------------------------------------------------------------------
# DDIM
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_step(
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    pred_noise: torch.Tensor,
    schedule,
    eta: float = 0.0,
    clip_x0: bool = True,
) -> torch.Tensor:
    """One DDIM step (Song et al., 2020).

    eta=0 → deterministic; eta=1 → DDPM.
    """
    x0_pred = schedule.predict_start_from_noise(x_t, t, pred_noise)
    if clip_x0:
        x0_pred = x0_pred.clamp(-1.0, 1.0)

    acp_t    = schedule._extract(schedule.alphas_cumprod, t, x_t)
    acp_prev = schedule._extract(schedule.alphas_cumprod, t_prev, x_t)

    sigma = eta * ((1.0 - acp_prev) / (1.0 - acp_t) * (1.0 - acp_t / acp_prev)).sqrt()
    c     = (1.0 - acp_prev - sigma ** 2).sqrt()

    x_prev = acp_prev.sqrt() * x0_pred + c * pred_noise + sigma * torch.randn_like(x_t)
    return x_prev


@torch.no_grad()
def ddim_sample(
    model_fn: Callable,
    schedule,
    shape: Tuple[int, ...],
    device: torch.device,
    num_steps: int = 50,
    eta: float = 0.0,
    clip_x0: bool = True,
) -> torch.Tensor:
    """DDIM generation with sub-sampled timestep sequence."""
    x = torch.randn(shape, device=device)
    T = schedule.num_timesteps
    # Select evenly-spaced subset of timesteps
    step_size  = T // num_steps
    timesteps  = list(reversed(range(0, T, step_size)))
    prev_steps = timesteps[1:] + [0]

    for t_int, tp_int in zip(timesteps, prev_steps):
        t    = torch.full((shape[0],), t_int,  device=device, dtype=torch.long)
        t_p  = torch.full((shape[0],), tp_int, device=device, dtype=torch.long)
        out  = model_fn(x, t)
        pred_noise = out[0] if isinstance(out, (tuple, list)) else out
        x = ddim_step(x, t, t_p, pred_noise, schedule, eta, clip_x0)
    return x


# ---------------------------------------------------------------------------
# Flow Matching ODE
# ---------------------------------------------------------------------------

@torch.no_grad()
def ode_euler_step(
    x_t: torch.Tensor,
    t_float: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Euler step: x_{t-dt} = x_t - dt * v_θ(x_t, t)."""
    return x_t - dt * velocity


@torch.no_grad()
def ode_midpoint_step(
    x_t: torch.Tensor,
    t_float: torch.Tensor,
    model_fn: Callable,
    dt: float,
) -> torch.Tensor:
    """Midpoint (RK2) step for more accurate ODE integration."""
    v1 = model_fn(x_t, t_float)
    x_mid = x_t - 0.5 * dt * v1
    t_mid = t_float - 0.5 * dt
    v2 = model_fn(x_mid, t_mid)
    return x_t - dt * v2


@torch.no_grad()
def flow_ode_sample(
    model_fn: Callable,           # (x_t, t_float) → velocity
    shape: Tuple[int, ...],
    device: torch.device,
    num_steps: int = 50,
    solver: str = "euler",
) -> torch.Tensor:
    """Integrate flow-matching ODE from t=1 (noise) to t=0 (data).

    Returns final sample (B, T, D).
    """
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_val = 1.0 - i * dt            # goes from 1 → 0
        t = torch.full((shape[0],), t_val, device=device, dtype=torch.float32)

        if solver == "euler":
            v = model_fn(x, t)
            x = ode_euler_step(x, t, v, dt)
        elif solver in ("midpoint", "rk2"):
            x = ode_midpoint_step(x, t, model_fn, dt)
        else:
            raise ValueError(f"Unknown solver '{solver}'")

    return x


# ---------------------------------------------------------------------------
# MDLM sampling (discrete, confidence-ordered)
# ---------------------------------------------------------------------------

@torch.no_grad()
def mdlm_step(
    x_t: torch.LongTensor,
    t: torch.LongTensor,
    t_prev: torch.LongTensor,
    logits: torch.Tensor,         # (B, T, vocab_size) model output
    mask_token_id: int,
    schedule,                     # MaskSchedule instance
    temperature: float = 1.0,
) -> torch.LongTensor:
    """Single MDLM reverse step: unmask some tokens using model logits.

    Only positions that are currently masked (x_t == mask_token_id) can
    be unmasked.  The number unmasked is determined by the schedule
    transition probability from t to t_prev.

    Args:
        x_t:           (B, T) token ids at timestep t.
        t:             (B,) current timestep.
        t_prev:        (B,) previous (less noisy) timestep.
        logits:        (B, T, V) raw logits from the denoising model.
        mask_token_id: Integer id of the [MASK] token.
        schedule:      MaskSchedule instance.
        temperature:   Sampling temperature.

    Returns:
        x_{t_prev}: (B, T) partially unmasked token sequence.
    """
    B, T = x_t.shape
    is_masked = (x_t == mask_token_id)  # (B, T) bool

    # Sample tokens from model distribution at masked positions
    probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)  # (B, T, V)
    # Zero out the mask token itself so we never sample it
    probs[..., mask_token_id] = 0.0
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    sampled = torch.multinomial(
        probs.reshape(B * T, -1), num_samples=1
    ).reshape(B, T)

    # Fraction of masked tokens to reveal this step
    trans_prob = schedule.get_transition_prob(t, t_prev)  # (B,)

    # For each sequence, keep the fraction (1 - trans_prob) masked
    # by randomly re-masking some of the sampled tokens
    keep_mask_prob = (1.0 - trans_prob).clamp(0.0, 1.0)  # (B,)
    random_keep    = torch.rand(B, T, device=x_t.device)
    # Expand per-batch keep probability
    keep_mask_this = random_keep < keep_mask_prob.unsqueeze(-1)  # (B, T)

    # Build output: keep original non-masked; at masked positions, either
    # reveal sampled token or keep masked according to keep_mask_this
    x_new = x_t.clone()
    reveal = is_masked & ~keep_mask_this
    x_new[reveal] = sampled[reveal]
    return x_new


@torch.no_grad()
def mdlm_sample(
    model_fn: Callable,           # (x_t, t) → logits (B, T, V)
    schedule,                     # MaskSchedule
    vocab_size: int,
    mask_token_id: int,
    shape: Tuple[int, int],       # (B, T)
    device: torch.device,
    num_steps: Optional[int] = None,
    temperature: float = 1.0,
) -> torch.LongTensor:
    """Full MDLM generation.

    Args:
        shape:     (batch_size, seq_len)
        num_steps: Number of denoising steps (≤ T). None → all T steps.
    """
    B, T_seq = shape
    T = schedule.num_timesteps
    num_steps = num_steps or T

    # Start fully masked
    x = torch.full((B, T_seq), mask_token_id, device=device, dtype=torch.long)

    step_size  = T // num_steps
    timesteps  = list(reversed(range(0, T, step_size)))
    prev_steps = timesteps[1:] + [0]

    for t_int, tp_int in zip(timesteps, prev_steps):
        t    = torch.full((B,), t_int,  device=device, dtype=torch.long)
        t_p  = torch.full((B,), tp_int, device=device, dtype=torch.long)
        logits = model_fn(x, t)   # (B, T_seq, V)
        x = mdlm_step(x, t, t_p, logits, mask_token_id, schedule, temperature)

    return x
