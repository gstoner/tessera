"""
tessera_diffusion_llm/training/losses.py

Standalone loss functions for all three diffusion model variants.

Each function takes a model and a batch of token ids and returns a scalar
loss tensor suitable for back-propagation.  They delegate to the model's
own `compute_loss` method but add any cross-model utilities (e.g. gradient
clipping helpers, loss scale tracking) that don't belong inside the models.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MDLM — Masked Discrete Diffusion
# ---------------------------------------------------------------------------

def mdlm_elbo_loss(
    model: nn.Module,
    input_ids: torch.LongTensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ELBO loss for the Masked Discrete Diffusion LM.

    Args:
        model:      MDLM instance.
        input_ids:  (B, T) token ids.
        attn_mask:  Optional (B, T) padding mask (1=real, 0=pad).

    Returns:
        Scalar loss.
    """
    return model.compute_loss(input_ids, attn_mask=attn_mask)


# ---------------------------------------------------------------------------
# Continuous Diffusion
# ---------------------------------------------------------------------------

def continuous_diffusion_loss(
    model: nn.Module,
    input_ids: torch.LongTensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hybrid MSE + VLB loss for the Continuous Diffusion LM.

    Args:
        model:      ContinuousDiffusionLLM instance.
        input_ids:  (B, T) token ids (used to obtain embeddings).
        attn_mask:  Optional (B, T) padding mask.

    Returns:
        Scalar loss.
    """
    return model.compute_loss(input_ids, attn_mask=attn_mask)


# ---------------------------------------------------------------------------
# Flow Matching
# ---------------------------------------------------------------------------

def flow_matching_loss(
    model: nn.Module,
    input_ids: torch.LongTensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Conditional flow-matching objective for the FlowMatchingLLM.

    Args:
        model:      FlowMatchingLLM instance.
        input_ids:  (B, T) token ids.
        attn_mask:  Optional (B, T) padding mask.

    Returns:
        Scalar loss.
    """
    return model.compute_loss(input_ids, attn_mask=attn_mask)


# ---------------------------------------------------------------------------
# Utility: per-sample loss breakdown (for diagnostics)
# ---------------------------------------------------------------------------

def per_timestep_loss(
    model: nn.Module,
    input_ids: torch.LongTensor,
    num_buckets: int = 10,
) -> dict:
    """Compute average loss binned by noise level.

    Runs `num_buckets` forward passes with deterministic timestep bins and
    returns a dict mapping bucket midpoint → mean loss.  Useful for
    diagnosing whether a particular noise level is causing training issues.

    Args:
        model:       Any of MDLM / ContinuousDiffusionLLM / FlowMatchingLLM.
        input_ids:   (B, T) token ids.
        num_buckets: Number of evenly-spaced timestep bins.

    Returns:
        dict[float, float] — {t_midpoint: mean_loss}
    """
    B, T = input_ids.shape
    device = input_ids.device
    results = {}

    # Retrieve num_timesteps from the model's schedule or config
    if hasattr(model, "schedule") and hasattr(model.schedule, "num_timesteps"):
        N = model.schedule.num_timesteps
    elif hasattr(model, "cfg"):
        N = getattr(model.cfg, "num_timesteps", getattr(model.cfg, "num_sampling_steps", 100))
    else:
        N = 100

    bucket_size = max(N // num_buckets, 1)

    with torch.no_grad():
        for i in range(num_buckets):
            t_start = i * bucket_size + 1
            t_end   = min((i + 1) * bucket_size, N)
            t_mid   = (t_start + t_end) // 2
            midpoint_float = t_mid / N

            # Patch the model's random timestep sampling via a monkey-patch
            # on torch.randint just for this call is fragile; instead we
            # call the underlying q_sample + forward directly when possible.
            if hasattr(model, "q_sample") and hasattr(model, "lm_head"):
                # MDLM path
                t_fixed = torch.full((B,), t_mid, device=device, dtype=torch.long)
                x_t = model.q_sample(input_ids, t_fixed)
                logits = model.forward(x_t, t_fixed)
                is_masked = (x_t == model.mask_token_id)
                if is_masked.any():
                    import torch.nn.functional as F
                    loss = F.cross_entropy(
                        logits[is_masked], input_ids[is_masked]
                    ).item()
                else:
                    loss = 0.0
            else:
                loss = float("nan")

            results[midpoint_float] = loss

    return results
