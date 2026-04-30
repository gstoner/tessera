"""
tessera_diffusion_llm/models/flow_match.py

Flow Matching Language Model (Rectified Flow).

Forward process (linear interpolation):
    x_t = (1 - t) · x_0 + t · x_1,   x_1 ~ N(0, I), t ∈ [0, 1]

Velocity field:
    v_θ(x_t, t) ≈ x_1 - x_0  (constant along straight paths)

Reverse ODE:
    dx/dt = -v_θ(x_t, t)  integrated from t=1 → t=0

Loss:
    MSE(v_θ(x_t, t), x_1 - x_0)  (conditional flow matching objective)

Reference:
    Liu et al. (2022) "Flow Straight and Fast: Learning to Generate and
    Transfer Data with Rectified Flow." arXiv:2209.03003.
    Lipman et al. (2022) "Flow Matching for Generative Modeling."
    arXiv:2210.02747.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import FlowMatchingConfig
from ..schedules.sampling import flow_ode_sample
from .transformer import DiffusionTransformer


class FlowMatchingLLM(nn.Module):
    """
    Rectified-Flow language model operating in embedding space.

    The model learns a velocity field v_θ(x_t, t) that maps Gaussian noise
    (t=1) straight to data embeddings (t=0) via linear interpolation paths.

    Usage::

        cfg   = FlowMatchingConfig.debug_tiny()
        model = FlowMatchingLLM(cfg).eval()

        # Training
        loss = model.compute_loss(input_ids)

        # Generation
        tokens = model.generate(batch_size=4, seq_len=64)
    """

    def __init__(self, cfg: FlowMatchingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        tcfg = cfg.transformer

        self.backbone = DiffusionTransformer(tcfg)

        # Output logit projection (tied to input embedding)
        self.lm_head = nn.Linear(tcfg.hidden_size, tcfg.vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embed_tokens.weight

        # Velocity prediction head: hidden → hidden (same dim as embeddings)
        self.vel_head = nn.Linear(tcfg.hidden_size, tcfg.hidden_size, bias=False)

        # Optional: learn a per-sample importance weight for loss reweighting
        # (monotone MLP mapping t → scalar weight ∝ 1/σ²(t))
        self._use_lognorm_weighting = getattr(cfg, "lognorm_weighting", False)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity field v_θ(x_t, t).

        Args:
            x_t:       (B, T, H) interpolated embedding at time t.
            t:         (B,) float time in [0, 1].
            attn_mask: Optional (B, T, T) attention mask.

        Returns:
            velocity: (B, T, H)
        """
        # The backbone accepts float t for flow matching (sinusoidal embed
        # works on continuous values)
        h = self.backbone(token_ids=None, t=t, x_emb=x_t, attn_mask=attn_mask)
        return self.vel_head(h)

    # -----------------------------------------------------------------------
    # Diffusion utilities
    # -----------------------------------------------------------------------
    def interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Linear interpolation: x_t = (1-t)·x_0 + t·x_1.

        Args:
            x_0: (B, T, H) clean embeddings.
            x_1: (B, T, H) noise.
            t:   (B,) float time ∈ [0, 1].

        Returns:
            x_t: (B, T, H)
        """
        # Expand t from (B,) → (B, 1, 1) for broadcasting
        t_exp = t.reshape(-1, 1, 1)
        return (1.0 - t_exp) * x_0 + t_exp * x_1

    # -----------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------
    def compute_loss(
        self,
        x_0_ids: torch.LongTensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Conditional flow matching objective.

        Samples t ~ Uniform(0, 1), interpolates between x_0 (data embeddings)
        and x_1 (noise), then regresses on the constant velocity x_1 - x_0.

        Args:
            x_0_ids:   (B, T) clean token ids.
            attn_mask: Optional (B, T) padding mask (1=real, 0=pad).

        Returns:
            Scalar loss tensor.
        """
        B, T = x_0_ids.shape
        device = x_0_ids.device

        # Embed clean tokens
        with torch.no_grad():
            x_0 = self.backbone.embed_tokens(x_0_ids)  # (B, T, H)

        # Source noise
        x_1 = torch.randn_like(x_0)

        # Sample time uniformly from (0, 1)
        t = torch.rand(B, device=device)

        # Interpolate
        x_t = self.interpolate(x_0, x_1, t)

        # Target velocity (constant along straight paths)
        target_vel = x_1 - x_0  # (B, T, H)

        # Predict velocity
        pred_vel = self.forward(x_t, t, attn_mask)

        # MSE loss
        loss = F.mse_loss(pred_vel, target_vel, reduction="none")  # (B, T, H)

        if attn_mask is not None:
            loss = loss * attn_mask.float().unsqueeze(-1)

        if self._use_lognorm_weighting:
            # Log-normal time weighting (Karras et al. 2022 style):
            # w(t) = 1 / (t · (1-t) + ε)  encourages learning mid-t
            w = 1.0 / (t * (1.0 - t) + 1e-4)  # (B,)
            loss = loss * w.reshape(B, 1, 1)

        return loss.mean()

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: Optional[int] = None,
        solver: str = "euler",
        temperature: float = 1.0,
        top_k: int = 0,
        prompt_ids: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """Generate token sequences via flow-matching ODE integration.

        Args:
            batch_size: Number of sequences to generate.
            seq_len:    Length of each generated sequence.
            num_steps:  ODE integration steps.
            solver:     "euler" or "midpoint" / "rk2".
            temperature: Logit temperature for final token sampling.
            top_k:      If >0, sample from top-k instead of argmax.
            prompt_ids: Optional (1, P) or (B, P) prefix tokens to anchor.

        Returns:
            (B, seq_len) integer token tensor.
        """
        device = next(self.parameters()).device
        H = self.cfg.transformer.hidden_size
        shape = (batch_size, seq_len, H)

        steps = num_steps or self.cfg.num_sampling_steps

        def _model_fn(x_t, t_batch):
            return self.forward(x_t, t_batch)

        x_0_hat = flow_ode_sample(
            model_fn=_model_fn,
            shape=shape,
            device=device,
            num_steps=steps,
            solver=solver,
        )

        # Inject prompt embeddings if provided
        if prompt_ids is not None:
            P = prompt_ids.shape[1]
            prompt_emb = self.backbone.embed_tokens(prompt_ids.to(device))
            x_0_hat[:, :P] = prompt_emb[:, :P]

        # Project to vocabulary logits
        logits = self.lm_head(x_0_hat)  # (B, seq_len, vocab_size)

        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            v, _ = logits.topk(top_k, dim=-1)
            logits[logits < v[..., -1:]] = -float("Inf")
            probs = logits.softmax(dim=-1)
            tokens = torch.multinomial(
                probs.reshape(-1, probs.shape[-1]), num_samples=1
            ).reshape(batch_size, seq_len)
        else:
            tokens = logits.argmax(dim=-1)

        return tokens
