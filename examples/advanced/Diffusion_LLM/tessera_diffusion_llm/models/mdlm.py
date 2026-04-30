"""
tessera_diffusion_llm/models/mdlm.py

Masked Discrete Diffusion Language Model (MDLM).

Forward process (absorbing-state diffusion):
    q(x_t | x_0): replace each token with [MASK] independently with
    probability m(t), where m(t) is the masking schedule.

Reverse process:
    p_θ(x_{t-1} | x_t) = argmax / sample from logits produced by the
    bidirectional transformer denoising network.

Loss:
    ELBO = E_t [ 1/(m(t)+ε) · CE(f_θ(x_t, t), x_0) ] on masked positions.
    (The 1/m(t) re-weighting achieves uniform SNR across noise levels.)

Reference:
    Shi et al. (2024) "Simplified and Generalized Masked Diffusion for
    Discrete Data". arXiv:2406.04329.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import MDLMConfig
from ..schedules.noise import MaskSchedule
from ..schedules.sampling import mdlm_sample
from .transformer import DiffusionTransformer


class MDLM(nn.Module):
    """
    Masked Discrete Diffusion Language Model.

    Usage::

        cfg   = MDLMConfig.debug_tiny()
        model = MDLM(cfg).eval()

        # Training
        loss = model.compute_loss(input_ids)

        # Generation
        tokens = model.generate(batch_size=4, seq_len=64)
    """

    def __init__(self, cfg: MDLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        tcfg = cfg.transformer

        self.backbone = DiffusionTransformer(tcfg)

        # Output projection → logits over vocabulary
        self.lm_head = nn.Linear(tcfg.hidden_size, tcfg.vocab_size, bias=False)
        # Tie embedding weights (saves params, improves alignment)
        self.lm_head.weight = self.backbone.embed_tokens.weight

        # Self-conditioning projection
        if cfg.self_condition:
            self.self_cond_proj = nn.Linear(
                tcfg.vocab_size, tcfg.hidden_size, bias=False
            )

        # Noise schedule
        self.schedule = MaskSchedule(cfg.num_timesteps, cfg.mask_schedule)

        self.mask_token_id = tcfg.mask_token_id

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
    def forward(
        self,
        x_t: torch.LongTensor,
        t: torch.LongTensor,
        x_self_cond: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t:          (B, T) partially-masked token ids at timestep t.
            t:            (B,) integer timestep.
            x_self_cond:  Optional (B, T, V) soft-token prediction from
                          previous denoising step (self-conditioning).
            attn_mask:    Optional (B, T, T) boolean mask.

        Returns:
            logits: (B, T, vocab_size)
        """
        x_emb = self.backbone.embed_tokens(x_t)  # (B, T, H)

        # Self-conditioning: inject previous soft prediction
        if self.cfg.self_condition and x_self_cond is not None:
            sc_emb = self.self_cond_proj(x_self_cond)  # (B, T, H)
            x_emb = x_emb + sc_emb

        h = self.backbone(
            token_ids=None, t=t, x_emb=x_emb, attn_mask=attn_mask
        )
        return self.lm_head(h)  # (B, T, V)

    # -----------------------------------------------------------------------
    # Diffusion utilities
    # -----------------------------------------------------------------------
    def q_sample(
        self,
        x_0: torch.LongTensor,
        t: torch.LongTensor,
    ) -> torch.LongTensor:
        """Forward diffusion: randomly mask tokens according to schedule.

        Args:
            x_0: (B, T) clean token ids.
            t:   (B,) integer timestep.
        Returns:
            x_t: (B, T) with some tokens replaced by mask_token_id.
        """
        mask_prob = self.schedule.get_mask_prob(t)  # (B,)
        # Draw a uniform random variable per position
        rand = torch.rand_like(x_0, dtype=torch.float32)
        # Expand mask_prob to (B, T) for element-wise comparison
        should_mask = rand < mask_prob.unsqueeze(-1)
        x_t = x_0.clone()
        x_t[should_mask] = self.mask_token_id
        return x_t

    # -----------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------
    def compute_loss(
        self,
        x_0: torch.LongTensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the MDLM ELBO loss.

        Samples a random timestep, applies forward diffusion, runs the
        denoising model, and computes cross-entropy on masked positions
        weighted by 1 / m(t) for uniform-SNR training.

        Args:
            x_0:      (B, T) clean token ids.
            attn_mask: Optional (B, T) padding mask (1=real, 0=pad).

        Returns:
            Scalar loss tensor.
        """
        B, T = x_0.shape
        device = x_0.device

        # Sample timestep t ∈ {1, …, T}
        t = torch.randint(1, self.cfg.num_timesteps + 1, (B,), device=device)

        # Forward diffusion
        x_t = self.q_sample(x_0, t)

        # Self-conditioning (50% of training steps)
        x_self_cond = None
        if self.cfg.self_condition and torch.rand(1).item() < self.cfg.self_cond_prob:
            with torch.no_grad():
                logits_sc = self.forward(x_t, t)
                x_self_cond = logits_sc.softmax(dim=-1).detach()

        # Denoising model
        logits = self.forward(x_t, t, x_self_cond)  # (B, T, V)

        # Mask of positions that were masked in x_t
        is_masked = (x_t == self.mask_token_id)  # (B, T)

        # Apply padding mask
        if attn_mask is not None:
            is_masked = is_masked & attn_mask.bool()

        if not is_masked.any():
            return logits.sum() * 0.0  # graceful zero when nothing to predict

        # Cross-entropy on masked positions
        logits_flat = logits[is_masked]   # (N_masked, V)
        targets     = x_0[is_masked]     # (N_masked,)
        ce = F.cross_entropy(logits_flat, targets, reduction="none")  # (N_masked,)

        if self.cfg.reweight_loss:
            # Weight by 1 / m(t) to achieve uniform SNR
            m_t = self.schedule.get_mask_prob(t)  # (B,)
            # Expand weights to match masked positions
            weights = 1.0 / (m_t.unsqueeze(-1).expand_as(x_t)[is_masked] + 1e-8)
            ce = ce * weights

        return ce.mean()

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        prompt_ids: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """Generate token sequences via masked diffusion.

        Args:
            batch_size: Number of sequences to generate.
            seq_len:    Sequence length.
            num_steps:  Denoising steps (≤ num_timesteps; None = all).
            temperature: Sampling temperature.
            prompt_ids: Optional (1, P) or (B, P) prompt tokens to condition on.

        Returns:
            (B, seq_len) integer token tensor.
        """
        device = next(self.parameters()).device

        def _model_fn(x_t, t_batch):
            return self.forward(x_t, t_batch)

        tokens = mdlm_sample(
            model_fn=_model_fn,
            schedule=self.schedule,
            vocab_size=self.cfg.transformer.vocab_size,
            mask_token_id=self.mask_token_id,
            shape=(batch_size, seq_len),
            device=device,
            num_steps=num_steps or self.cfg.num_timesteps,
            temperature=temperature,
        )

        # Inject prompt tokens (override first P positions)
        if prompt_ids is not None:
            P = prompt_ids.shape[1]
            tokens[:, :P] = prompt_ids[:, :P].to(device)

        return tokens
