"""
tessera_diffusion_llm/models/continuous.py

Continuous Diffusion Language Model.

Forward process (Gaussian diffusion in embedding space):
    q(x_t | x_0): x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0, I)

Reverse process:
    p_θ(x_{t-1} | x_t) via DDPM or DDIM sampling.

Prediction targets:
    epsilon  — predict noise ε (standard)
    x_start  — predict clean x_0 directly

Loss:
    Simple: MSE(pred, target) on entire sequence.
    If learned_variance=True: also minimise a VLB term on variance.

Reference:
    Ho et al. (2020)   "DDPM" arXiv:2006.11239
    Nichol & Dhariwal  "Improved DDPM" arXiv:2102.09672
    Austin et al.      "Structured Denoising Diffusion" arXiv:2107.03006
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import ContinuousDiffusionConfig
from ..schedules.noise import NoiseSchedule
from ..schedules.sampling import ddpm_sample, ddim_sample
from .transformer import DiffusionTransformer


class ContinuousDiffusionLLM(nn.Module):
    """
    Continuous (Gaussian) Diffusion Language Model.

    Diffusion operates in the token-embedding space — the model learns to
    denoise corrupted embedding sequences rather than discrete token ids
    directly.  The embedding lookup is used for the forward process only;
    during generation the final continuous vector is projected back to
    logits via the tied output head.

    Usage::

        cfg   = ContinuousDiffusionConfig.debug_tiny()
        model = ContinuousDiffusionLLM(cfg).eval()

        # Training
        loss = model.compute_loss(input_ids)

        # Generation
        tokens = model.generate(batch_size=4, seq_len=64)
    """

    def __init__(self, cfg: ContinuousDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        tcfg = cfg.transformer

        self.backbone = DiffusionTransformer(tcfg)

        # Output projection → logits (tied to input embedding)
        self.lm_head = nn.Linear(tcfg.hidden_size, tcfg.vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embed_tokens.weight

        # Noise prediction head (ε or x_0 in embedding space)
        # Hidden → hidden_size (same dim as embedding)
        self.noise_head = nn.Linear(tcfg.hidden_size, tcfg.hidden_size, bias=False)

        # Learned variance head (optional, Nichol & Dhariwal 2021)
        if cfg.learned_variance:
            self.var_head = nn.Linear(tcfg.hidden_size, tcfg.hidden_size, bias=False)

        # Pre-computed noise schedule
        self.schedule = NoiseSchedule(cfg.num_timesteps, cfg.beta_schedule)

        self._pred_type = cfg.prediction_type   # "epsilon" or "x_start"

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.LongTensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x_t:       (B, T, H) noisy embedding sequence at timestep t.
            t:         (B,) integer timestep.
            attn_mask: Optional (B, T, T) boolean attention mask.

        Returns:
            pred:     (B, T, H) — predicted ε or x_0 in embedding space.
            log_var:  (B, T, H) or None — learned log variance (if enabled).
        """
        h = self.backbone(token_ids=None, t=t, x_emb=x_t, attn_mask=attn_mask)
        pred = self.noise_head(h)

        log_var = None
        if self.cfg.learned_variance:
            log_var = self.var_head(h)

        return pred, log_var

    def predict_logits(
        self,
        x_t: torch.Tensor,
        t: torch.LongTensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the model and project continuous output to vocabulary logits.

        Useful for perplexity evaluation or beam search on top of continuous
        diffusion outputs.

        Returns:
            logits: (B, T, vocab_size)
        """
        h = self.backbone(token_ids=None, t=t, x_emb=x_t, attn_mask=attn_mask)
        return self.lm_head(h)

    # -----------------------------------------------------------------------
    # Diffusion utilities
    # -----------------------------------------------------------------------
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.LongTensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: corrupt x_0 embeddings to x_t.

        Args:
            x_0:   (B, T, H) clean embedding sequence.
            t:     (B,) integer timestep.
            noise: Optional (B, T, H) noise tensor; sampled if None.

        Returns:
            (x_t, noise): both (B, T, H).
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.schedule.q_sample(x_0, t, noise)
        return x_t, noise

    def _predict_x0_from_pred(
        self,
        x_t: torch.Tensor,
        t: torch.LongTensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """Convert model prediction to x_0 estimate (regardless of pred type)."""
        if self._pred_type == "epsilon":
            return self.schedule.predict_start_from_noise(x_t, t, pred)
        else:  # x_start
            return pred

    # -----------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------
    def compute_loss(
        self,
        x_0_ids: torch.LongTensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the diffusion training loss.

        Samples a random timestep, corrupts the token embeddings, runs the
        denoising model, and computes MSE against the prediction target.
        If learned_variance is enabled, also adds a VLB term.

        Args:
            x_0_ids:   (B, T) clean token ids.
            attn_mask: Optional (B, T) padding mask (1=real, 0=pad).

        Returns:
            Scalar loss tensor.
        """
        B, T = x_0_ids.shape
        device = x_0_ids.device

        # Embed clean tokens → continuous x_0
        with torch.no_grad():
            x_0 = self.backbone.embed_tokens(x_0_ids)  # (B, T, H)

        # Sample timestep
        t = torch.randint(1, self.cfg.num_timesteps + 1, (B,), device=device)

        # Forward diffusion
        noise = torch.randn_like(x_0)
        x_t, noise = self.q_sample(x_0, t, noise)

        # Denoising prediction
        pred, log_var = self.forward(x_t, t, attn_mask)

        # Prediction target
        target = noise if self._pred_type == "epsilon" else x_0

        # Simple MSE loss
        mse = F.mse_loss(pred, target, reduction="none")  # (B, T, H)

        # Apply padding mask if provided
        if attn_mask is not None:
            # attn_mask: (B, T), expand to (B, T, H)
            mse = mse * attn_mask.float().unsqueeze(-1)

        simple_loss = mse.mean()

        if not self.cfg.learned_variance or log_var is None:
            return simple_loss

        # VLB term: penalise log variance using KL between predicted and
        # true posterior (Nichol & Dhariwal eq. 9, simplified)
        # Only applied when t > 1 to avoid the t=0 boundary
        t_gt1 = (t > 1).float()  # (B,)

        # Posterior log-variance (fixed)
        post_logv = self.schedule._extract(
            self.schedule.posterior_log_var_clipped, t, x_t
        )  # (B, 1, 1) broadcastable

        # Predicted log-variance (interpolated between posterior bounds)
        min_log = post_logv
        max_log = self.schedule._extract(
            torch.log(self.schedule.betas), t, x_t
        )
        # log_var output interpreted as unconstrained mix coeff ∈ [-1, 1]
        frac = (log_var.tanh() + 1.0) / 2.0
        pred_logv = frac * max_log + (1.0 - frac) * min_log

        # KL divergence (diagonal Gaussian)
        kl = 0.5 * (pred_logv - post_logv + post_logv.exp() / (pred_logv.exp() + 1e-8) - 1.0)
        kl_loss = kl.mean(-1)  # (B, T)
        if attn_mask is not None:
            kl_loss = kl_loss * attn_mask.float()

        vlb = (t_gt1.reshape(B, 1) * kl_loss).mean()

        return simple_loss + 1e-3 * vlb

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: Optional[int] = None,
        sampler: str = "ddpm",
        temperature: float = 1.0,
        eta: float = 0.0,
        top_k: int = 0,
        prompt_ids: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """Generate token sequences via continuous diffusion.

        1. Sample noise in embedding space.
        2. Run DDPM/DDIM reverse process to recover clean embeddings.
        3. Project to logits and argmax (or sample) to get token ids.

        Args:
            batch_size: Number of sequences to generate.
            seq_len:    Length of each generated sequence.
            num_steps:  Denoising steps (< num_timesteps for DDIM).
            sampler:    "ddpm" or "ddim".
            temperature: Logit temperature for final token sampling.
            eta:        DDIM stochasticity (0 = deterministic, 1 = DDPM).
            top_k:      If >0, sample from top-k logits rather than argmax.
            prompt_ids: Optional (1, P) or (B, P) prefix tokens to anchor.

        Returns:
            (B, seq_len) integer token tensor.
        """
        device = next(self.parameters()).device
        H = self.cfg.transformer.hidden_size
        shape = (batch_size, seq_len, H)

        steps = num_steps or self.cfg.num_timesteps

        def _model_fn(x_t, t_batch):
            pred, _ = self.forward(x_t, t_batch)
            if self._pred_type == "epsilon":
                return pred
            # Convert x_0 prediction to noise for DDPM/DDIM interface
            s = self.schedule._extract(
                self.schedule.sqrt_alphas_cumprod, t_batch, x_t
            )
            sm = self.schedule._extract(
                self.schedule.sqrt_one_minus_alphas_cumprod, t_batch, x_t
            )
            return (x_t - s * pred) / (sm + 1e-8)

        if sampler == "ddim":
            x_0_hat = ddim_sample(
                _model_fn, self.schedule, shape, device,
                num_steps=steps, eta=eta,
            )
        else:
            x_0_hat = ddpm_sample(
                _model_fn, self.schedule, shape, device,
            )

        # Inject prompt embeddings if provided (overwrite first P positions)
        if prompt_ids is not None:
            P = prompt_ids.shape[1]
            prompt_emb = self.backbone.embed_tokens(prompt_ids.to(device))
            x_0_hat[:, :P] = prompt_emb[:, :P]

        # Project continuous embeddings to vocabulary logits
        # Use the lm_head (tied weight: vocab_emb · x_0_hat^T then argmax)
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
