"""
tessera_diffusion_llm/configs.py

Configuration dataclasses for all three Diffusion LLM variants:
  • MDLM   — Masked Discrete Diffusion Language Model (token-space)
  • ContinuousDiffusion — Gaussian noise in embedding space (ε-prediction / x₀-prediction)
  • FlowMatching — Rectified Flow in embedding space (velocity prediction)

Design notes
------------
All three share a common ``TransformerConfig`` for the denoising backbone.
The backbone uses **bidirectional** (non-causal) attention — unlike autoregressive
LMs, diffusion models process all positions in parallel per denoising step.

Tessera compiler integration flags mirror GemmaConfig: set
``use_tessera_compile=True`` to route the backbone attention / matmuls through
the tessera-compile CLI pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional


# ---------------------------------------------------------------------------
# Shared backbone config
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """Bidirectional transformer backbone used by all three model variants."""

    # Vocabulary
    vocab_size: int = 50_257          # GPT-2 default; set to match tokenizer
    mask_token_id: int = 50_256       # [MASK] token for MDLM

    # Architecture
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_kv_heads: int = 12            # set < num_attention_heads for GQA
    head_dim: Optional[int] = None    # None → hidden_size // num_attention_heads
    intermediate_size: int = 3_072    # MLP hidden; typically 4× hidden_size
    mlp_type: Literal["swiglu", "geglu"] = "swiglu"
    max_position_embeddings: int = 2_048
    rms_norm_eps: float = 1e-6
    dropout_p: float = 0.1

    # Time / noise-level conditioning style
    # "adaLN"  — adaptive layer-norm (DiT-style): scale+shift from time MLP
    # "concat" — concatenate time embedding to each token embedding
    # "add"    — add time embedding to token embeddings
    time_cond_style: Literal["adaLN", "concat", "add"] = "adaLN"

    # Tessera compiler
    use_tessera_compile: bool = False
    tessera_pipeline: str = "full"
    tessera_arch: str = "sm_90"
    tessera_platform: str = "cuda"

    def __post_init__(self) -> None:
        if self.head_dim is None:
            assert self.hidden_size % self.num_attention_heads == 0
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_kv_heads == 0

    @property
    def groups(self) -> int:
        return self.num_attention_heads // self.num_kv_heads

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    # --- Factory methods ---

    @classmethod
    def small(cls) -> "TransformerConfig":
        """~117M param backbone — comparable to GPT-2 small."""
        return cls(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                   intermediate_size=3_072)

    @classmethod
    def medium(cls) -> "TransformerConfig":
        """~345M param backbone — comparable to GPT-2 medium."""
        return cls(hidden_size=1_024, num_hidden_layers=24, num_attention_heads=16,
                   intermediate_size=4_096)

    @classmethod
    def large(cls) -> "TransformerConfig":
        """~762M param backbone."""
        return cls(hidden_size=1_280, num_hidden_layers=36, num_attention_heads=20,
                   intermediate_size=5_120)

    @classmethod
    def debug_tiny(cls) -> "TransformerConfig":
        """Very small config for fast CPU unit tests."""
        return cls(
            vocab_size=32_000, mask_token_id=31_999,
            hidden_size=256, num_hidden_layers=4,
            num_attention_heads=4, num_kv_heads=2,
            intermediate_size=512, max_position_embeddings=512,
            dropout_p=0.0,
        )


# ---------------------------------------------------------------------------
# MDLM — Masked Discrete Diffusion Language Model
# ---------------------------------------------------------------------------

@dataclass
class MDLMConfig:
    """
    Configuration for Masked Discrete Diffusion LM (MDLM).

    MDLM uses an *absorbing-state* forward process:
      q(x_t | x_0) = Mask(x_0) with probability m(t),  else x_0

    where m(t) is the masking schedule.  The denoising model learns to
    predict the original token at each masked position given the partially-
    masked sequence and noise level t.

    Reference: Shi et al. (2024) "Simplified and Generalized Masked Diffusion
    for Discrete Data". https://arxiv.org/abs/2406.04329
    """
    transformer: TransformerConfig = field(default_factory=TransformerConfig.small)

    # Diffusion schedule
    num_timesteps: int = 1_000
    mask_schedule: Literal["linear", "cosine"] = "cosine"

    # Loss options
    loss_type: Literal["ce", "elbo"] = "elbo"  # ELBO = time-weighted CE
    reweight_loss: bool = True    # weight loss by 1/m(t) for uniform SNR

    # Self-conditioning (predict x_0, then re-use as additional input)
    self_condition: bool = True
    self_cond_prob: float = 0.5   # probability of using self-conditioning during training

    @classmethod
    def debug_tiny(cls) -> "MDLMConfig":
        return cls(transformer=TransformerConfig.debug_tiny(),
                   num_timesteps=100, mask_schedule="linear")


# ---------------------------------------------------------------------------
# Continuous Embedding Diffusion
# ---------------------------------------------------------------------------

@dataclass
class ContinuousDiffusionConfig:
    """
    Configuration for continuous Gaussian diffusion in embedding space.

    Uses standard DDPM/DDIM with ε-prediction (default) or x₀-prediction.
    The forward process adds Gaussian noise to token embeddings; the model
    learns to denoise.

    Reference: Li et al. (2022) "Diffusion-LM Improves Controllable Text
    Generation". https://arxiv.org/abs/2205.14217
    """
    transformer: TransformerConfig = field(default_factory=TransformerConfig.small)

    # Diffusion schedule
    num_timesteps: int = 2_000
    beta_schedule: Literal["linear", "cosine", "sqrt"] = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Prediction target
    prediction_type: Literal["epsilon", "x_start", "v"] = "epsilon"

    # Learned variance (à la Improved DDPM)
    learned_variance: bool = True
    lambda_vlb: float = 0.001     # weight of variational lower bound term

    # Embedding rounding / projection back to token space
    logits_proj_type: Literal["linear", "norm_linear"] = "norm_linear"

    @classmethod
    def debug_tiny(cls) -> "ContinuousDiffusionConfig":
        return cls(transformer=TransformerConfig.debug_tiny(),
                   num_timesteps=200, learned_variance=False)


# ---------------------------------------------------------------------------
# Flow Matching
# ---------------------------------------------------------------------------

@dataclass
class FlowMatchingConfig:
    """
    Configuration for Flow Matching LM (Rectified Flow / OT-CFM).

    The model learns a velocity field v_θ(x_t, t) such that integrating
    from t=1 (noise) to t=0 (data) recovers clean embeddings.

    Default: linear interpolation (Rectified Flow):
        x_t = (1-t)·x_0 + t·ε,    ε ~ N(0,I)
        target velocity = ε - x_0

    Reference: Liu et al. (2022) "Flow Straight and Fast: Learning to Generate
    and Transfer Data with Rectified Flow". https://arxiv.org/abs/2209.03003
    """
    transformer: TransformerConfig = field(default_factory=TransformerConfig.small)

    # Interpolation type
    interpolation: Literal["linear", "cosine"] = "linear"

    # Training timestep distribution
    t_distribution: Literal["uniform", "logit_normal"] = "logit_normal"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0

    # ODE sampling
    num_sampling_steps: int = 50    # Euler integration steps
    solver: Literal["euler", "midpoint", "rk4"] = "euler"

    @classmethod
    def debug_tiny(cls) -> "FlowMatchingConfig":
        return cls(transformer=TransformerConfig.debug_tiny(),
                   num_sampling_steps=10)
