"""
tessera_gemma/configs.py — Model configuration for Tessera-Gemma.

Covers Gemma 4 (4B / 12B / 27B) and a debug-tiny variant.
Key Gemma 4 features captured here:
  • head_dim = 256 (decoupled from hidden_size / num_heads)
  • Alternating sliding-window attention (SWA) / full-attention layers
  • GeGLU MLP (gate + up + down projection, no packing)
  • GQA (num_kv_heads < num_attention_heads)
  • NTK-scaled RoPE for long context (rope_scaling)
  • Optional Tessera compiler path (use_tessera_compile)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

AttentionKind = Literal["full", "sliding_window"]
MlpKind       = Literal["swiglu", "geglu"]
SWAPattern    = Literal["alternating", "all", "none"]


@dataclass
class GemmaConfig:
    # -----------------------------------------------------------------------
    # Vocabulary / embedding
    # -----------------------------------------------------------------------
    vocab_size: int = 256_000
    tie_word_embeddings: bool = True

    # -----------------------------------------------------------------------
    # Architecture dimensions
    # -----------------------------------------------------------------------
    hidden_size: int = 3_072
    intermediate_size: int = 24_576
    num_hidden_layers: int = 46

    # Attention — GQA: num_kv_heads <= num_attention_heads
    num_attention_heads: int = 32
    num_kv_heads: int = 16
    head_dim: int = 256          # explicit; Gemma4 decouples this from hidden_size/H

    # MLP variant
    mlp_type: MlpKind = "geglu"

    # -----------------------------------------------------------------------
    # Position embeddings
    # -----------------------------------------------------------------------
    rope_theta: float = 10_000.0
    # rope_scaling: None = no scaling; dict with keys "type", "factor"
    rope_scaling: Optional[Dict] = None
    max_position_embeddings: int = 131_072

    # -----------------------------------------------------------------------
    # Normalization / regularisation
    # -----------------------------------------------------------------------
    rms_norm_eps: float = 1e-6
    dropout_p: float = 0.0

    # -----------------------------------------------------------------------
    # Sliding-window attention
    # -----------------------------------------------------------------------
    sliding_window_size: Optional[int] = 4_096
    # "alternating": even layers = full, odd layers = SWA (0-indexed)
    # "all":         every layer uses SWA
    # "none":        every layer uses full attention
    sliding_window_pattern: SWAPattern = "alternating"

    # -----------------------------------------------------------------------
    # Special tokens
    # -----------------------------------------------------------------------
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1

    # -----------------------------------------------------------------------
    # Flash / Tessera kernel selection
    # -----------------------------------------------------------------------
    use_flash: bool = True  # prefer torch.nn.functional.scaled_dot_product_attention

    # -----------------------------------------------------------------------
    # Tessera compiler integration
    # -----------------------------------------------------------------------
    use_tessera_compile: bool = False   # route through tessera-compile CLI
    tessera_pipeline: str = "full"      # CLI pipeline alias
    tessera_arch: str = "sm_90"         # target arch for tessera-compile
    tessera_platform: str = "cuda"

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------
    @property
    def q_dim(self) -> int:
        """Total Q projection dimension."""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """Total KV projection dimension."""
        return self.num_kv_heads * self.head_dim

    @property
    def groups(self) -> int:
        """Number of Q-head groups per KV head (GQA ratio)."""
        assert self.num_attention_heads % self.num_kv_heads == 0
        return self.num_attention_heads // self.num_kv_heads

    def layer_attention_kind(self, layer_idx: int) -> AttentionKind:
        """Return whether layer *layer_idx* should use sliding-window attention."""
        if (
            self.sliding_window_size is None
            or self.sliding_window_pattern == "none"
        ):
            return "full"
        if self.sliding_window_pattern == "all":
            return "sliding_window"
        # "alternating": layer 0 = full, layer 1 = SWA, layer 2 = full, …
        return "sliding_window" if layer_idx % 2 == 1 else "full"

    # -----------------------------------------------------------------------
    # Factory methods — named Gemma 4 configurations
    # -----------------------------------------------------------------------
    @classmethod
    def gemma4_4b(cls) -> "GemmaConfig":
        """Gemma 4 4B — 34 layers, hidden=2560, head_dim=256, GeGLU, SWA."""
        return cls(
            hidden_size=2_560,
            intermediate_size=10_240,
            num_hidden_layers=34,
            num_attention_heads=16,
            num_kv_heads=8,
            head_dim=256,
            mlp_type="geglu",
            sliding_window_size=4_096,
            sliding_window_pattern="alternating",
            rope_theta=10_000.0,
            max_position_embeddings=131_072,
        )

    @classmethod
    def gemma4_12b(cls) -> "GemmaConfig":
        """Gemma 4 12B — 46 layers, hidden=3840, head_dim=256, GeGLU, SWA."""
        return cls(
            hidden_size=3_840,
            intermediate_size=15_360,
            num_hidden_layers=46,
            num_attention_heads=24,
            num_kv_heads=8,
            head_dim=256,
            mlp_type="geglu",
            sliding_window_size=4_096,
            sliding_window_pattern="alternating",
            rope_theta=10_000.0,
            max_position_embeddings=131_072,
        )

    @classmethod
    def gemma4_27b(cls) -> "GemmaConfig":
        """Gemma 4 27B — 62 layers, hidden=5632, head_dim=256, GeGLU, SWA."""
        return cls(
            hidden_size=5_632,
            intermediate_size=22_528,
            num_hidden_layers=62,
            num_attention_heads=32,
            num_kv_heads=16,
            head_dim=256,
            mlp_type="geglu",
            sliding_window_size=4_096,
            sliding_window_pattern="alternating",
            rope_theta=10_000.0,
            max_position_embeddings=131_072,
        )

    @classmethod
    def debug_tiny(cls) -> "GemmaConfig":
        """Tiny config for fast unit tests — CPU runnable, no CUDA required."""
        return cls(
            vocab_size=32_000,
            hidden_size=512,
            intermediate_size=2_048,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_kv_heads=2,
            head_dim=64,
            mlp_type="swiglu",
            sliding_window_size=256,
            sliding_window_pattern="alternating",
            rope_theta=10_000.0,
            max_position_embeddings=2_048,
        )
