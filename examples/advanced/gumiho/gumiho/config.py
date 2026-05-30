"""Gumiho draft-model geometry.

Tiny by design — the point of this example is the *hybrid draft architecture*
and its end-to-end execution on the Apple GPU/CPU compiler backend, not model
scale. Dimensions are small enough to validate every tensor against a float64
numpy reference, large enough that the serial/parallel split is real.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GumihoConfig:
    # Target ("big") model geometry.
    vocab: int = 32
    d_model: int = 16
    num_heads: int = 2
    ffn_hidden: int = 32
    rmsnorm_eps: float = 1e-5

    # Hybrid draft heads (Gumiho, ICML'25).
    #   serial_tokens : EAGLE-style 2-layer Transformer, autoregressive.
    #   parallel_heads: Medusa-style MLPs predicting the remaining positions.
    serial_layers: int = 2
    serial_tokens: int = 2
    parallel_heads: int = 5
    parallel_hidden: int = 24

    # Full Tree Attention: top-k tokens per parallel head, top-n paths kept.
    fta_tokens_per_head: int = 2
    fta_top_paths: int = 8

    # Prompt length for the demo context.
    context_len: int = 4

    @property
    def head_dim(self) -> int:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        return self.d_model // self.num_heads

    @property
    def total_draft_tokens(self) -> int:
        return self.serial_tokens + self.parallel_heads


def tiny_config() -> GumihoConfig:
    return GumihoConfig()
