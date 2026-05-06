"""Configuration for the current FlashMLA compiler sample."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLAConfig:
    batch_size: int = 2
    seq_len: int = 8
    model_dim: int = 64
    latent_dim: int = 16
    num_q_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 16
    rope_dim: int = 4
    rms_norm_eps: float = 1.0e-5


def tiny_config() -> MLAConfig:
    """Return a deterministic tiny config used by smoke tests."""

    return MLAConfig()
