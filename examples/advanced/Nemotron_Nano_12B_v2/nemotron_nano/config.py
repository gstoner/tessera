"""Configuration objects for the Nemotron Nano sample.

The production model dimensions are documented in ``MODEL_SPECS.md``.  The
helpers here intentionally default to a small shape so the example can run in
the repository test environment without model weights or PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NemotronNanoConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    hybrid_override_pattern: str
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if len(self.hybrid_override_pattern) != self.num_hidden_layers:
            raise ValueError("hybrid_override_pattern length must equal num_hidden_layers")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")


def tiny_config() -> NemotronNanoConfig:
    """Return a deterministic tiny M/*/- stack for smoke tests."""

    return NemotronNanoConfig(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=3,
        hybrid_override_pattern="M*-",
    )
