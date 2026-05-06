"""Configuration objects for the current Fast dLLM v2 compiler smoke."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FastDLLMConfig:
    vocab_size: int = 257
    hidden_size: int = 64
    intermediate_size: int = 128
    branch_count: int = 4
    block_tokens: int = 8
    decode_steps: int = 6
    confidence_tau: float = 0.62
    topk_margin_tau: float = 0.025
    rms_norm_eps: float = 1.0e-5


def tiny_config() -> FastDLLMConfig:
    """Return a deterministic tiny config used by tests and smoke scripts."""

    return FastDLLMConfig()
