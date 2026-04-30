from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CachePolicy:
    k_bits: int
    v_bits: int
    residual_bits: int
    retrieval_head_fraction: float
    streaming_window: int


def estimate_request_cache(heads: int, head_dim: int, context_tokens: int, policy: CachePolicy) -> float:
    retrieval_heads = max(1, round(heads * policy.retrieval_head_fraction))
    streaming_heads = heads - retrieval_heads
    retrieval_bits = retrieval_heads * context_tokens * head_dim * 2 * (policy.k_bits + policy.residual_bits)
    streaming_bits = streaming_heads * min(context_tokens, policy.streaming_window) * head_dim * 2 * policy.v_bits
    return (retrieval_bits + streaming_bits) / 8.0
