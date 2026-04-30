from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HeadStats:
    retrieval_mass: float
    recent_mass: float
    sink_mass: float


def synthetic_head_stats(num_heads: int) -> list[HeadStats]:
    stats = []
    for h in range(num_heads):
        phase = h / max(num_heads - 1, 1)
        retrieval = 0.15 + 0.70 * max(0.0, math.sin(phase * math.pi))
        recent = 0.70 - 0.40 * retrieval
        sink = max(0.05, 1.0 - retrieval - recent)
        stats.append(HeadStats(round(retrieval, 3), round(recent, 3), round(sink, 3)))
    return stats


def classify_heads(stats: list[HeadStats], retrieval_threshold: float = 0.55) -> list[str]:
    roles = []
    for stat in stats:
        if stat.retrieval_mass >= retrieval_threshold:
            roles.append("retrieval")
        elif stat.sink_mass >= 0.15:
            roles.append("sink_stream")
        else:
            roles.append("streaming")
    return roles


def estimate_cache_bytes(
    heads: int,
    seq_len: int,
    head_dim: int,
    active_tokens: int,
    sink_tokens: int,
    bytes_per_value: int = 2,
) -> int:
    tokens = min(seq_len, active_tokens + sink_tokens)
    # K and V are both cached.
    return heads * tokens * head_dim * 2 * bytes_per_value
