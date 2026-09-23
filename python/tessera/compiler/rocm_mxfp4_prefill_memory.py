"""Model-level weight-residency accounting for manual gfx1201 prefill trials."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MXFP4LayerShape:
    """One canonical MXFP4 linear layer; count may share its shape."""

    n: int
    k: int
    count: int = 1

    def __post_init__(self) -> None:
        if min(self.n, self.k, self.count) <= 0 or self.n % 16 or self.k % 32:
            raise ValueError("MXFP4 residency requires positive N16, K32, and count")


def assess_prefill_weight_residency(
    layers: Iterable[MXFP4LayerShape], *, available_extra_bytes: int,
    retain_packed_for_decode: bool = True,
) -> dict[str, object]:
    """Account for both representations; never authorize a performance selector.

    Packed decode remains resident by default. Package images, activations,
    fragmentation, and other model buffers are outside this weight-only budget.
    """
    if available_extra_bytes < 0:
        raise ValueError("available extra weight bytes must be nonnegative")
    packed_bytes = 0
    expanded_bytes = 0
    layer_count = 0
    for layer in layers:
        layer_count += layer.count
        packed_bytes += layer.count * (layer.n * layer.k // 2 + (layer.k // 32 + 1) * layer.n)
        expanded_bytes += layer.count * (layer.n * layer.k + layer.n)
    if not layer_count:
        raise ValueError("MXFP4 residency requires at least one layer")
    extra_bytes = (
        expanded_bytes if retain_packed_for_decode
        else expanded_bytes - packed_bytes
    )
    return {
        "layer_count": layer_count,
        "packed_weight_and_scale_bytes": packed_bytes,
        "expanded_weight_and_reference_bytes": expanded_bytes,
        "both_resident_bytes": packed_bytes + expanded_bytes,
        "retain_packed_for_decode": retain_packed_for_decode,
        "incremental_bytes": extra_bytes,
        "available_extra_bytes": available_extra_bytes,
        "budget_allows_trial": extra_bytes <= available_extra_bytes,
        "selection_state": "manual_trial_only" if extra_bytes <= available_extra_bytes else "refused_budget",
    }


__all__ = ["MXFP4LayerShape", "assess_prefill_weight_residency"]
