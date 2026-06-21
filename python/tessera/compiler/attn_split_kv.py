"""tessera.compiler.attn_split_kv — tail-KV split (flash-decoding) cost model.

When the number of independent attention work-blocks (``batch × heads ×
q_blocks``) does not divide evenly across the device's compute units, the last
wave leaves CUs idle.  The moonmath CDNA3 attention writeup recovers that idle
parallelism by splitting the stranded blocks' K/V range into ``G`` parts spread
across the idle CUs; each part computes a *partial* output plus its online-
softmax statistics ``(m, l)``, and a small **merge kernel** rescales and combines
the partials (the standard FlashAttention split-K / flash-decoding pattern).

This module is the host-side **cost model** that decides ``G`` from grid
occupancy.  It deliberately *declines* (``G = 1``) when:

* the last wave is already ≥ ``occupancy_threshold`` full (default 95%) — there
  is no meaningful idle parallelism to recover, so the merge overhead is waste; or
* the sequence is too short (``kv_len < min_kv_len``) — splitting a short K/V
  range pays merge cost it cannot earn back.

Pure data + arithmetic, stdlib only (no ``tessera`` imports) so the planner and
the runtime can both import it.  The actual partial/merge kernels reuse the
online-softmax machinery already in the attention lowering; this module only
sizes the split and records the merge contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


class TesseraSplitKVError(ValueError):
    """Raised for malformed split-KV cost-model inputs."""


def _last_wave_occupancy(blocks: int, num_cu: int) -> float:
    """Fraction of CUs busy in the final wave (1.0 when blocks divide evenly)."""
    rem = blocks % num_cu
    return 1.0 if rem == 0 else rem / num_cu


@dataclass(frozen=True)
class SplitKVPlan:
    """The chosen split factor + the occupancy/merge bookkeeping behind it.

    ``split_factor`` (``G``) is 1 when the model declines to split.  When
    ``G > 1`` each original block becomes ``G`` partial computations and a merge
    kernel combines them via online-softmax rescale (``merge_partials`` partials
    total).
    """

    split_factor: int
    grid_blocks: int
    num_cu: int
    kv_len: int
    occupancy_before: float
    occupancy_after: float
    reason: str

    @property
    def is_split(self) -> bool:
        return self.split_factor > 1

    @property
    def effective_blocks(self) -> int:
        """Work-blocks after splitting (``grid_blocks × G``)."""
        return self.grid_blocks * self.split_factor

    @property
    def merge_partials(self) -> int:
        """Number of partial results the merge kernel combines (0 when no split)."""
        return self.effective_blocks if self.is_split else 0

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "kind": "split_kv_plan",
            "split_factor": self.split_factor,
            "grid_blocks": self.grid_blocks,
            "effective_blocks": self.effective_blocks,
            "num_cu": self.num_cu,
            "kv_len": self.kv_len,
            "occupancy_before": round(self.occupancy_before, 4),
            "occupancy_after": round(self.occupancy_after, 4),
            "merge_partials": self.merge_partials,
            "is_split": self.is_split,
            "reason": self.reason,
        }


def plan_split_kv(
    grid_blocks: int,
    num_cu: int,
    kv_len: int,
    *,
    kv_block: int = 128,
    max_split: int = 8,
    occupancy_threshold: float = 0.95,
    min_kv_len: int = 1024,
) -> SplitKVPlan:
    """Decide the tail-KV split factor ``G`` from grid occupancy.

    Args:
        grid_blocks: independent attention work-blocks before any split
            (``batch × heads × q_blocks``).
        num_cu: compute units available on the device (e.g. 304 on MI300X).
        kv_len: K/V sequence length being attended over.
        kv_block: K/V tile size; bounds the split (cannot exceed the number of
            KV blocks ``ceil(kv_len / kv_block)``).
        max_split: hard cap on ``G`` (merge cost grows with it).
        occupancy_threshold: decline to split when the last wave is at least this
            full (default 0.95 — the writeup's "last round > 95% full" rule).
        min_kv_len: decline to split sequences shorter than this (merge overhead
            outweighs the recovered parallelism).

    Returns:
        A :class:`SplitKVPlan`.  ``G = 1`` (with a ``reason``) when the model
        declines; otherwise the ``G`` that maximizes last-wave occupancy,
        tie-broken toward the *smaller* ``G`` (less merge overhead).
    """
    if grid_blocks <= 0 or num_cu <= 0 or kv_len <= 0:
        raise TesseraSplitKVError(
            f"plan_split_kv: grid_blocks ({grid_blocks}), num_cu ({num_cu}) and "
            f"kv_len ({kv_len}) must all be positive")
    if kv_block <= 0:
        raise TesseraSplitKVError(f"plan_split_kv: kv_block must be positive, got {kv_block}")
    if max_split < 1:
        raise TesseraSplitKVError(f"plan_split_kv: max_split must be >= 1, got {max_split}")
    if not (0.0 < occupancy_threshold <= 1.0):
        raise TesseraSplitKVError(
            f"plan_split_kv: occupancy_threshold must be in (0, 1], got {occupancy_threshold}")

    occ_before = _last_wave_occupancy(grid_blocks, num_cu)

    def decline(reason: str) -> SplitKVPlan:
        return SplitKVPlan(
            split_factor=1,
            grid_blocks=grid_blocks,
            num_cu=num_cu,
            kv_len=kv_len,
            occupancy_before=occ_before,
            occupancy_after=occ_before,
            reason=reason,
        )

    # Rule 1: last wave already (nearly) full → nothing to recover.
    if occ_before >= occupancy_threshold:
        return decline(
            f"last wave {occ_before:.0%} full (>= {occupancy_threshold:.0%} threshold)")

    # Rule 2: grid already saturates the device many times over — a single
    # underfull tail wave out of many is not worth a global split.
    if grid_blocks >= num_cu:
        return decline(
            f"grid ({grid_blocks}) >= num_cu ({num_cu}); only the tail wave is "
            f"underfull — not worth a global K/V split")

    # Rule 3: sequence too short for the merge to pay off.
    if kv_len < min_kv_len:
        return decline(
            f"kv_len ({kv_len}) < min_kv_len ({min_kv_len}) — merge overhead "
            f"outweighs recovered parallelism")

    # Choose G: maximize last-wave occupancy of (grid_blocks * G), bounded by the
    # number of KV blocks (cannot split finer than that) and max_split.
    kv_blocks = max(1, ceil(kv_len / kv_block))
    g_cap = min(max_split, kv_blocks)
    if g_cap < 2:
        return decline(
            f"kv_len ({kv_len}) spans < 2 KV blocks of {kv_block} — nothing to split")

    best_g = 1
    best_occ = occ_before
    for g in range(2, g_cap + 1):
        occ = _last_wave_occupancy(grid_blocks * g, num_cu)
        if occ > best_occ + 1e-9:  # strictly better; ties keep the smaller G
            best_occ = occ
            best_g = g

    if best_g == 1:
        return decline(
            f"no split factor up to {g_cap} improves last-wave occupancy "
            f"({occ_before:.0%})")

    return SplitKVPlan(
        split_factor=best_g,
        grid_blocks=grid_blocks,
        num_cu=num_cu,
        kv_len=kv_len,
        occupancy_before=occ_before,
        occupancy_after=best_occ,
        reason=(
            f"split K/V {best_g}-way: last-wave occupancy "
            f"{occ_before:.0%} -> {best_occ:.0%}"),
    )


__all__ = [
    "SplitKVPlan",
    "TesseraSplitKVError",
    "plan_split_kv",
]
