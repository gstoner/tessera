"""Architecture-owned Apple SIMD-group matrix fragment contracts.

This is deliberately narrower than a general GEMM scheduler: it describes the
physical unit that a portable Tile materializer may select on Apple7+ without
borrowing NVIDIA or AMD lane maps.  The backing MSL emitter owns the eventual
packing/store implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .apple_target import AppleGPUArch, AppleGPUTargetProfile


class AppleFragmentError(ValueError):
    """No exact Apple physical fragment accepts the requested Tile contract."""


@dataclass(frozen=True)
class AppleSimdgroupFragment:
    arch: AppleGPUArch
    storage_dtype: str
    accumulator_dtype: str
    m: int = 8
    n: int = 8
    k: int = 8
    lanes: int = 32
    threadgroup: tuple[int, int, int] = (32, 1, 1)

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "family": "simdgroup_matrix",
            "arch": self.arch.name.lower(),
            "storage_dtype": self.storage_dtype,
            "accumulator_dtype": self.accumulator_dtype,
            "shape": (self.m, self.n, self.k),
            "lanes": self.lanes,
            "threadgroup": self.threadgroup,
        }


@dataclass(frozen=True)
class AppleTileResourceRecord:
    """Target-owned launch and threadgroup-memory record for one Tile artifact."""

    threadgroup: tuple[int, int, int]
    simdgroup_lanes: int
    staged_a_bytes: int
    staged_b_bytes: int
    edge_scratch_bytes: int
    total_threadgroup_bytes: int
    target_threadgroup_capacity_bytes: int
    double_buffered: bool
    partial_edge_store: bool

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "threadgroup": self.threadgroup,
            "simdgroup_lanes": self.simdgroup_lanes,
            "staged_a_bytes": self.staged_a_bytes,
            "staged_b_bytes": self.staged_b_bytes,
            "edge_scratch_bytes": self.edge_scratch_bytes,
            "total_threadgroup_bytes": self.total_threadgroup_bytes,
            "target_threadgroup_capacity_bytes": self.target_threadgroup_capacity_bytes,
            "double_buffered": self.double_buffered,
            "partial_edge_store": self.partial_edge_store,
        }


@dataclass(frozen=True)
class AppleTilePromotionEvidence:
    """Two warm runs of one exact Tile candidate in a single timing domain.

    A false counter capability is valid evidence: it records the hardware
    limitation rather than inventing occupancy or spill metrics.  A missing
    capability field is not evidence and therefore cannot promote a route.
    """

    route: Literal["mps", "simdgroup_matrix"]
    dtype: str
    shape: tuple[int, int, int]
    timing_domain: Literal["end_to_end", "kernel"]
    native_gpu: bool
    numerically_validated: bool
    placement_validated: bool
    run_medians_ns: tuple[int, int]
    resource_record: Mapping[str, object]
    counter_sampling_supported: bool | None
    counter_timestamp_deltas: tuple[int | None, int | None]


def select_apple_tile_promotion(
    mps: AppleTilePromotionEvidence,
    simdgroup: AppleTilePromotionEvidence,
    *,
    minimum_win_fraction: float = 0.05,
) -> str:
    """Select simdgroup only after two comparable, proven warm-run wins.

    The incumbent MPS route is retained for every incomplete or mixed evidence
    row.  This intentionally makes the decision independent for end-to-end and
    kernel timing; callers must name the domain they intend to optimize.
    """
    if not 0.0 < minimum_win_fraction < 1.0:
        raise ValueError("minimum_win_fraction must be in (0, 1)")
    if mps.route != "mps" or simdgroup.route != "simdgroup_matrix":
        raise ValueError("promotion compares MPS incumbent with simdgroup_matrix")
    comparable = (
        mps.dtype == simdgroup.dtype
        and mps.shape == simdgroup.shape
        and mps.timing_domain == simdgroup.timing_domain
    )
    if not comparable:
        return "mps"
    for evidence in (mps, simdgroup):
        if not (evidence.native_gpu and evidence.numerically_validated
                and evidence.placement_validated and evidence.resource_record
                and evidence.counter_sampling_supported is not None):
            return "mps"
        if len(evidence.run_medians_ns) != 2 or any(ns <= 0 for ns in evidence.run_medians_ns):
            return "mps"
        if evidence.counter_sampling_supported and any(
                delta is None or delta <= 0 for delta in evidence.counter_timestamp_deltas):
            return "mps"
        if not evidence.counter_sampling_supported and any(
                delta is not None for delta in evidence.counter_timestamp_deltas):
            return "mps"
    threshold = 1.0 - minimum_win_fraction
    if all(simd < incumbent * threshold
           for simd, incumbent in zip(simdgroup.run_medians_ns, mps.run_medians_ns)):
        return "simdgroup_matrix"
    return "mps"


def select_apple_simdgroup_fragment(
    target: AppleGPUTargetProfile, storage_dtype: str, *, accumulator_dtype: str = "fp32",
) -> AppleSimdgroupFragment:
    """Select the exact Apple7+ 8x8x8 fragment for a logical Tile MMA.

    Inputs are f16/bf16, accumulation is f32.  Tile edge handling is outside
    the fragment itself and must be supplied by the selected materializer.
    """
    aliases = {"f16": "fp16", "bf16": "bf16"}
    storage = aliases.get(storage_dtype, storage_dtype)
    if not target.supports_simdgroup_matrix:
        raise AppleFragmentError(
            f"APPLE_FRAGMENT_UNSUPPORTED_ARCH: {target.arch.name.lower()} has no simdgroup_matrix")
    if storage not in {"fp16", "bf16"}:
        raise AppleFragmentError(
            f"APPLE_FRAGMENT_UNSUPPORTED_DTYPE: {storage_dtype!r} needs fp16 or bf16 storage")
    if accumulator_dtype not in {"fp32", "f32"}:
        raise AppleFragmentError(
            "APPLE_FRAGMENT_UNSUPPORTED_ACCUMULATOR: Apple simdgroup Tile fragments require fp32")
    return AppleSimdgroupFragment(target.arch, storage, "fp32")


__all__ = [
    "AppleFragmentError", "AppleSimdgroupFragment", "AppleTilePromotionEvidence",
    "AppleTileResourceRecord", "select_apple_simdgroup_fragment",
    "select_apple_tile_promotion",
]
