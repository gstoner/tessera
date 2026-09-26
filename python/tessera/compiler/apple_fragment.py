"""Architecture-owned Apple SIMD-group matrix fragment contracts.

This is deliberately narrower than a general GEMM scheduler: it describes the
physical unit that a portable Tile materializer may select on Apple7+ without
borrowing NVIDIA or AMD lane maps.  The backing MSL emitter owns the eventual
packing/store implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

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


#: Accumulator byte widths for the admitted accumulators.
ACCUMULATOR_BYTES: dict[str, int] = {"fp32": 4, "fp16": 2}

#: Simdgroup accumulators this lane admits, per storage dtype. Every entry is a
#: measured fact on Apple7 (M1 Max, macOS 27.0, Metal toolchain 32023.921), not
#: an inference from the header alone (APPLE-ACCUM-1 in
#: docs/audit/backend/apple/todo.md):
#:
#: * ``metal_simdgroup_matrix`` (Metal toolchain headers) constrains
#:   ``simdgroup_multiply_accumulate`` only by ``is_floating_point_v`` on every
#:   operand and declares ``simdgroup_matrix`` for half, bfloat and float, so
#:   every half/bfloat/float storage x accumulator pair compiles at MSL 3.1 (the
#:   runtime's compile version) and every integer accumulator fails to compile.
#: * Run on the GPU against a numpy model of the same arithmetic, an fp32
#:   accumulator is bit-exact with a sequential fp32 FMA chain. An fp16
#:   accumulator is bit-exact with a sequential fp16 FMA chain for fp16
#:   storage, and with "fp32 inside each 8-deep MMA, round-to-nearest-even to
#:   fp16 after every MMA" for bf16 storage. Both are genuine fp16
#:   accumulation: every partial sum carried between MMAs is an fp16 value.
#: * A **bf16** accumulator is not bf16 accumulation on this part: the result is
#:   bit-exact with fp32 accumulation carried across the whole K loop and
#:   truncated (round-toward-zero) to bf16 at ``simdgroup_store``. Admitting it
#:   would execute a different numeric class -- fp32 accumulation with an RTZ
#:   output rounding -- than the program declared, so it is refused.
#:
#: ``tests/unit/test_apple_simdgroup_accumulator_device.py`` re-derives every
#: admitted row on the device.
SIMDGROUP_ACCUMULATORS: dict[str, tuple[str, ...]] = {
    "fp16": ("fp32", "fp16"),
    "bf16": ("fp32", "fp16"),
}

_REFUSED_ACCUMULATOR_REASONS: dict[str, str] = {
    "bf16": (
        "Apple7 does not accumulate a simdgroup_matrix<bfloat> in bf16 -- measured "
        "on the M1 Max it carries fp32 across the K loop and truncates "
        "(round-toward-zero) to bf16 at simdgroup_store, so accum=bf16 would run "
        "fp32 accumulation with an RTZ output instead of the declared bf16 "
        "accumulation"
    ),
    "int32": (
        "simdgroup_matrix has no integer element type (metal_simdgroup_matrix "
        "requires is_floating_point_v; an int accumulator does not compile)"
    ),
}


def canonical_accumulator_dtype(accumulator_dtype: str) -> str:
    """Normalize an accumulator spelling to its canonical Decision #15a name.

    One mapping for the whole stack: ``tessera.dtype.canonicalize_dtype``
    (canonical names plus ``_DTYPE_ALIASES``, one lowercase fold). The C++
    side (``appleAccumulatorType``) accepts exactly its fp32/fp16/bf16
    spellings, drift-gated by
    ``test_apple_accumulator_spellings_match_tessera_dtype``. An unknown
    spelling is returned unchanged so the caller refuses it by name.
    """
    from tessera.dtype import TesseraDtypeError, canonicalize_dtype

    try:
        return canonicalize_dtype(accumulator_dtype)
    except (TesseraDtypeError, TypeError):
        return accumulator_dtype


def select_apple_simdgroup_fragment(
    target: AppleGPUTargetProfile, storage_dtype: str, *, accumulator_dtype: str,
) -> AppleSimdgroupFragment:
    """Select the exact Apple7+ 8x8x8 fragment for a logical Tile MMA.

    ``accumulator_dtype`` is the program's ``numeric_policy.accum`` and is
    required: the accumulator selects semantics (Decision #21a), so this
    boundary never supplies one. Storage is fp16/bf16 and the admitted
    accumulators are :data:`SIMDGROUP_ACCUMULATORS`. Tile edge handling is
    outside the fragment itself and must be supplied by the selected
    materializer.
    """
    aliases = {"f16": "fp16", "bf16": "bf16"}
    storage = aliases.get(storage_dtype, storage_dtype)
    arch = target.arch.name.lower()
    if not target.supports_simdgroup_matrix:
        raise AppleFragmentError(
            f"APPLE_FRAGMENT_UNSUPPORTED_ARCH: {arch} has no simdgroup_matrix")
    if storage not in SIMDGROUP_ACCUMULATORS:
        raise AppleFragmentError(
            f"APPLE_FRAGMENT_UNSUPPORTED_DTYPE: {storage_dtype!r} needs fp16 or bf16 storage")
    accum = canonical_accumulator_dtype(accumulator_dtype)
    if accum not in SIMDGROUP_ACCUMULATORS[storage]:
        reason = _REFUSED_ACCUMULATOR_REASONS.get(
            accum, "no simdgroup_matrix accumulator of that type is proven on this lane")
        raise AppleFragmentError(
            "APPLE_FRAGMENT_UNSUPPORTED_ACCUMULATOR: apple_gpu simdgroup_matrix "
            f"({arch}) accepts accum in {list(SIMDGROUP_ACCUMULATORS[storage])} for "
            f"storage={storage}; got accum={accumulator_dtype!r}: {reason}")
    return AppleSimdgroupFragment(target.arch, storage, accum)


__all__ = [
    "ACCUMULATOR_BYTES", "AppleFragmentError", "AppleSimdgroupFragment",
    "AppleTilePromotionEvidence", "AppleTileResourceRecord",
    "SIMDGROUP_ACCUMULATORS", "canonical_accumulator_dtype",
    "select_apple_simdgroup_fragment", "select_apple_tile_promotion",
]
