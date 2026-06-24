"""tessera.compiler.rocm_tiling — register-budget-aware tiling (B1).

The AMD Gluon GEMM tutorial taught one dominant lesson: **the big perf
decision is fitting the per-lane register budget.**  Double-buffering the
operands looked like a free latency win but regressed −73% on gfx950
because the extra staging registers spilled the accumulator out of the
VGPR/AGPR file.  The real lever was *slicing the output tile* so the
accumulator fits the budget.

This module models that lever, hardware-free, on top of ``rocm_target``'s
per-arch register budgets (Architecture Decision #19 — a backend-shaped
object lit/unit tests can reason over before any HIP emission):

  - ``TileShape`` / ``TileCandidate`` — the (M, N, K) output tile + the
    knobs (dtype, double-buffering, N-slicing) a candidate carries.
  - ``estimate_vgpr_usage`` — a DOCUMENTED heuristic register model.  It is
    an *estimate*, not exact ISA register accounting: the accumulator lives
    in registers spread over the wave's lanes, plus operand-staging
    registers (doubled when double-buffering).
  - ``fits_budget`` / ``prune_candidates`` — reject candidates that spill,
    returning an auditable ``PruneResult`` so nothing is silently dropped.
  - ``quad_slice`` / ``n_slice`` — the output-tile slicing transforms that
    shrink the accumulator footprint to fit (the real Gluon fix).

Leaf-ish: depends only on ``rocm_target`` + the stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .rocm_target import ROCmTargetProfile, TesseraROCmTargetError

# ── Accumulator word counts per canonical dtype ─────────────────────────────
# Number of 32-bit register words a single accumulator element occupies.  The
# accumulator dtype follows the rocWMMA rule (low-precision float -> fp32,
# int8 -> int32, fp64 -> fp64), so the accumulator is fp32/int32 (1 word) for
# the common matrix paths and fp64 (2 words) for the double-precision path.
_ACC_WORDS_BY_DTYPE: dict[str, int] = {
    "fp64": 2,
    "fp32": 1,
    "bf16": 1,   # accumulates in fp32 -> 1 word
    "fp16": 1,   # accumulates in fp32 -> 1 word
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp4_e2m1": 1,
    "int8": 1,   # accumulates in int32 -> 1 word
    "int32": 1,
}

_STORAGE_BITS_BY_DTYPE: dict[str, int] = {
    "fp64": 64,
    "fp32": 32,
    "bf16": 16,
    "fp16": 16,
    "fp8_e4m3": 8,
    "fp8_e5m2": 8,
    "fp4_e2m1": 4,
    "int8": 8,
    "int32": 32,
}


def _acc_words(dtype: str) -> int:
    """Accumulator register words for an input ``dtype``.

    Raises ``TesseraROCmTargetError`` for an unmodelled dtype rather than
    guessing — the register model must never silently mis-size."""
    try:
        return _ACC_WORDS_BY_DTYPE[dtype]
    except KeyError as e:
        raise TesseraROCmTargetError(
            f"no accumulator-word model for dtype {dtype!r} "
            f"(known: {sorted(_ACC_WORDS_BY_DTYPE)})"
        ) from e


def _storage_bits(dtype: str) -> int:
    try:
        return _STORAGE_BITS_BY_DTYPE[dtype]
    except KeyError as e:
        raise TesseraROCmTargetError(
            f"no storage-bit model for dtype {dtype!r} "
            f"(known: {sorted(_STORAGE_BITS_BY_DTYPE)})"
        ) from e


@dataclass(frozen=True)
class TileShape:
    """An (M, N, K) output/contraction tile shape.  M, N are the output-tile
    extent (the accumulator footprint); K is the contraction depth."""

    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        for name, val in (("m", self.m), ("n", self.n), ("k", self.k)):
            if val <= 0:
                raise ValueError(
                    f"TileShape.{name} must be positive, got {val}")

    @property
    def output_area(self) -> int:
        """Output-tile area (``m * n``) — the accumulator-footprint driver."""
        return self.m * self.n

    def as_metadata_dict(self) -> dict[str, int]:
        return {"m": self.m, "n": self.n, "k": self.k}


@dataclass(frozen=True)
class TileCandidate:
    """A tiling candidate: an output tile + the knobs that affect its register
    footprint.

    Attributes:
        tile          : the (M, N, K) output tile.
        dtype         : input/storage dtype (drives accumulator word count).
        double_buffer : if True, operand staging is double-buffered (the Gluon
                        knob that *looks* free but doubles operand registers).
        n_slice       : number of N sub-slices the output tile is split into
                        (>=1).  ``n_slice=2`` halves the per-pass accumulator
                        footprint — the real Gluon fit lever.
    """

    tile: TileShape
    dtype: str
    double_buffer: bool = False
    n_slice: int = 1

    def __post_init__(self) -> None:
        if self.n_slice < 1:
            raise ValueError(
                f"n_slice must be >= 1, got {self.n_slice}")
        if not self.dtype:
            raise ValueError("dtype must be a non-empty canonical dtype string")

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "tile": self.tile.as_metadata_dict(),
            "dtype": self.dtype,
            "double_buffer": self.double_buffer,
            "n_slice": self.n_slice,
        }


@dataclass(frozen=True)
class PruneResult:
    """The auditable result of pruning a candidate set against a budget.

    ``kept`` are the candidates that fit; ``dropped`` are the ones that would
    spill.  Nothing is silently discarded — every rejected candidate is here so
    "what got dropped, and why" is inspectable."""

    kept: tuple[TileCandidate, ...]
    dropped: tuple[TileCandidate, ...]

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "kept": [c.as_metadata_dict() for c in self.kept],
            "dropped": [c.as_metadata_dict() for c in self.dropped],
        }


@dataclass(frozen=True)
class RankedTileCandidate:
    """A hardware-free ROCm autotune ranking record.

    ``score`` is an ordering heuristic, not measured performance.  Lower is
    better.  The fields expose the reasons so a later hardware sweep can replace
    the weights without changing the Tile-IR contract or hiding rejected risks.
    """

    candidate: TileCandidate
    vgpr_usage: int
    register_margin: int
    lds_bytes: int
    lds_margin: int
    bank_padding_required: bool
    register_macro_tile: tuple[int, int]
    split_k_required: bool
    pipeline_depth: int
    score: float
    reasons: tuple[str, ...]

    @property
    def fits_register_budget(self) -> bool:
        return self.register_margin >= 0

    @property
    def fits_lds_budget(self) -> bool:
        return self.lds_margin >= 0

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.as_metadata_dict(),
            "vgpr_usage": self.vgpr_usage,
            "register_margin": self.register_margin,
            "lds_bytes": self.lds_bytes,
            "lds_margin": self.lds_margin,
            "bank_padding_required": self.bank_padding_required,
            "register_macro_tile": self.register_macro_tile,
            "split_k_required": self.split_k_required,
            "pipeline_depth": self.pipeline_depth,
            "score": self.score,
            "reasons": list(self.reasons),
            "measured": False,
        }


def estimate_vgpr_usage(candidate: TileCandidate, profile: ROCmTargetProfile) -> int:
    """Estimate per-lane register usage for ``candidate`` on ``profile``.

    DOCUMENTED HEURISTIC (a model, not exact ISA accounting):

    1. **Accumulator** — the output tile ``M * N`` lives in registers, spread
       evenly over the wave's lanes (``threads_per_wave``).  Each element costs
       ``acc_words(dtype)`` 32-bit words.  Slicing N into ``n_slice`` parts
       means only ``1/n_slice`` of the columns are live per pass, so the
       accumulator footprint divides by ``n_slice`` (this is the Gluon fit
       lever).  We round UP so a tile that doesn't divide evenly across lanes
       is charged for the partial register, never under-counted::

           acc_regs = ceil( (M * (N / n_slice) * acc_words) / lanes_per_wave )

    2. **Operand staging** — the A/B fragments being loaded for the current K
       step need staging registers.  We model this as one register per K
       element along each of the two operands per lane's share, i.e.
       ``2 * K`` words spread over the wave, rounded up.  Double-buffering
       holds two K-steps in flight at once, so this term DOUBLES — the knob
       that looked free but pushes a borderline tile over budget::

           stage_regs = ceil( (2 * K) / lanes_per_wave ) * (2 if double_buffer else 1)

    The estimate is ``acc_regs + stage_regs``.  It deliberately ignores
    address/loop bookkeeping registers (a small constant) — the model exists to
    rank candidates by the dominant accumulator+staging term, not to predict the
    exact VGPR count the assembler emits.
    """
    lanes = profile.threads_per_wave
    acc_words = _acc_words(candidate.dtype)
    tile = candidate.tile

    # Accumulator footprint, sliced along N.
    live_cols = tile.n / candidate.n_slice
    acc_elems = tile.m * live_cols * acc_words
    acc_regs = _ceil_div_float(acc_elems, lanes)

    # Operand staging (A + B), doubled when double-buffering.
    stage_per_lane = _ceil_div(2 * tile.k, lanes)
    stage_regs = stage_per_lane * (2 if candidate.double_buffer else 1)

    return acc_regs + stage_regs


def estimate_lds_footprint_bytes(
    candidate: TileCandidate,
    profile: ROCmTargetProfile,
    *,
    pipeline_depth: int | None = None,
) -> int:
    """Estimate the LDS bytes needed for one candidate.

    This is intentionally a planning estimate: A and B panels for one
    ``M/N-slice`` by ``K`` step, multiplied by the requested pipeline depth
    (or by the double-buffer minimum of 2).  Sub-byte storage is charged at its
    byte container because ROCm lowering only accepts it after an explicit
    storage-pack consumer has materialized the packing descriptor.
    """
    depth = profile.pipeline_stages if pipeline_depth is None else pipeline_depth
    if depth < 1:
        raise ValueError(f"pipeline_depth must be >= 1, got {depth}")
    if candidate.double_buffer:
        depth = max(depth, 2)
    tile = candidate.tile
    bytes_per_element = max(1, _ceil_div(_storage_bits(candidate.dtype), 8))
    live_n = _ceil_div(tile.n, candidate.n_slice)
    panels = (tile.m * tile.k) + (tile.k * live_n)
    return panels * bytes_per_element * depth


def requires_lds_bank_padding(candidate: TileCandidate) -> bool:
    """Heuristic bank-padding signal for row-strided LDS panels.

    A row stride that lands exactly on a 128-byte bank period is a classic
    conflict shape.  The planner records this as a penalty/metadata bit, but
    does not rewrite the layout; the future LDS bank-padding pass owns that.
    """
    bytes_per_element = max(1, _ceil_div(_storage_bits(candidate.dtype), 8))
    live_n = _ceil_div(candidate.tile.n, candidate.n_slice)
    row_bytes = live_n * bytes_per_element
    return row_bytes % 128 == 0


def _estimated_bank_padding_bytes(candidate: TileCandidate) -> int:
    if not requires_lds_bank_padding(candidate):
        return 0
    bytes_per_element = max(1, _ceil_div(_storage_bits(candidate.dtype), 8))
    # One extra element per row in both panels is enough metadata for the V1
    # cost model; exact padding policy is backend-lowering work.
    return (candidate.tile.m + _ceil_div(candidate.tile.n, candidate.n_slice)) * bytes_per_element


def _register_macro_tile(candidate: TileCandidate) -> tuple[int, int]:
    return (
        max(1, _ceil_div(candidate.tile.m, 16)),
        max(1, _ceil_div(_ceil_div(candidate.tile.n, candidate.n_slice), 16)),
    )


def rank_candidates(
    candidates: list[TileCandidate],
    profile: ROCmTargetProfile,
    *,
    pipeline_depth: int | None = None,
    split_k_threshold: int = 4096,
) -> tuple[RankedTileCandidate, ...]:
    """Rank ROCm tiling candidates without claiming measured performance.

    The ranking feeds autotune/planner plumbing with the dimensions the plan
    cares about: register fit, LDS footprint, bank-padding requirement,
    register macro-tile size, split-K need, and pipeline depth.  It is a
    deterministic pre-silicon ordering; hardware timing must still pick the
    measured winner.
    """
    depth = profile.pipeline_stages if pipeline_depth is None else pipeline_depth
    if depth < 1:
        raise ValueError(f"pipeline_depth must be >= 1, got {depth}")

    ranked: list[RankedTileCandidate] = []
    for cand in candidates:
        vgpr = estimate_vgpr_usage(cand, profile)
        register_margin = profile.total_reg_budget - vgpr
        lds = estimate_lds_footprint_bytes(cand, profile, pipeline_depth=depth)
        padding = _estimated_bank_padding_bytes(cand)
        lds_with_padding = lds + padding
        lds_margin = profile.lds_capacity_bytes - lds_with_padding
        macro_tile = _register_macro_tile(cand)
        split_k_required = cand.tile.k > split_k_threshold or lds_margin < 0
        bank_padding = padding > 0

        reasons: list[str] = []
        score = float(vgpr)
        if register_margin < 0:
            score += 1_000_000 + abs(register_margin) * 100
            reasons.append("register_over_budget")
        else:
            reasons.append("register_fit")
        if lds_margin < 0:
            score += 500_000 + abs(lds_margin) / 16
            reasons.append("lds_over_budget")
        else:
            reasons.append("lds_fit")
        score += lds_with_padding / 256
        if bank_padding:
            score += 64
            reasons.append("bank_padding_required")
        if split_k_required:
            score += 256
            reasons.append("split_k_required")
        score += depth * 8
        score += cand.n_slice * 4
        score -= min(macro_tile[0] * macro_tile[1], 16) * 4

        ranked.append(
            RankedTileCandidate(
                candidate=cand,
                vgpr_usage=vgpr,
                register_margin=register_margin,
                lds_bytes=lds_with_padding,
                lds_margin=lds_margin,
                bank_padding_required=bank_padding,
                register_macro_tile=macro_tile,
                split_k_required=split_k_required,
                pipeline_depth=depth,
                score=score,
                reasons=tuple(reasons),
            )
        )

    return tuple(
        sorted(
            ranked,
            key=lambda r: (
                r.score,
                r.vgpr_usage,
                r.lds_bytes,
                -r.register_macro_tile[0] * r.register_macro_tile[1],
                repr(r.candidate.as_metadata_dict()),
            ),
        )
    )


def _ceil_div(a: int, b: int) -> int:
    """Integer ceiling division (b > 0)."""
    return -(-a // b)


def _ceil_div_float(a: float, b: int) -> int:
    """Ceiling division for a float numerator (the N-slice can be fractional)."""
    return math.ceil(a / b)


def fits_budget(candidate: TileCandidate, profile: ROCmTargetProfile) -> bool:
    """True iff ``candidate``'s estimated register usage fits the arch budget.

    Compares ``estimate_vgpr_usage`` against ``profile.total_reg_budget`` (the
    combined VGPR+AGPR budget — 512 on CDNA, 256 on RDNA/wave32)."""
    return estimate_vgpr_usage(candidate, profile) <= profile.total_reg_budget


def prune_candidates(
    candidates: list[TileCandidate], profile: ROCmTargetProfile
) -> PruneResult:
    """Partition ``candidates`` into those that fit ``profile``'s budget and
    those that would spill.

    Returns a ``PruneResult`` — nothing is silently dropped; every rejected
    candidate is recorded in ``dropped`` for audit (the v6 lesson: a spilling
    tile that gets quietly discarded hides the real perf cliff)."""
    kept: list[TileCandidate] = []
    dropped: list[TileCandidate] = []
    for cand in candidates:
        if fits_budget(cand, profile):
            kept.append(cand)
        else:
            dropped.append(cand)
    return PruneResult(kept=tuple(kept), dropped=tuple(dropped))


def quad_slice(tile: TileShape) -> tuple[TileShape, TileShape, TileShape, TileShape]:
    """Slice an output tile into four quadrant sub-tiles (``m//2 x n//2``, same
    K).  Each quadrant has a quartered ``m * n`` accumulator area — the
    coarsest output-tile slice for fitting a too-large tile into budget.

    Requires even M and N (a quadrant must be a whole tile); raises
    ``ValueError`` otherwise."""
    if tile.m % 2 != 0 or tile.n % 2 != 0:
        raise ValueError(
            f"quad_slice requires even m and n, got m={tile.m}, n={tile.n}")
    hm, hn = tile.m // 2, tile.n // 2
    quad = TileShape(hm, hn, tile.k)
    return (quad, quad, quad, quad)


def n_slice(tile: TileShape, parts: int) -> tuple[TileShape, ...]:
    """Slice an output tile along N into ``parts`` equal sub-tiles (``m x
    n//parts``, same K), halving (or finer) the per-slice N footprint.

    Requires ``parts >= 1`` and ``parts`` to divide N evenly; raises
    ``ValueError`` otherwise."""
    if parts < 1:
        raise ValueError(f"parts must be >= 1, got {parts}")
    if tile.n % parts != 0:
        raise ValueError(
            f"n_slice parts={parts} must divide n={tile.n} evenly")
    sub_n = tile.n // parts
    sub = TileShape(tile.m, sub_n, tile.k)
    return tuple(sub for _ in range(parts))


__all__ = [
    "TileShape",
    "TileCandidate",
    "PruneResult",
    "RankedTileCandidate",
    "estimate_vgpr_usage",
    "estimate_lds_footprint_bytes",
    "fits_budget",
    "rank_candidates",
    "prune_candidates",
    "requires_lds_bank_padding",
    "quad_slice",
    "n_slice",
]
