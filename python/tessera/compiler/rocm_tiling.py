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
    "estimate_vgpr_usage",
    "fits_budget",
    "prune_candidates",
    "quad_slice",
    "n_slice",
]
