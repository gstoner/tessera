"""tessera.compiler.mma_selector — shared, target-agnostic MMA selector (A4).

Workstream A4 (`COMPILER_REFACTOR_PLAN`): the one place a *lead* abstraction lifts
**upward**. ROCm already models matrix-core selection as a cost-aware choice — a
per-arch shape table ranked by the ``M×N // lanes`` accumulator footprint
(:func:`rocm_target.mfma_accumulator_regs` + ``rank_mfma_shapes_by_footprint``).
NVIDIA, Apple, and x86 had **no** cost-aware MMA selector. This module promotes
ROCm's model into one ``lane_count``-parameterized selector so every cooperative-
matrix target gets the same footprint-ranked selection, keyed by
``(target, arch, dtype)`` — the ``shape_table`` + ``cost_model`` the D1 arbiter
(`emit/candidate.py`) parks here (Decision #28).

**Lead-safety (Theory rule #1).** ROCm stays the reference: its ISA record is built
*from* the existing `rocm_target`/`rocm_mma` tables (never a copy), and
:func:`rank_shapes_by_footprint` reduces to ROCm's own ranking on a ROCm ISA —
`test_mma_selector.py` gates that equivalence, so this can't silently perturb the
lead. No emit path changes; this is a hardware-free selector object (Decision #19)
that the arbiter / backend manifest / lit tests reason over before any lowering.

**The footprint model.** The accumulator is the M×N output tile spread one fp32 per
lane across a cooperative group (wave / warp / simdgroup), so the per-lane cost is
``M * N // lane_count`` — independent of K. That is what makes MMA selection
comparable *across* targets: a warp (32) mma.sync ``m16n8`` costs ``16*8//32 = 4``
accumulator regs/lane, an Apple simdgroup (32) ``8×8`` costs ``2``, a wave64 CDNA
``16×16`` costs ``4``. x86 AMX is tile-register, **not** lane-cooperative, so it
carries ``cooperative=False`` / ``lane_count=None`` and the per-lane footprint is
``None`` (honest — the model does not apply).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .rocm_mma import MmaOperand, _accum_dtype, _k_packing_hint
from .rocm_target import (
    AMDArch,
    mfma_accumulator_regs,
    mfma_variants,
    wmma_variants,
)

# ── the arch-neutral MMA ISA record ─────────────────────────────────────────


class MmaSelectorError(ValueError):
    """A dtype has no MMA path on the ISA, or a shape/preference is illegal —
    raised, never a silent fallback (Decision #21)."""


@dataclass(frozen=True)
class MmaIsa:
    """A per-arch cooperative-matrix ISA record — the parameter table that makes
    MMA selection target-agnostic.

    ``lane_count`` is the cooperative width the accumulator spreads across (wave =
    32/64, warp = 32, simdgroup = 32); ``None`` for a non-lane-cooperative unit
    (x86 AMX tile registers). ``shapes`` is every legal ``(M, N, K)`` tile;
    ``k_by_dtype`` maps an input storage dtype to the contraction width it lowers
    to, so a dtype selects its shape family from ``shapes``.
    """

    target: str                    # "rocm" | "nvidia" | "apple" | "x86"
    arch: str                      # "gfx1151" | "sm_120" | "apple7" | "amx"
    mma_class: str                 # "wmma"|"mfma"|"mma_sync"|"wgmma"|"simdgroup"|"amx"
    cooperative: bool              # True for wave/warp/simdgroup lane-cooperative MMAs
    lane_count: Optional[int]      # cooperative width; None if not lane-cooperative
    shapes: tuple[tuple[int, int, int], ...]
    k_by_dtype: Mapping[str, int]
    acc_by_dtype: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.cooperative and (self.lane_count is None or self.lane_count < 1):
            raise MmaSelectorError(
                f"{self.target}:{self.arch} is cooperative but lane_count="
                f"{self.lane_count!r} (must be a positive int)")
        if not self.cooperative and self.lane_count is not None:
            raise MmaSelectorError(
                f"{self.target}:{self.arch} is not lane-cooperative; lane_count "
                f"must be None, got {self.lane_count!r}")

    @property
    def dtypes(self) -> frozenset[str]:
        return frozenset(self.k_by_dtype)


@dataclass(frozen=True)
class MmaSelection:
    """A target-neutral chosen-MMA descriptor: the shape anchor + derived operands
    + the per-lane accumulator footprint (the cost the arbiter ranks on)."""

    target: str
    arch: str
    mma_class: str
    shape: tuple[int, int, int]
    in_dtype: str
    acc_dtype: str
    lane_count: Optional[int]
    accumulator_regs: Optional[int]   # per-lane footprint; None if not cooperative
    operand_a: MmaOperand
    operand_b: MmaOperand
    accumulator: MmaOperand

    @property
    def m(self) -> int:
        return self.shape[0]

    @property
    def n(self) -> int:
        return self.shape[1]

    @property
    def k(self) -> int:
        return self.shape[2]

    def as_metadata_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "arch": self.arch,
            "mma_class": self.mma_class,
            "shape": list(self.shape),
            "in_dtype": self.in_dtype,
            "acc_dtype": self.acc_dtype,
            "lane_count": self.lane_count,
            "accumulator_regs": self.accumulator_regs,
            "operands": [op.as_metadata_dict()
                         for op in (self.operand_a, self.operand_b,
                                    self.accumulator)],
        }


# ── the promoted footprint / cost model ─────────────────────────────────────


def accumulator_regs_per_lane(shape: tuple[int, ...],
                              lane_count: Optional[int]) -> Optional[int]:
    """Per-lane accumulator registers for ``shape`` (``M * N // lane_count``) — the
    promoted ROCm footprint, generalized by ``lane_count``. ``None`` when the ISA
    is not lane-cooperative (x86 AMX). Delegates to
    :func:`rocm_target.mfma_accumulator_regs` so the arithmetic + validation stay
    single-sourced with the lead."""
    if lane_count is None:
        return None
    return mfma_accumulator_regs(shape, lanes=lane_count)


def rank_shapes_by_footprint(isa: MmaIsa, *, k: Optional[int] = None
                             ) -> list[tuple[tuple[int, int, int], Optional[int]]]:
    """Legal shapes of ``isa`` ranked by accumulator footprint (cheapest first),
    tie-broken by *descending* arithmetic density ``M*N*K`` (prefer the larger
    contraction at equal register cost) — the exact ROCm ordering, generalized. On
    a non-cooperative ISA the footprint is ``None`` and the order falls back to
    density-descending only. ``k`` filters to that contraction width."""
    shapes = [s for s in isa.shapes if k is None or s[2] == k]
    ranked = [(s, accumulator_regs_per_lane(s, isa.lane_count)) for s in shapes]
    ranked.sort(key=lambda sr: (sr[1] if sr[1] is not None else 0,
                                -(sr[0][0] * sr[0][1] * sr[0][2])))
    return ranked


def cheapest_shape(isa: MmaIsa, dtype: str) -> tuple[int, int, int]:
    """The lowest-footprint legal shape for ``dtype`` on ``isa`` (its K family).
    Raises if the dtype has no MMA path or its K family is absent."""
    if dtype not in isa.k_by_dtype:
        raise MmaSelectorError(
            f"dtype {dtype!r} has no MMA path on {isa.target}:{isa.arch} "
            f"(supported: {sorted(isa.dtypes)})")
    k = isa.k_by_dtype[dtype]
    ranked = rank_shapes_by_footprint(isa, k=k)
    if not ranked:
        raise MmaSelectorError(
            f"{isa.target}:{isa.arch} has no shape at K={k} for {dtype!r} "
            f"(shapes: {sorted(isa.shapes)})")
    return ranked[0][0]


def select_mma(isa: MmaIsa, dtype: str, *,
               prefer_shape: Optional[tuple[int, int, int]] = None,
               out_dtype: Optional[str] = None) -> MmaSelection:
    """Select the cooperative-matrix instruction for ``dtype`` on ``isa`` and derive
    its operands + footprint. Picks the cheapest legal shape unless ``prefer_shape``
    is given (which must be legal). Raises ``MmaSelectorError`` — never a silent
    fallback — when the dtype/shape has no path."""
    if prefer_shape is not None:
        if prefer_shape not in isa.shapes:
            raise MmaSelectorError(
                f"prefer_shape {prefer_shape} is not legal on {isa.target}:"
                f"{isa.arch} (legal: {sorted(isa.shapes)})")
        shape = prefer_shape
    else:
        shape = cheapest_shape(isa, dtype)

    acc = out_dtype or (isa.acc_by_dtype or {}).get(dtype) or _accum_dtype(dtype)
    kw = _k_packing_hint(dtype)
    # Default nt formulation: A row-major (K-major), B col-major, C row-major.
    op_a = MmaOperand("matrix_a", dtype, "row_major", kw)
    op_b = MmaOperand("matrix_b", dtype, "col_major", kw)
    op_c = MmaOperand("accumulator", acc, "row_major", 1)
    return MmaSelection(
        target=isa.target, arch=isa.arch, mma_class=isa.mma_class,
        shape=shape, in_dtype=dtype, acc_dtype=acc,
        lane_count=isa.lane_count,
        accumulator_regs=accumulator_regs_per_lane(shape, isa.lane_count),
        operand_a=op_a, operand_b=op_b, accumulator=op_c)


# ── per-arch ISA records ─────────────────────────────────────────────────────
#
# ROCm records are built FROM the existing rocm_target tables (never copied), so
# they can't drift from the lead. NVIDIA/Apple/x86 records are grounded in the
# PTX-ISA fragment shapes / simdgroup_matrix / AMX tile facts.


def rocm_isa(arch: AMDArch) -> MmaIsa:
    """The ROCm ISA record for ``arch``, derived from `rocm_target`'s variant tables
    (the single source of truth): WMMA (wave32) on RDNA, MFMA (wave64) on CDNA. K
    families mirror `rocm_mma`'s dtype→K mapping."""
    is_wmma = bool(wmma_variants(arch))
    if is_wmma:
        shapes = tuple(sorted(wmma_variants(arch)))
        lane_count, mma_class = 32, "wmma"
    else:
        shapes = tuple(sorted({v[:3] for v in mfma_variants(arch)}))
        lane_count, mma_class = 64, "mfma"
    if not shapes:
        raise MmaSelectorError(f"{arch.name} exposes no matrix-core shapes")
    # dtype→K is derived from the reference selector itself (never a copy), so
    # every feature gate — fp8-on-gfx1151, fp4, xf32-on-RDNA — is inherited by
    # construction. A dtype the reference rejects is simply absent here.
    from .rocm_mma import _MMA_INPUT_DTYPES
    from .rocm_mma import select_mma as _rocm_select
    k_by_dtype: dict[str, int] = {}
    for d in _MMA_INPUT_DTYPES:
        try:
            k_by_dtype[d] = _rocm_select(arch, d).k
        except Exception:
            continue
    return MmaIsa(target="rocm", arch=arch.name.lower().replace("_", ""),
                  mma_class=mma_class, cooperative=True, lane_count=lane_count,
                  shapes=shapes, k_by_dtype=k_by_dtype)


#: NVIDIA tensor-core `mma.sync` fragment shapes (PTX ISA): warp = 32 lanes. Per
#: dtype family — tf32 m16n8k8, bf16/f16 m16n8k16, fp8/int8 m16n8k32, fp4 m16n8k64.
_NVIDIA_MMA_SYNC = MmaIsa(
    target="nvidia", arch="sm_120", mma_class="mma_sync",
    cooperative=True, lane_count=32,
    shapes=((16, 8, 8), (16, 8, 16), (16, 8, 32), (16, 8, 64)),
    k_by_dtype={"fp32": 8, "bf16": 16, "fp16": 16,
                "fp8_e4m3": 32, "fp8_e5m2": 32, "int8": 32, "fp4_e2m1": 64},
)

#: Apple `simdgroup_matrix<T,8,8>` (Apple7+): simdgroup = 32 threads, 8×8×8.
_APPLE_SIMDGROUP = MmaIsa(
    target="apple", arch="apple7", mma_class="simdgroup",
    cooperative=True, lane_count=32,
    shapes=((8, 8, 8),),
    k_by_dtype={"fp16": 8, "bf16": 8, "fp32": 8},
)

#: x86 AMX: tile-register, NOT lane-cooperative — the per-lane footprint model does
#: not apply (cooperative=False, lane_count=None). bf16 tile is 16×16×32.
_X86_AMX = MmaIsa(
    target="x86", arch="amx", mma_class="amx",
    cooperative=False, lane_count=None,
    shapes=((16, 16, 32),),
    k_by_dtype={"bf16": 32, "int8": 64},
)

_STATIC_ISAS: dict[tuple[str, str], MmaIsa] = {
    ("nvidia", "sm_120"): _NVIDIA_MMA_SYNC,
    ("apple", "apple7"): _APPLE_SIMDGROUP,
    ("x86", "amx"): _X86_AMX,
}


def get_isa(target: str, arch: str) -> MmaIsa:
    """The MMA ISA record for ``(target, arch)``. ROCm resolves through
    :func:`rocm_isa` (from the live `rocm_target` tables); other targets from the
    static grounded records. Raises ``MmaSelectorError`` for an unknown pair."""
    if target == "rocm":
        key = arch.upper().replace("GFX", "GFX_") if arch.upper().startswith("GFX") \
            else arch.upper()
        try:
            return rocm_isa(AMDArch[key])
        except KeyError as e:
            raise MmaSelectorError(f"unknown ROCm arch {arch!r}") from e
    isa = _STATIC_ISAS.get((target, arch))
    if isa is None:
        raise MmaSelectorError(
            f"no MMA ISA record for ({target!r}, {arch!r}); known: "
            f"{sorted(set(_STATIC_ISAS) | {('rocm', '<gfxNNNN>')})}")
    return isa


__all__ = [
    "MmaIsa",
    "MmaSelection",
    "MmaSelectorError",
    "accumulator_regs_per_lane",
    "rank_shapes_by_footprint",
    "cheapest_shape",
    "select_mma",
    "rocm_isa",
    "get_isa",
]
