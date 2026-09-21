"""RDNA/CDNA occupancy model: VGPR file, LDS pool and wave slots -> waves/SIMD.

No consumer yet
---------------
**Nothing in the compiler calls this module.**  It is a model plus its device
measurement, landed ahead of the code that will use it, and it is owned by
sync ``RDNA-OCCUPANCY-GRANULE-2026-09-20`` in
``docs/audit/backend/rocm/todo.md``.  Read it as a declared debt under
Decision #29a, not as a closed contract: the compiler does not yet consult
occupancy when it selects a tile, and this file does not change any decision
the compiler makes today.  Its 53 tests assert behaviour -- including 12
points measured on gfx1201 -- rather than mere presence, which is what keeps
an unwired model from being silently wrong.

Wiring it into tile selection is the follow-on and is deliberately not in the
change that introduced this file.

Why this module exists
----------------------
``rocm_target`` already carries every *ingredient* of an occupancy verdict --
:data:`~tessera.compiler.rocm_target._VGPR_BUDGET` (the per-wave architectural
cap), ``_LDS_BYTES``, ``_MAX_WAVES``, and measured
:func:`~tessera.compiler.rocm_target.dispatch_slots` -- and nothing composed
them.  The composition is the part that decides whether a kernel can hide load
latency, which is the binding constraint measured on gfx1201: the staging copy
is latency-bound with an unhidden-load ceiling, and occupancy is the only lever
that hides it.  A tile chosen without this model is chosen without knowing
whether its latency can be covered.

The three limiters, and the one that bites
------------------------------------------
Occupancy is ``min`` over three independent ceilings -- registers, LDS, and
wave slots -- and the report names which one binds, because shedding registers
on an LDS-bound kernel buys exactly nothing.

The register ceiling is **quantised**, and that is the whole point of this
module.  RDNA4 ISA 3.3.2.1:

    "VGPRs are allocated in blocks of 16 for wave32 or 8 for wave64, and a
    shader may have up to 256 VGPRs. [...] Devices that have 1536 VGPRs per
    SIMD allocate in blocks of 24 for wave32 and 12 for wave64."

So a wave using 121 VGPRs does not occupy 121 registers; it occupies
``ceil(121/granule)*granule``.  Occupancy therefore moves in **rungs**, and a
kernel is routinely one or two registers above a rung it could have reached for
free.  :func:`headroom_to_next_rung` is the actionable output of this file:
it answers "how many VGPRs must this kernel shed to gain a wave", which is a
question a spill count cannot answer.

Two quantities that are NOT the same number
-------------------------------------------
``_VGPR_BUDGET`` in ``rocm_target`` is the **per-wave architectural cap** (256
on RDNA) -- the most one wave may address.  The **VGPR file per SIMD** is the
physical pool those allocations come out of.  Conflating them yields
``256/256 = 1 wave`` on every RDNA kernel, which is wrong by up to 16x; it is
also the exact shape of the recorded trap that CDNA's "512 VGPRs, one wave per
SIMD" recipe does not transfer to RDNA.  They are kept in separate tables here
and the docstrings say which is which.

Open question this module makes precise
---------------------------------------
The granule for gfx1151/gfx1201 is *contested*, and the contest is decidable.
RDNA4 ISA 3.3.2.1 prescribes 24 for a 1536-register file; the ROCm queue's two
recorded figures for one attention kernel (256 VGPRs -> 6 waves/SIMD, 121 ->
12) are jointly explained by a 1536-register file at granule 4, 8, 16 or 32 --
and by **no** candidate at granule 24.  So the documented rule and this repo's
own evidence disagree, and until ``scripts/probe_rdna_vgpr_granule.py`` runs on
Tajasarus, every consumer should treat a rung boundary as accurate to within
one block.  :attr:`WorkgroupOccupancy.granule_contested` carries that on the
result.  The ladder *shape* -- that occupancy is quantised at all, and that
shedding registers between rungs buys nothing -- does not depend on the answer.

Fail-closed
-----------
Every per-arch constant returns ``None`` (or raises :class:`TesseraOccupancyError`)
when this fleet has not established it, following the precedent
:func:`~tessera.compiler.rocm_target.dispatch_slots` set: a caller must decline
to conclude rather than substitute a default.  An occupancy number is acted on
downstream -- it selects tiles -- so a fabricated one is worse than no answer.
Each entry carries its :class:`~tessera.compiler.target_perf.Provenance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from tessera.compiler.rocm_target import (
    _IS_RDNA,
    _LDS_BYTES,
    _VGPR_BUDGET,
    AMDArch,
    WorkgroupProcessorMode,
    max_waves_per_cu,
    simds_per_cu,
)
from tessera.compiler.target_perf import Provenance

__all__ = [
    "TesseraOccupancyError",
    "WAVE32",
    "WAVE64",
    "VgprAllocation",
    "OccupancyRung",
    "WorkgroupOccupancy",
    "vgpr_regs_per_simd",
    "vgpr_alloc_granule",
    "granule_is_contested",
    "wave_slots_provenance",
    "VGPR_GRANULE_CONTESTED",
    "wave_slots_per_simd",
    "simds_per_slot",
    "allocate_vgprs",
    "occupancy_rungs",
    "headroom_to_next_rung",
    "LDS_ALLOC_GRANULE_BYTES",
    "MAX_WORKGROUPS_PER_WGP",
    "MAX_WORKITEMS_PER_WORKGROUP",
    "lds_alloc_granule",
    "per_wave_vgpr_cap",
    "lds_request_cap_bytes",
    "lds_pool_bytes",
    "estimate_occupancy",
]


WAVE32 = 32
WAVE64 = 64

#: LDS allocation block size, **per architecture family**.  A request is
#: quantised exactly as a VGPR request is, and dividing the pool by an
#: unrounded request overstates how many work-groups are resident.
#:
#: This is not one number.  RDNA4 ISA 3.3.5 and RDNA3.5 ISA 3.3.4 both say
#: "allocated in blocks of 1024 bytes"; CDNA5 ISA 3.3.4 says "LDS space is
#: allocated in blocks of **2048** bytes".  A single module constant would be
#: silently wrong by 2x on gfx125x -- the same per-family shape as the VGPR
#: granule, and the same failure mode as the per-CU/per-SIMD wave-slot mixup.
_LDS_ALLOC_GRANULE: dict[AMDArch, int] = {
    AMDArch.GFX_1100: 1024,
    AMDArch.GFX_1151: 1024,
    AMDArch.GFX_1200: 1024,
    AMDArch.GFX_1201: 1024,
    AMDArch.GFX_1250: 2048,
    AMDArch.GFX_1251: 2048,
    # CDNA 1-4 not established from a manual on this fleet; absent so a caller
    # declines rather than inheriting RDNA's block size.
}

#: Back-compat alias for the RDNA value.  Prefer :func:`lds_alloc_granule`.
LDS_ALLOC_GRANULE_BYTES = 1024

#: Work-groups resident on one WGP, ISA 2.3: "The WGP supports up to 32
#: work-groups with a maximum of 1024 work-items per work-group."  Single-wave
#: work-groups are exempt -- they allocate no barrier resource.
#:
#: **It does not bind on any current fleet part, and that is measured, not
#: assumed.**  A WGP holds 4 SIMDs x 16 wave slots = 64 waves, and a non-exempt
#: work-group is at least 2 waves, so the wave-slot ceiling already caps
#: residency at exactly 32 groups.  The two limits are coincident.  It is
#: applied anyway because the coincidence is arithmetic, not a law: a part with
#: more than 16 slots per SIMD would make this the binding ceiling, and a
#: silently-absent limit is harder to notice than a redundant one.
#: ``test_workgroup_ceiling_is_currently_coincident_with_wave_slots`` asserts
#: the redundancy so nobody reads it as load-bearing (Decision #29a).
MAX_WORKGROUPS_PER_WGP = 32

#: ISA 2.3.  A launch above this is rejected rather than clamped.
MAX_WORKITEMS_PER_WORKGROUP = 1024


def lds_alloc_granule(arch: AMDArch) -> Optional[int]:
    """LDS allocation block size for *arch*, or None when unestablished.

    1024 B on RDNA3/3.5/4, 2048 B on CDNA5 (gfx125x).  None rather than a
    default: inheriting RDNA's block size on a CDNA part understates the
    allocation by up to 1 KiB per work-group.
    """
    return _LDS_ALLOC_GRANULE.get(arch)


def per_wave_vgpr_cap(arch: AMDArch) -> Optional[int]:
    """Architectural per-wave VGPR ceiling, read from ``rocm_target``.

    256 on RDNA, **1024 on CDNA5** (its ISA 3.3.2.1 says so explicitly, and
    VGPRs above 255 are reached through VGPR-MSB indexing).  Taken from the
    existing ``_VGPR_BUDGET`` table rather than defaulted, so a CDNA5 wave is
    not refused at 257 registers for being over an RDNA limit.
    """
    return _VGPR_BUDGET.get(arch)


class TesseraOccupancyError(Exception):
    """An occupancy constant is not established for this arch on this fleet.

    Names the missing quantity and the probe that fills it, rather than
    returning a fabricated default.
    """


# ── Physical VGPR file per SIMD ──────────────────────────────────────────────
#
# NOT the per-wave cap (that is ``rocm_target._VGPR_BUDGET`` = 256 on RDNA).
# This is the pool a SIMD hands out, and it is the numerator of the register
# occupancy ceiling.
#
# gfx1151 / gfx1201 = 1536 is DERIVED, not measured here, and the derivation is
# independent of the granule question below: the ROCm queue records a
# split-wave attention kernel moving 256 -> 121 VGPRs and modeled occupancy
# 6 -> 12 waves/SIMD.  ``1536/256 == 6`` exactly, and 256 is a multiple of both
# candidate granules (16 and 24), so that data point pins the file size without
# assuming a granule.  Confirm on device with::
#
#     hipcc --offload-arch=gfx1201 -Rpass-analysis=kernel-resource-usage -c k.hip
#
# whose remark prints VGPRs and Occupancy [waves/SIMD] for the same kernel --
# two numbers whose ratio exposes both constants at once.
_VGPR_REGS_PER_SIMD: dict[AMDArch, tuple[int, Provenance]] = {
    # Measured on Princess-Luna 2026-09-20, unique solution over 12
    # observations: benchmarks/baselines/gfx1151_vgpr_granule_20260920/.
    AMDArch.GFX_1151: (1536, Provenance.MEASURED),
    # Measured on Tajasarus 2026-09-20, unique solution over 12 observations:
    # benchmarks/baselines/gfx1201_vgpr_granule_20260920/.
    AMDArch.GFX_1201: (1536, Provenance.MEASURED),
}


#: Architectures whose per-SIMD wave-slot count this fleet has measured, as
#: opposed to derived from ``rocm_target._MAX_WAVES``.  Both RDNA parts were
#: measured on 2026-09-20: occupancy plateaus at exactly 16 for every kernel
#: far below the register ceiling, on gfx1201 (Tajasarus) and gfx1151
#: (Princess-Luna) independently.
_WAVE_SLOTS_MEASURED: frozenset[AMDArch] = frozenset(
    {AMDArch.GFX_1151, AMDArch.GFX_1201}
)


#: SIMDs behind one dispatch slot, per arch and per mode.
#:
#: **Established per arch, never inferred from "is this RDNA?".**  That
#: inference was wrong: it sent every non-RDNA part down a CU-shaped branch
#: returning 2, while CDNA5 ISA 2.2 says a work-group's waves "can run on any
#: of the 4 SIMD32s" of its WGP -- 2x wrong, and silent.  gfx1251 is a far
#: larger part than gfx1201, which is exactly the case where a fallback shaped
#: by the small part goes unnoticed.
#:
#: An arch absent from a table declines rather than borrowing the other one.
_SIMDS_PER_WGP: dict[AMDArch, int] = {
    AMDArch.GFX_1100: 4,  # RDNA3 ISA 2.3
    AMDArch.GFX_1151: 4,  # RDNA3.5 ISA 2.3
    AMDArch.GFX_1200: 4,
    AMDArch.GFX_1201: 4,  # RDNA4 ISA 2.3
    AMDArch.GFX_1250: 4,  # CDNA5 ISA 2.2 -- GFX12-derived, has WGPs
    AMDArch.GFX_1251: 4,
    # CDNA1-4 have no WGP tier at all; absent, not zero.
}

#: CU-mode SIMD count.  RDNA splits its WGP into two CUs of 2 SIMD32s each
#: (RDNA4 ISA 2.3).  CDNA5's manual describes only the WGP tier, so CU mode is
#: not established there.
_SIMDS_PER_CU_MODE: dict[AMDArch, int] = {
    AMDArch.GFX_1100: 2,
    AMDArch.GFX_1151: 2,
    AMDArch.GFX_1200: 2,
    AMDArch.GFX_1201: 2,
    AMDArch.GFX_90A: 4,  # CDNA CU is 4 SIMDs; a dispatch slot is a CU
    AMDArch.GFX_940: 4,
    AMDArch.GFX_942: 4,
    AMDArch.GFX_950: 4,
}


def vgpr_regs_per_simd(arch: AMDArch) -> Optional[int]:
    """Physical VGPRs per SIMD, or None when this fleet has not established it.

    None means "never established", not "zero" -- a caller must decline to
    conclude.  See the module docstring for why this is not ``_VGPR_BUDGET``.
    """
    entry = _VGPR_REGS_PER_SIMD.get(arch)
    return None if entry is None else entry[0]


def vgpr_regs_per_simd_provenance(arch: AMDArch) -> Provenance:
    """How the file size for *arch* was established."""
    entry = _VGPR_REGS_PER_SIMD.get(arch)
    return Provenance.UNKNOWN if entry is None else entry[1]


#: Architectures whose ISA states a granule outright, with no dependence on
#: the VGPR file size.  CDNA5 ISA 3.3.2.1 says only "VGPRs are allocated in
#: blocks of 16 for wave32" -- it carries neither the "devices that have 1536
#: VGPRs per SIMD allocate in blocks of 24" clause nor a wave64 form, because
#: CDNA5 compute waves are wave32.  Stating it here means the granule is known
#: for gfx125x even though its file size is not, instead of being lost to a
#: derivation that needs a constant nobody has measured.
_STATED_GRANULE_WAVE32: dict[AMDArch, int] = {
    AMDArch.GFX_1250: 16,
    AMDArch.GFX_1251: 16,
}


def vgpr_alloc_granule(arch: AMDArch, *, wave_size: int = WAVE32) -> Optional[int]:
    """VGPR allocation block size, per RDNA4 / RDNA3.5 / CDNA5 ISA 3.3.2.1.

    On RDNA the rule is a function of the file size, not a free parameter:
    devices with 1536 VGPRs per SIMD allocate in blocks of 24 (wave32) / 12
    (wave64); all others use 16 / 8.  Deriving it rather than tabling it means
    the one thing a caller must supply is the file size.

    CDNA5 states its granule outright (16, wave32 only) and has no
    file-size-conditional clause, so it comes from
    :data:`_STATED_GRANULE_WAVE32` and is known even where the file size is
    not.  Occupancy there still declines -- a granule alone cannot give a wave
    count -- but the granule itself is not thrown away.

    This granule is why occupancy has rungs.  At 121 VGPRs it is the difference
    between 10 waves/SIMD (granule 24 -> 144 allocated) and 12 (granule 16 ->
    128 allocated), so it is not a rounding detail -- it moves the answer 20%
    and moves where the next rung sits by 8 registers.
    """
    _check_wave_size(wave_size)
    stated = _STATED_GRANULE_WAVE32.get(arch)
    if stated is not None:
        if wave_size != WAVE32:
            # CDNA5 compute waves are wave32; no wave64 granule is stated.
            return None
        return stated
    regs = vgpr_regs_per_simd(arch)
    if regs is None:
        return None
    if regs == 1536:
        return 24 if wave_size == WAVE32 else 12
    return 16 if wave_size == WAVE32 else 8


#: Architectures whose allocation granule this fleet has not measured, so
#: :func:`vgpr_alloc_granule` returns the ISA-derived value on trust.
#:
#: **Currently empty: both fleet ROCm parts are measured.**  gfx1201
#: (Tajasarus) and gfx1151 (Princess-Luna) each resolved to 1536 VGPRs/SIMD,
#: granule 24, 16 wave slots/SIMD, on their own silicon -- proof does not
#: transfer between them, so each needed its own run.  Both agree with RDNA4
#: ISA 3.3.2.1 and the identically-worded RDNA3.5 3.3.2.1.
#:
#: The set is kept rather than deleted because it is the gate a *new* arch
#: passes through: add one with an ISA-derived granule and no probe run, and
#: it belongs here until someone measures it.
VGPR_GRANULE_CONTESTED: frozenset[AMDArch] = frozenset()


def granule_is_contested(arch: AMDArch) -> bool:
    """Whether :func:`vgpr_alloc_granule` is disputed for *arch*.

    See :data:`VGPR_GRANULE_CONTESTED`.  A contested granule does not make the
    occupancy figure useless -- the *ladder shape* is right either way -- but it
    does mean a rung boundary may be off by one block, so do not promote a
    register target that sits within one granule of a rung on the strength of
    this model alone.
    """
    return arch in VGPR_GRANULE_CONTESTED


def wave_slots_per_simd(arch: AMDArch) -> Optional[int]:
    """Hardware wave slots per SIMD, or None when not established.

    **Derived from ``rocm_target._MAX_WAVES`` rather than tabled here.**  A
    second table would be a second authority for one hardware constant, and
    two copies is exactly how the per-CU/per-SIMD confusion survived: that
    table said 16 "per CU" while this module needed 16 per SIMD, and both were
    the same literal.  Now there is one number (per CU) and one conversion.

    Returns None when either the slot count or the CU geometry is
    unestablished -- an occupancy verdict declines rather than guesses.
    """
    per_cu = max_waves_per_cu(arch)
    simds = simds_per_cu(arch)
    if per_cu is None or simds is None:
        return None
    return per_cu // simds


def wave_slots_provenance(arch: AMDArch) -> Provenance:
    """Whether the per-SIMD slot count for *arch* was measured or derived."""
    if wave_slots_per_simd(arch) is None:
        return Provenance.UNKNOWN
    return Provenance.MEASURED if arch in _WAVE_SLOTS_MEASURED else Provenance.SPEC


def simds_per_slot(arch: AMDArch, mode: WorkgroupProcessorMode) -> Optional[int]:
    """SIMDs behind one :func:`dispatch_slots` unit, or None when unestablished.

    The mode is required for the same reason it is required there: an occupancy
    figure computed against the wrong tier is wrong by the ratio between them.
    Returns None rather than falling back to the other mode's shape -- see
    :data:`_SIMDS_PER_WGP` for the fallback that was wrong.
    """
    table = _SIMDS_PER_CU_MODE if mode is WorkgroupProcessorMode.CU else _SIMDS_PER_WGP
    return table.get(arch)


def _check_wave_size(wave_size: int) -> None:
    if wave_size not in (WAVE32, WAVE64):
        raise TesseraOccupancyError(
            f"wave_size must be {WAVE32} or {WAVE64}, got {wave_size!r}"
        )


def _require(value: Optional[int], what: str, arch: AMDArch, probe: str) -> int:
    if value is None:
        raise TesseraOccupancyError(
            f"{what} is not established for {arch.name} on this fleet, so no "
            f"occupancy verdict can be given.  Establish it with: {probe}"
        )
    return value


def _resolve_cap(arch: AMDArch, per_wave_cap: Optional[int]) -> int:
    """Per-wave VGPR ceiling: the caller's, else the arch's, else refuse.

    Defaulting this to 256 hardcoded RDNA's ceiling into every call site and
    would refuse a legal 300-VGPR CDNA5 wave (its cap is 1024).
    """
    if per_wave_cap is not None:
        if per_wave_cap < 1:
            raise TesseraOccupancyError(
                f"per_wave_cap must be >= 1, got {per_wave_cap!r}"
            )
        return per_wave_cap
    return _require(
        per_wave_vgpr_cap(arch),
        "per-wave VGPR cap",
        arch,
        "add the arch to rocm_target._VGPR_BUDGET",
    )


# ── Register ceiling ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VgprAllocation:
    """What a wave asking for *requested* VGPRs actually occupies."""

    requested: int
    #: Allocation block size (ISA 3.3.2.1).
    granule: int
    #: Whole blocks consumed.
    blocks: int
    #: ``blocks * granule`` -- what the SIMD actually reserves.
    allocated: int
    #: Registers reserved and not asked for.  Free headroom: raising
    #: ``requested`` up to ``allocated`` costs nothing in occupancy.
    wasted: int
    #: Waves this allocation permits, ignoring LDS and slot ceilings.
    waves_per_simd: int

    def as_metadata_dict(self) -> dict[str, int]:
        return {
            "requested_vgprs": self.requested,
            "vgpr_granule": self.granule,
            "vgpr_blocks": self.blocks,
            "allocated_vgprs": self.allocated,
            "wasted_vgprs": self.wasted,
            "vgpr_waves_per_simd": self.waves_per_simd,
        }


def allocate_vgprs(
    vgprs: int,
    *,
    arch: AMDArch,
    wave_size: int = WAVE32,
    per_wave_cap: Optional[int] = None,
    granule: Optional[int] = None,
) -> VgprAllocation:
    """Round *vgprs* to the hardware allocation block and derive the wave count.

    The rounded allocation is clamped to *per_wave_cap*: the granule does not
    push a wave past the architectural maximum it may address.  This is not a
    cosmetic clamp -- with granule 24 an unclamped 256 rounds to 264 and
    reports 5 waves/SIMD where the hardware gives 6, and 6 is the figure the
    ROCm queue actually recorded for a 256-VGPR kernel.

    Pass *granule* to override the ISA-derived block size; see
    :data:`VGPR_GRANULE_CONTESTED` for why an override exists at all.

    Raises when the arch's file size is not established -- see module docstring.
    """
    _check_wave_size(wave_size)
    per_wave_cap = _resolve_cap(arch, per_wave_cap)
    if vgprs < 1:
        # ISA 3.3.2.1: "A wave may not be created with zero VGPRs."
        raise TesseraOccupancyError(
            f"a wave requires at least 1 VGPR, got {vgprs!r} (ISA 3.3.2.1)"
        )
    regs = _require(
        vgpr_regs_per_simd(arch),
        "VGPR file size per SIMD",
        arch,
        "hipcc -Rpass-analysis=kernel-resource-usage on that device",
    )
    if granule is None:
        granule = _require(
            vgpr_alloc_granule(arch, wave_size=wave_size),
            "VGPR allocation granule",
            arch,
            "derive from the file size (ISA 3.3.2.1)",
        )
    elif granule < 1:
        raise TesseraOccupancyError(f"granule must be >= 1, got {granule!r}")
    if vgprs > per_wave_cap:
        raise TesseraOccupancyError(
            f"{vgprs} VGPRs exceeds the {per_wave_cap}-register per-wave cap on "
            f"{arch.name} (ISA 3.3.2.1); the wave cannot address them"
        )
    blocks = -(-vgprs // granule)  # ceil
    allocated = min(blocks * granule, per_wave_cap)

    # A VGPR is one dword per lane, so a wave64 register costs twice the
    # storage of a wave32 one.  ISA 3.3.2.1 makes this explicit by giving the
    # block size in dwords -- "16*32 or 8*64 = 512 DWORDs", and 24*32 = 12*64
    # on a 1536-register part -- i.e. a block is the same physical size either
    # way, while the *register count* that fills it differs by 2x.  Dividing
    # the file's wave32 register count by a wave64 register count overstates
    # wave64 occupancy by exactly 2x, so both sides are normalised to dwords.
    lanes = wave_size
    pool_dwords = regs * WAVE32
    wave_dwords = allocated * lanes
    if wave_dwords > pool_dwords:
        raise TesseraOccupancyError(
            f"{vgprs} wave{wave_size} VGPRs round to {allocated} "
            f"({wave_dwords} dwords), which exceeds the {pool_dwords}-dword "
            f"register file on {arch.name}: no wave can be created"
        )
    return VgprAllocation(
        requested=vgprs,
        granule=granule,
        blocks=blocks,
        allocated=allocated,
        wasted=allocated - vgprs,
        waves_per_simd=pool_dwords // wave_dwords,
    )


@dataclass(frozen=True)
class OccupancyRung:
    """One step of the register-occupancy ladder."""

    waves_per_simd: int
    #: Largest VGPR request that still reaches ``waves_per_simd``.
    max_vgprs: int

    def as_metadata_dict(self) -> dict[str, int]:
        return {"waves_per_simd": self.waves_per_simd, "max_vgprs": self.max_vgprs}


def occupancy_rungs(
    arch: AMDArch,
    *,
    wave_size: int = WAVE32,
    per_wave_cap: Optional[int] = None,
    granule: Optional[int] = None,
) -> tuple[OccupancyRung, ...]:
    """The register-occupancy ladder, highest occupancy first.

    Every reachable ``waves_per_simd`` paired with the largest VGPR request
    that still reaches it.  This is the shape a register-pressure heuristic has
    to respect: *between* rungs, shedding registers buys nothing at all, and
    *at* a rung boundary one register buys a whole wave.  A heuristic that
    treats register pressure as continuous cannot see either fact.

    *per_wave_cap* is the architectural per-wave maximum (256 on RDNA, from
    ``rocm_target._VGPR_BUDGET``).  The top allocation is clamped to it rather
    than rounded past it -- see :func:`allocate_vgprs`.
    """
    _check_wave_size(wave_size)
    per_wave_cap = _resolve_cap(arch, per_wave_cap)
    regs = _require(
        vgpr_regs_per_simd(arch),
        "VGPR file size per SIMD",
        arch,
        "hipcc -Rpass-analysis=kernel-resource-usage on that device",
    )
    if granule is None:
        granule = _require(
            vgpr_alloc_granule(arch, wave_size=wave_size),
            "VGPR allocation granule",
            arch,
            "derive from the file size (ISA 3.3.2.1)",
        )
    if granule < 1:
        raise TesseraOccupancyError(f"granule must be >= 1, got {granule!r}")

    # Reachable allocation sizes, clamped at the per-wave cap.
    n_blocks = -(-per_wave_cap // granule)
    sizes = sorted({min(b * granule, per_wave_cap) for b in range(1, n_blocks + 1)})

    pool_dwords = regs * WAVE32
    best: dict[int, int] = {}
    for size in sizes:
        waves = pool_dwords // (size * wave_size)
        if waves < 1:
            continue
        # Largest request reaching this wave count wins the rung.
        if waves not in best or size > best[waves]:
            best[waves] = size
    return tuple(
        OccupancyRung(waves_per_simd=w, max_vgprs=best[w])
        for w in sorted(best, reverse=True)
    )


def headroom_to_next_rung(
    vgprs: int,
    *,
    arch: AMDArch,
    wave_size: int = WAVE32,
    per_wave_cap: Optional[int] = None,
    granule: Optional[int] = None,
) -> Optional[tuple[int, int]]:
    """``(registers_to_shed, waves_gained)`` for the next occupancy rung.

    ``None`` when already at the top rung.  ``registers_to_shed`` is at least 1.

    This is the question a spill count cannot answer.  "121 VGPRs, 82 spills
    removed" does not say whether the kernel is one register short of a wave --
    and on a latency-bound kernel that one register is the difference between
    covering a global load and stalling on it.  On gfx1201 under the ISA-derived
    granule, 121 VGPRs is exactly one register above the 12-wave rung.
    """
    alloc = allocate_vgprs(
        vgprs,
        arch=arch,
        wave_size=wave_size,
        per_wave_cap=per_wave_cap,
        granule=granule,
    )
    ladder = occupancy_rungs(
        arch, wave_size=wave_size, per_wave_cap=per_wave_cap, granule=granule
    )
    # Rungs are ordered highest-occupancy first; the next one up is the last
    # rung strictly above the current wave count.
    better = [r for r in ladder if r.waves_per_simd > alloc.waves_per_simd]
    if not better:
        return None
    target = min(better, key=lambda r: r.waves_per_simd)
    return (vgprs - target.max_vgprs, target.waves_per_simd - alloc.waves_per_simd)


# ── LDS ceiling ──────────────────────────────────────────────────────────────


def lds_request_cap_bytes(arch: AMDArch) -> Optional[int]:
    """Largest LDS allocation a single work-group may request on *arch*.

    RDNA4 ISA 12.1 caps this at 64 KiB even though the WGP holds 128 KiB; see
    :func:`lds_pool_bytes` for why the two must not be interchanged.  This
    reads the same table as ``ROCmTargetProfile.lds_capacity_bytes`` but is
    arch-keyed and returns None for an unlisted arch instead of raising, so an
    occupancy caller can decline to conclude.
    """
    return _LDS_BYTES.get(arch)


def lds_pool_bytes(arch: AMDArch, mode: WorkgroupProcessorMode) -> Optional[int]:
    """LDS available to one :func:`dispatch_slots` unit, or None if unknown.

    RDNA4 ISA 3.3.5 / RDNA3.5 ISA 3.3.4: "There are 128kB of memory per
    work-group processor [...] One work-group can request up to 64kB", split
    into two 64 KiB halves at byte addresses 0-65535 (CU0) and 65536-131071
    (CU1).  Those are different numbers and the distinction is load-bearing:

    * :func:`lds_request_cap_bytes` is the **per-work-group request cap** --
      what a single group may ask for.
    * the **pool** is what the occupancy denominator divides into: twice the
      cap in WGP mode, and the cap itself in CU mode where the array splits.

    Dividing the request cap instead of the pool halves every WGP-mode LDS
    occupancy figure; dividing the pool in CU mode doubles it.

    **Only established where a manual states the pool separately from the
    cap.**  CDNA5's ISA gives a 320 kB per-work-group maximum and no distinct
    per-WGP pool figure, so this returns None there rather than assuming
    pool == cap -- an assumption that would silently cap gfx1251 residency at
    one work-group for any LDS-heavy kernel.
    """
    if not _IS_RDNA.get(arch, False):
        return None
    cap = lds_request_cap_bytes(arch)
    if cap is None:
        return None
    return cap * 2 if mode is WorkgroupProcessorMode.WGP else cap


# ── Composition ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkgroupOccupancy:
    """Occupancy verdict for one work-group shape, naming the binding limiter."""

    arch: AMDArch
    mode: WorkgroupProcessorMode
    wave_size: int
    waves_per_group: int

    vgpr: VgprAllocation
    lds_bytes_per_group: int
    #: ``lds_bytes_per_group`` rounded up to :data:`LDS_ALLOC_GRANULE_BYTES`.
    lds_allocated_bytes: int

    #: Per-SIMD wave ceilings from each independent limiter.
    waves_by_vgpr: int
    waves_by_lds: int
    waves_by_slots: int

    #: ``min`` of the three -- the actual occupancy.
    waves_per_simd: int
    #: Whole work-groups resident on one dispatch slot.
    groups_per_slot: int
    #: ``waves_per_simd / wave_slots_per_simd``.
    utilization: float
    #: Which ceiling binds: "vgpr", "lds", "slots", or a "+"-joined tie.
    limiter: str
    #: True when the VGPR granule behind ``waves_by_vgpr`` is disputed for this
    #: arch (:data:`VGPR_GRANULE_CONTESTED`).  Carried on the result rather than
    #: left to the caller to remember, so a consumer that would *promote* on
    #: this figure can see that a rung boundary may be off by one block.
    granule_contested: bool

    def as_metadata_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "arch": self.arch.name,
            "mode": self.mode.value,
            "wave_size": self.wave_size,
            "waves_per_group": self.waves_per_group,
            "lds_bytes_per_group": self.lds_bytes_per_group,
            "lds_allocated_bytes": self.lds_allocated_bytes,
            "waves_by_vgpr": self.waves_by_vgpr,
            "waves_by_lds": self.waves_by_lds,
            "waves_by_slots": self.waves_by_slots,
            "waves_per_simd": self.waves_per_simd,
            "groups_per_slot": self.groups_per_slot,
            "utilization": self.utilization,
            "limiter": self.limiter,
            "granule_contested": self.granule_contested,
        }
        d.update(self.vgpr.as_metadata_dict())
        return d


def estimate_occupancy(
    *,
    arch: AMDArch,
    mode: WorkgroupProcessorMode,
    vgprs: int,
    lds_bytes: int,
    workgroup_threads: int,
    wave_size: int = WAVE32,
    per_wave_cap: Optional[int] = None,
    granule: Optional[int] = None,
) -> WorkgroupOccupancy:
    """Compose the register, LDS and wave-slot ceilings into one verdict.

    *mode* is required, never defaulted: it changes the LDS pool and the SIMD
    count behind a dispatch slot, so an occupancy computed against the wrong
    one is wrong by 2x in both terms.  gfx1201's register and LDS matmul bodies
    carry ``WGP_MODE=1``, but the ISA lets any dispatch choose otherwise.

    Raises :class:`TesseraOccupancyError` rather than guessing when a constant
    is unestablished or an input exceeds a hardware limit.
    """
    _check_wave_size(wave_size)
    if workgroup_threads < 1:
        raise TesseraOccupancyError(
            f"workgroup_threads must be >= 1, got {workgroup_threads!r}"
        )
    if workgroup_threads > MAX_WORKITEMS_PER_WORKGROUP:
        raise TesseraOccupancyError(
            f"workgroup_threads={workgroup_threads} exceeds the "
            f"{MAX_WORKITEMS_PER_WORKGROUP} work-item limit per work-group "
            f"(RDNA4 ISA 2.3); this dispatch cannot be created"
        )
    if workgroup_threads % wave_size:
        raise TesseraOccupancyError(
            f"workgroup_threads={workgroup_threads} is not a multiple of the "
            f"wave size {wave_size}; a partial wave still occupies a full slot, "
            f"so round the launch shape rather than the occupancy"
        )
    if lds_bytes < 0:
        raise TesseraOccupancyError(f"lds_bytes must be >= 0, got {lds_bytes!r}")

    request_cap = lds_request_cap_bytes(arch)
    if request_cap is not None and lds_bytes > request_cap:
        raise TesseraOccupancyError(
            f"a work-group requested {lds_bytes} B of LDS, above the "
            f"{request_cap} B per-work-group cap on {arch.name} (RDNA4 ISA 12.1); "
            f"this dispatch cannot be created"
        )

    waves_per_group = workgroup_threads // wave_size
    slots_per_simd = _require(
        wave_slots_per_simd(arch),
        "wave slots per SIMD",
        arch,
        "re-source from the generation block diagram or a device read",
    )
    simds = _require(
        simds_per_slot(arch, mode),
        f"SIMDs per {mode.value} dispatch slot",
        arch,
        "read the SIMD count from that family's ISA 2.2/2.3 and add it to "
        "_SIMDS_PER_WGP / _SIMDS_PER_CU_MODE",
    )

    alloc = allocate_vgprs(
        vgprs,
        arch=arch,
        wave_size=wave_size,
        per_wave_cap=per_wave_cap,
        granule=granule,
    )
    waves_by_vgpr = alloc.waves_per_simd

    pool = _require(
        lds_pool_bytes(arch, mode),
        "LDS pool per dispatch slot",
        arch,
        "add the arch to rocm_target._LDS_BYTES",
    )
    # Allocation is quantised to a per-family block: 1 KiB on RDNA3.5/4,
    # 2 KiB on CDNA5.  A module-wide constant would be 2x wrong on gfx125x.
    lds_granule = _require(
        lds_alloc_granule(arch),
        "LDS allocation block size",
        arch,
        "read it from that family's ISA 3.3.4/3.3.5 and add it to "
        "_LDS_ALLOC_GRANULE",
    )
    lds_allocated = -(-lds_bytes // lds_granule) * lds_granule
    if lds_bytes == 0:
        groups_by_lds = None  # unconstrained
    else:
        groups_by_lds = pool // lds_allocated
        if groups_by_lds < 1:
            raise TesseraOccupancyError(
                f"one work-group needs {lds_bytes} B of LDS (allocated as "
                f"{lds_allocated} B in {lds_granule}-byte blocks) but the "
                f"{mode.value} pool is {pool} B on {arch.name}: not even one "
                f"group is resident"
            )

    waves_by_slots = slots_per_simd

    # Residency is a count of *work-groups*, so derive it from group-level
    # capacity and only then express it per SIMD.  Computing a per-SIMD wave
    # ceiling from LDS first and multiplying back discards runnable waves
    # whenever the wave count does not divide evenly across the SIMDs: three
    # 2-wave groups on four SIMDs is six waves, which floors to one per SIMD
    # and loses a whole group on the way back.
    per_simd_cap = min(waves_by_vgpr, waves_by_slots)
    groups_per_slot = (per_simd_cap * simds) // waves_per_group
    if groups_by_lds is not None:
        groups_per_slot = min(groups_per_slot, groups_by_lds)
    # ISA 2.3 work-group ceiling; single-wave groups are exempt.
    if waves_per_group > 1 and mode is WorkgroupProcessorMode.WGP:
        groups_per_slot = min(groups_per_slot, MAX_WORKGROUPS_PER_WGP)
    if groups_per_slot < 1:
        raise TesseraOccupancyError(
            f"no work-group of {workgroup_threads} threads is resident on "
            f"{arch.name} in {mode.value} mode at {vgprs} VGPRs"
        )

    # An LDS ceiling is expressed in groups; convert to the per-SIMD unit for
    # comparison, rounding *up* because waves need not distribute evenly --
    # three 2-wave groups on four SIMDs is 2,2,1,1, and the busiest SIMD is
    # what a register file has to hold.
    waves_by_lds = (
        waves_by_slots
        if groups_by_lds is None
        else min(waves_by_slots, -(-(groups_by_lds * waves_per_group) // simds))
    )

    # ``waves_per_simd`` is the per-SIMD *ceiling* -- the quantity the
    # compiler's `Occupancy [waves/SIMD]` remark reports and the device
    # measurements pin.  It is deliberately NOT achieved residency for one
    # launch shape: a 8-wave group against a 9-wave ceiling fits four groups
    # and leaves the ninth slot idle, which says something about that grid,
    # not about the kernel's occupancy.  Residency is `groups_per_slot`.
    waves_per_simd = min(waves_by_vgpr, waves_by_lds, waves_by_slots)
    binding = [
        name
        for name, value in (
            ("vgpr", waves_by_vgpr),
            ("lds", waves_by_lds),
            ("slots", waves_by_slots),
        )
        if value == waves_per_simd
    ] or ["lds"]

    return WorkgroupOccupancy(
        arch=arch,
        mode=mode,
        wave_size=wave_size,
        waves_per_group=waves_per_group,
        vgpr=alloc,
        lds_bytes_per_group=lds_bytes,
        lds_allocated_bytes=lds_allocated,
        waves_by_vgpr=waves_by_vgpr,
        waves_by_lds=waves_by_lds,
        waves_by_slots=waves_by_slots,
        waves_per_simd=waves_per_simd,
        groups_per_slot=groups_per_slot,
        utilization=waves_per_simd / slots_per_simd,
        limiter="+".join(binding),
        granule_contested=granule is None and granule_is_contested(arch),
    )
