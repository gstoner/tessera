"""Workstream A4 — the shared, target-agnostic MMA selector.

Three things this must hold:

1. **Lead-safety (ROCm is the reference).** The shared footprint ranking on a
   ROCm ISA is *byte-identical* to ROCm's own ``rank_mfma_shapes_by_footprint``,
   and the ROCm ISA's dtype set is exactly what ``rocm_mma.select_mma`` accepts —
   including the feature gates (no FP8 WMMA on gfx1151, no fp32 WMMA on RDNA). So
   promoting the model can't silently perturb the lead.
2. **The lift.** NVIDIA / Apple gain the cost-aware selection they lacked — a
   grounded per-arch shape table ranked by the ``M×N // lane_count`` footprint.
3. **Honesty.** x86 AMX is tile-register, not lane-cooperative → no per-lane
   footprint (``None``); an unknown dtype/arch raises, never a silent fallback.
"""
from __future__ import annotations

import pytest

from tessera.compiler import mma_selector as M
from tessera.compiler.rocm_target import (
    AMDArch,
    rank_mfma_shapes_by_footprint,
)
from tessera.compiler.rocm_mma import select_mma as rocm_select_mma


# ── 1. Lead-safety: ROCm equivalence ─────────────────────────────────────────

@pytest.mark.parametrize("arch", [AMDArch.GFX_942, AMDArch.GFX_950])
def test_shared_ranking_matches_rocm_reference(arch):
    # The shared footprint ranking on a ROCm (CDNA/MFMA) ISA reproduces ROCm's own
    # rank_mfma_shapes_by_footprint exactly (same order, same per-lane reg cost).
    isa = M.get_isa("rocm", arch.name.lower().replace("_", ""))
    shared = M.rank_shapes_by_footprint(isa)
    ref = rank_mfma_shapes_by_footprint(arch)
    # reference carries (M,N,K,K_blocks); collapse to unique (M,N,K) in order.
    ref3, seen = [], set()
    for s, regs in ref:
        if s[:3] not in seen:
            ref3.append((s[:3], regs))
            seen.add(s[:3])
    assert shared == ref3


def test_rocm_isa_dtypes_match_reference_selector():
    # The ROCm ISA admits exactly the dtypes rocm_mma.select_mma accepts — the
    # feature gates are inherited by construction, not re-encoded.
    for arch in (AMDArch.GFX_1151, AMDArch.GFX_1200, AMDArch.GFX_942):
        isa = M.get_isa("rocm", arch.name.lower().replace("_", ""))
        for d in ("fp16", "bf16", "int8", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1",
                  "fp32"):
            try:
                rocm_select_mma(arch, d)
                accepted = True
            except Exception:
                accepted = False
            assert (d in isa.dtypes) == accepted, (arch.name, d)


def test_gfx1151_has_no_fp8_wmma():
    # The load-bearing RDNA3.5 distinction: gfx1151 has bf16/fp16/int8 WMMA, NO fp8.
    isa = M.get_isa("rocm", "gfx1151")
    assert isa.mma_class == "wmma" and isa.lane_count == 32
    assert isa.dtypes == frozenset({"fp16", "bf16", "int8"})
    with pytest.raises(M.MmaSelectorError):
        M.select_mma(isa, "fp8_e4m3")


def test_gfx1151_bf16_footprint():
    sel = M.select_mma(M.get_isa("rocm", "gfx1151"), "bf16")
    assert sel.shape == (16, 16, 16)
    assert sel.accumulator_regs == 16 * 16 // 32            # 8 regs/lane


# ── 2. The lift: NVIDIA + Apple cost-aware selection ──────────────────────────

@pytest.mark.parametrize("dtype,shape", [
    ("fp32", (16, 8, 8)),          # tf32 fragment
    ("bf16", (16, 8, 16)),
    ("fp16", (16, 8, 16)),
    ("fp8_e4m3", (16, 8, 32)),
    ("int8", (16, 8, 32)),
    ("fp4_e2m1", (16, 8, 64)),
])
def test_nvidia_mma_sync_shapes_and_footprint(dtype, shape):
    sel = M.select_mma(M.get_isa("nvidia", "sm_120"), dtype)
    assert sel.mma_class == "mma_sync" and sel.shape == shape
    # every mma.sync m16n8 tile is 4 fp32 accumulator regs/thread over a 32-warp.
    assert sel.accumulator_regs == 16 * 8 // 32 == 4


def test_apple_simdgroup_matrix():
    sel = M.select_mma(M.get_isa("apple", "apple7"), "fp16")
    assert sel.mma_class == "simdgroup" and sel.shape == (8, 8, 8)
    assert sel.accumulator_regs == 8 * 8 // 32              # 2 regs/thread


def test_operands_are_derived_nt():
    sel = M.select_mma(M.get_isa("nvidia", "sm_120"), "bf16")
    assert sel.operand_a.layout == "row_major"      # A K-major
    assert sel.operand_b.layout == "col_major"      # B col-major (nt)
    assert sel.accumulator.dtype == "fp32"
    assert sel.operand_a.k_width == 2               # 32 // 16 for bf16


def test_metadata_dict_is_json_shaped():
    md = M.select_mma(M.get_isa("nvidia", "sm_120"), "bf16").as_metadata_dict()
    assert md["shape"] == [16, 8, 16] and md["accumulator_regs"] == 4
    assert len(md["operands"]) == 3


# ── 3. Honesty: x86 AMX, preferences, errors ─────────────────────────────────

def test_x86_amx_is_not_lane_cooperative():
    isa = M.get_isa("x86", "amx")
    assert isa.cooperative is False and isa.lane_count is None
    sel = M.select_mma(isa, "bf16")
    assert sel.shape == (16, 16, 32)
    assert sel.accumulator_regs is None            # per-lane model does not apply


def test_prefer_shape_must_be_legal():
    isa = M.get_isa("nvidia", "sm_120")
    assert M.select_mma(isa, "bf16", prefer_shape=(16, 8, 16)).shape == (16, 8, 16)
    with pytest.raises(M.MmaSelectorError):
        M.select_mma(isa, "bf16", prefer_shape=(64, 64, 64))


def test_unknown_dtype_and_arch_raise():
    with pytest.raises(M.MmaSelectorError):
        M.select_mma(M.get_isa("apple", "apple7"), "int8")   # no int8 simdgroup
    with pytest.raises(M.MmaSelectorError):
        M.get_isa("nvidia", "sm_999")
    with pytest.raises(M.MmaSelectorError):
        M.get_isa("rocm", "gfx404")


def test_cooperative_isa_requires_lane_count():
    with pytest.raises(M.MmaSelectorError):
        M.MmaIsa(target="t", arch="a", mma_class="wmma", cooperative=True,
                 lane_count=None, shapes=((16, 16, 16),), k_by_dtype={"bf16": 16})
    with pytest.raises(M.MmaSelectorError):
        M.MmaIsa(target="t", arch="a", mma_class="amx", cooperative=False,
                 lane_count=32, shapes=((16, 16, 32),), k_by_dtype={"bf16": 32})


def test_out_dtype_override():
    sel = M.select_mma(M.get_isa("nvidia", "sm_120"), "int8")
    assert sel.acc_dtype == "int32"                # int8 → i32 accumulate
    sel2 = M.select_mma(M.get_isa("nvidia", "sm_120"), "bf16", out_dtype="fp16")
    assert sel2.acc_dtype == "fp16"
