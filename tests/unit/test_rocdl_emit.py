"""AMD rung-2.5 + rung-3 — llvm.amdgcn.wmma emit (RDNA 3/3.5, gfx11).

Completes the host-free Stage-A emit set (NVIDIA ptx_emit, Apple msl_gemm_emit,
AMD rocdl_emit). UNLIKE the other two, the rung-3 toolchain (LLVM `llc` with the
AMDGPU backend) ships with Homebrew LLVM here, so the rung-3 tests *actually run*
on the dev Mac and assert a real `v_wmma_*` instruction — only skip-cleaning if
no `llc` is found.
"""

from __future__ import annotations

import pytest

from tessera.compiler.rocdl_emit import (
    emit_wmma_gemm_layout_llvmir,
    emit_wmma_gemm_llvmir,
    emit_wmma_gemm_store_llvmir,
    emit_dependent_wmma_chain_llvmir,
    emit_wmma_gemm_threadgroup_llvmir,
    emit_wmma_gfx1250_llvmir,
    emit_wmma_llvmir,
    emit_wmma_rdna4_llvmir,
    llc_assemble,
    llc_object,
    validate_wmma_gemm_layout_structure,
    validate_wmma_gemm_store_structure,
    validate_wmma_gemm_structure,
    validate_wmma_gemm_threadgroup_structure,
    validate_wmma_gfx1250_structure,
    validate_wmma_llvmir_structure,
    validate_wmma_rdna4_structure,
    wmma_intrinsic,
    wmma_intrinsic_gfx1250,
    wmma_intrinsic_rdna4,
    wmma_scheduling,
)

def _llc_available() -> bool:
    from tessera.compiler.rocdl_emit import _find_llc
    return _find_llc() is not None


# ── rung 2.5: host-free emit + structural validation ──

@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_emit_and_validate(dtype):
    ir = emit_wmma_llvmir(dtype, arch="gfx1151")
    v = validate_wmma_llvmir_structure(ir, dtype=dtype, arch="gfx1151")
    assert v.ok, v.reasons
    assert f"@{wmma_intrinsic(dtype)}(" in ir
    assert "amdgpu_kernel" in ir


def test_intrinsic_names_match_isa_table33():
    assert wmma_intrinsic("f16") == "llvm.amdgcn.wmma.f32.16x16x16.f16"
    assert wmma_intrinsic("bf16") == "llvm.amdgcn.wmma.f32.16x16x16.bf16"


def test_bf16_uses_i16_operand_type():
    # Verified-by-llc fact: RDNA bf16 wmma takes <16 x i16>, not <16 x bfloat>.
    ir = emit_wmma_llvmir("bf16")
    assert "<16 x i16>" in ir
    assert "<16 x bfloat>" not in ir


def test_f16_uses_native_half():
    ir = emit_wmma_llvmir("f16")
    assert "<16 x half>" in ir


def test_accumulator_is_8xfloat():
    ir = emit_wmma_llvmir("f16")
    assert "<8 x float>" in ir   # wave32 16x16x16 → 8 fp32 acc regs per lane


def test_rdna35_rejects_fp8():
    # RDNA 3.5 has NO FP8 WMMA (that is gfx1200/RDNA 4).
    with pytest.raises(ValueError):
        emit_wmma_llvmir("fp8_e4m3")


def test_rejects_rdna4_arch():
    # gfx1200 uses the gfx12 v2 wmma ABI — out of scope for this gfx11 emitter.
    with pytest.raises(ValueError):
        emit_wmma_llvmir("f16", arch="gfx1200")


def test_validator_catches_dropped_intrinsic():
    ir = emit_wmma_llvmir("f16").replace("llvm.amdgcn.wmma", "bogus.wmma")
    v = validate_wmma_llvmir_structure(ir, dtype="f16")
    assert not v.ok


# ── rung 3: real AMDGCN lowering — RUNS on this host (skip only if no llc) ──

@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
@pytest.mark.parametrize("dtype,arch,want", [
    ("f16", "gfx1151", "v_wmma_f32_16x16x16_f16"),
    ("bf16", "gfx1151", "v_wmma_f32_16x16x16_bf16"),
    ("f16", "gfx1100", "v_wmma_f32_16x16x16_f16"),
])
def test_rung3_lowers_to_real_wmma_instruction(dtype, arch, want):
    """llc lowers the emitted IR to AMDGCN containing the documented v_wmma_*."""
    ir = emit_wmma_llvmir(dtype, arch=arch)
    r = llc_assemble(ir, arch=arch)
    assert r.status == "ok", r.detail
    assert want in r.wmma_instruction, r.wmma_instruction


# ── K-reduction GEMM emit (grows the lane toward a real tiled GEMM) ──

@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_gemm_emit_and_validate(dtype):
    ir = emit_wmma_gemm_llvmir(dtype, arch="gfx1151")
    v = validate_wmma_gemm_structure(ir, dtype=dtype, arch="gfx1151")
    assert v.ok, v.reasons


def test_gemm_has_kloop_accumulator_and_global_io():
    ir = emit_wmma_gemm_llvmir("f16")
    assert "phi <8 x float>" in ir              # K-loop accumulator
    assert ir.count("load <16 x half>") == 2    # A and B global fragment loads
    assert "llvm.amdgcn.workitem.id.x" in ir    # lane-based addressing
    assert "store <8 x float>" in ir            # D store


def test_gemm_validator_catches_dropped_kloop():
    ir = emit_wmma_gemm_llvmir("f16").replace("phi <8 x float>", "freeze <8 x float>")
    v = validate_wmma_gemm_structure(ir, dtype="f16")
    assert not v.ok
    assert any("accumulator phi" in r for r in v.reasons)


# Both RDNA3-class arches: gfx1151 (native Strix Halo) and gfx1100 (how the
# Radeon 8060S enumerates under WSL/ROCm 7.2.4 — the actual box target).
@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
@pytest.mark.parametrize("arch", ["gfx1151", "gfx1100"])
@pytest.mark.parametrize("dtype,want", [
    ("f16", "v_wmma_f32_16x16x16_f16"),
    ("bf16", "v_wmma_f32_16x16x16_bf16"),
])
def test_rung3_gemm_lowers_to_wmma_inside_a_loop(dtype, want, arch):
    """The K-reduction GEMM lowers to AMDGCN with the v_wmma_* inside a real loop."""
    ir = emit_wmma_gemm_llvmir(dtype, arch=arch)
    r = llc_assemble(ir, arch=arch)
    assert r.status == "ok", r.detail
    assert want in r.wmma_instruction
    # the wmma is inside a loop body (a conditional branch / loop label exists).
    assert "s_cbranch" in r.asm or ".LBB" in r.asm


# ── rung 3 (object form): the plan's "compiles A to a real object" gate ──

@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
@pytest.mark.parametrize("arch", ["gfx1100", "gfx1151"])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_rung3_gemm_assembles_to_amdgpu_elf_object(dtype, arch):
    """The WMMA GEMM lowers all the way to a real relocatable **object** that is
    an AMD GPU ELF (EM_AMDGPU) — Stage B's "compiles A to a real object" gate,
    machine-confirmed for the box target gfx1100 (and native gfx1151)."""
    ir = emit_wmma_gemm_llvmir(dtype, arch=arch)
    r = llc_object(ir, arch=arch)
    assert r.status == "ok", r.detail
    assert r.is_amdgpu_elf
    assert r.n_bytes > 0


# ── operand layout: lane replication (ISA 7.9) + nt contiguous loads ──

@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_gemm_layout_emit_and_validate(dtype):
    ir = emit_wmma_gemm_layout_llvmir(dtype, arch="gfx1151")
    v = validate_wmma_gemm_layout_structure(ir, dtype=dtype, arch="gfx1151")
    assert v.ok, v.reasons


def test_gemm_layout_has_lane_replication():
    ir = emit_wmma_gemm_layout_llvmir("f16")
    # wave32 lanes 0-15 replicated into 16-31 via (lane & 15).
    assert "and i32 %lane, 15" in ir
    assert "%rowK = mul i64 %row" in ir   # per-lane row base (nt contiguous layout)


def test_gemm_layout_validator_catches_dropped_replication():
    ir = emit_wmma_gemm_layout_llvmir("f16").replace("and i32 %lane, 15", "add i32 %lane, 0")
    v = validate_wmma_gemm_layout_structure(ir, dtype="f16")
    assert not v.ok
    assert any("lane-replication" in r for r in v.reasons)


@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
@pytest.mark.parametrize("arch", ["gfx1151", "gfx1100"])
@pytest.mark.parametrize("dtype,want", [
    ("f16", "v_wmma_f32_16x16x16_f16"),
    ("bf16", "v_wmma_f32_16x16x16_bf16"),
])
def test_rung3_layout_gemm_emits_replication_in_amdgcn(dtype, want, arch):
    """The operand-layout GEMM lowers to AMDGCN carrying BOTH a real v_wmma_* and
    the lane-replication mask (`v_and_b32 _, 15, _`) — the ISA 7.9 requirement,
    machine-confirmed on this host."""
    ir = emit_wmma_gemm_layout_llvmir(dtype, arch=arch)
    r = llc_assemble(ir, arch=arch)
    assert r.status == "ok", r.detail
    assert want in r.wmma_instruction
    assert "v_and_b32" in r.asm   # the (lane & 15) replication landed in the ISA


# ── D→C output element mapping: the grounded strided store (GPUOpen RDNA3 blog) ──

@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_gemm_store_emit_and_validate(dtype):
    ir = emit_wmma_gemm_store_llvmir(dtype, arch="gfx1151")
    v = validate_wmma_gemm_store_structure(ir, dtype=dtype, arch="gfx1151")
    assert v.ok, v.reasons


def test_gemm_store_has_grounded_output_mapping():
    ir = emit_wmma_gemm_store_llvmir("bf16")
    # D[2*ele + lane/16][lane%16]: col = lane & 15, row_base = lane >> 4.
    assert "and i32 %lane, 15" in ir          # output column
    assert "lshr i32 %lane, 4" in ir          # output row half (lane / 16)
    # 8 strided scalar stores (one per accumulator register), not a vector store.
    assert ir.count("store float %e") == 8
    assert "store <8 x float>" not in ir
    assert "extractelement <8 x float>" in ir


def test_gemm_store_has_column_major_a_load():
    # GPUOpen RDNA3 blog: A is column-major, a_frag[ele] = a[16*lane + ele].
    # Generalized over K-tiles: col = k0 + lane, base = col * 16 (leading dim = 16 rows).
    ir = emit_wmma_gemm_store_llvmir("f16")
    assert "%kcol = add i32 %k, %lane16" in ir   # column index = k0 + lane
    assert "mul i64 %kcol64, 16" in ir           # column-major leading dim = 16 rows
    # the old row-major A addressing (shared %aidx for A and B) is gone.
    assert "%aidx = add i64 %rowK" not in ir
    # B stays nt-contiguous (pre-transposed), a distinct base from A.
    assert "%bidx = add i64 %rowK, %k64" in ir


def test_gemm_store_validator_catches_rowmajor_a_regression():
    # Reverting A to the row-major shared-index load must fail validation.
    ir = emit_wmma_gemm_store_llvmir("f16").replace(
        "%kcol = add i32 %k, %lane16", "%kcol = add i32 %k, 0")
    v = validate_wmma_gemm_store_structure(ir, dtype="f16")
    assert not v.ok
    assert any("column-major A column index" in r for r in v.reasons)


def test_gemm_store_validator_catches_missing_rowbase():
    ir = emit_wmma_gemm_store_llvmir("f16").replace("lshr i32 %lane, 4", "add i32 %lane, 0")
    v = validate_wmma_gemm_store_structure(ir, dtype="f16")
    assert not v.ok
    assert any("row-base decomposition" in r for r in v.reasons)


def test_gemm_store_validator_catches_dropped_store():
    ir = emit_wmma_gemm_store_llvmir("f16").replace(
        "store float %e7", "; dropped store %e7")
    v = validate_wmma_gemm_store_structure(ir, dtype="f16")
    assert not v.ok
    assert any("strided scalar D stores" in r for r in v.reasons)


def test_gemm_store_rejects_rdna4():
    with pytest.raises(ValueError):
        emit_wmma_gemm_store_llvmir("f16", arch="gfx1200")


@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
@pytest.mark.parametrize("arch", ["gfx1151", "gfx1100"])
@pytest.mark.parametrize("dtype,want", [
    ("f16", "v_wmma_f32_16x16x16_f16"),
    ("bf16", "v_wmma_f32_16x16x16_bf16"),
])
def test_rung3_store_gemm_lowers_with_strided_global_stores(dtype, want, arch):
    """The grounded-D-store GEMM lowers to AMDGCN with a real v_wmma_*, the lane
    replication, AND strided per-register global stores — the D->C output mapping,
    machine-confirmed on this host."""
    ir = emit_wmma_gemm_store_llvmir(dtype, arch=arch)
    r = llc_assemble(ir, arch=arch)
    assert r.status == "ok", r.detail
    assert want in r.wmma_instruction
    assert "v_and_b32" in r.asm        # lane replication + column mask
    assert "global_store" in r.asm     # the strided D stores landed


# ── threadgroup tiling: LDS-staged MF×NF fragment grid + double barrier ──

@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_gemm_threadgroup_emit_and_validate(dtype):
    ir = emit_wmma_gemm_threadgroup_llvmir(dtype, mf=2, nf=2, arch="gfx1151")
    v = validate_wmma_gemm_threadgroup_structure(ir, dtype=dtype, mf=2, nf=2, arch="gfx1151")
    assert v.ok, v.reasons


@pytest.mark.parametrize("mf,nf", [(1, 1), (2, 2), (1, 4), (3, 2)])
def test_gemm_threadgroup_fragment_grid_counts(mf, nf):
    ir = emit_wmma_gemm_threadgroup_llvmir("f16", mf=mf, nf=nf)
    nfrag = mf * nf
    assert ir.count("phi <8 x float>") == nfrag                     # one acc per fragment
    assert ir.count("@llvm.amdgcn.wmma.f32.16x16x16.f16(") == nfrag + 1  # +1 declare
    assert ir.count("store float %fe") == nfrag * 8                 # grounded store per frag
    v = validate_wmma_gemm_threadgroup_structure(ir, dtype="f16", mf=mf, nf=nf)
    assert v.ok, v.reasons


def test_gemm_threadgroup_has_lds_and_double_barrier():
    ir = emit_wmma_gemm_threadgroup_llvmir("bf16", mf=2, nf=2)
    assert "addrspace(3) global" in ir                  # LDS staging
    # two barriers in the loop body (stage-complete + LDS-reuse guard) + 1 declare.
    assert ir.count("@llvm.amdgcn.s.barrier()") == 3
    assert "mul i64 %kcol64, 16" in ir                  # column-major A base reused


def test_gemm_threadgroup_validator_catches_missing_barrier():
    ir = emit_wmma_gemm_threadgroup_llvmir("f16", mf=2, nf=2).replace(
        "  call void @llvm.amdgcn.s.barrier()          ; guard LDS before next K-step overwrites it\n",
        "", 1)
    v = validate_wmma_gemm_threadgroup_structure(ir, dtype="f16", mf=2, nf=2)
    assert not v.ok
    assert any("s.barrier" in r for r in v.reasons)


def test_gemm_threadgroup_rejects_rdna4():
    with pytest.raises(ValueError):
        emit_wmma_gemm_threadgroup_llvmir("f16", arch="gfx1200")


@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
@pytest.mark.parametrize("dtype,want", [
    ("f16", "v_wmma_f32_16x16x16_f16"),
    ("bf16", "v_wmma_f32_16x16x16_bf16"),
])
def test_rung3_threadgroup_lowers_with_lds_and_barrier(dtype, want):
    """The threadgroup-tiled GEMM lowers to AMDGCN with the MF×NF v_wmma_* grid,
    a real s_barrier, and LDS ds_store/ds_load — machine-confirmed on this host."""
    ir = emit_wmma_gemm_threadgroup_llvmir(dtype, mf=2, nf=2, arch="gfx1151")
    r = llc_assemble(ir, arch="gfx1151")
    assert r.status == "ok", r.detail
    assert want in r.wmma_instruction
    assert r.asm.count("v_wmma_f32_16x16x16") == 4   # 2x2 fragment grid
    assert "s_barrier" in r.asm
    assert "ds_store" in r.asm and "ds_load" in r.asm


# ── §7.9.1 WMMA scheduling hazard: grounded + llc-verified both ways ──

@pytest.mark.parametrize("hazard", [False, True])
def test_dependent_chain_emits(hazard):
    ir = emit_dependent_wmma_chain_llvmir("f16", hazard=hazard, depth=3)
    assert ir.count("@llvm.amdgcn.wmma.f32.16x16x16.f16(") == 3 + 1   # 3 calls + declare
    if hazard:
        assert "bitcast <8 x float>" in ir   # SrcA reads the prior WMMA destination
    else:
        assert "bitcast <8 x float>" not in ir


def test_dependent_chain_requires_depth_2():
    with pytest.raises(ValueError):
        emit_dependent_wmma_chain_llvmir("f16", depth=1)


def test_dependent_chain_rejects_rdna4():
    with pytest.raises(ValueError):
        emit_dependent_wmma_chain_llvmir("f16", arch="gfx1200")


@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
def test_rung3_accumulation_chain_is_hazard_free():
    """§7.9.1: the in-place accumulation pattern (C/D feedback, independent SrcA/B)
    needs NO v_nop — llc schedules it with s_delay_alu. This is what the GEMMs emit,
    so they require no manual scheduling nop."""
    r = llc_assemble(emit_dependent_wmma_chain_llvmir("f16", hazard=False, depth=4))
    assert r.status == "ok", r.detail
    s = wmma_scheduling(r.asm)
    assert s.n_wmma == 4
    assert s.n_vnop == 0 and s.hazard_free
    assert s.n_sdelay >= 1   # scheduled with s_delay_alu, not v_nop


@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
def test_rung3_hazard_chain_triggers_vnop():
    """§7.9.1: when a WMMA's SrcA reads the prior WMMA's destination, llc's hazard
    recognizer inserts a mandatory v_nop between them."""
    r = llc_assemble(emit_dependent_wmma_chain_llvmir("f16", hazard=True, depth=3))
    assert r.status == "ok", r.detail
    s = wmma_scheduling(r.asm)
    assert s.n_wmma == 3
    assert s.n_vnop >= 1 and not s.hazard_free


@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_rung3_threadgroup_gemm_is_hazard_free(dtype):
    """The real threadgroup GEMM is §7.9.1 hazard-free by construction — its
    fragment grid reads independent LDS-loaded SrcA/B + per-fragment accumulator
    PHIs, never a prior WMMA destination, so llc inserts no v_nop."""
    r = llc_assemble(emit_wmma_gemm_threadgroup_llvmir(dtype, mf=2, nf=2))
    assert r.status == "ok", r.detail
    s = wmma_scheduling(r.asm)
    assert s.n_wmma == 4 and s.hazard_free


# ── RDNA 4 (gfx1200/1201): denser fragments + FP8 unlock, plain 3-arg ABI ──

@pytest.mark.parametrize("dtype", ["f16", "bf16", "fp8_e4m3", "fp8_e5m2"])
def test_rdna4_emit_and_validate(dtype):
    ir = emit_wmma_rdna4_llvmir(dtype, arch="gfx1200")
    v = validate_wmma_rdna4_structure(ir, dtype=dtype, arch="gfx1200")
    assert v.ok, v.reasons


def test_rdna4_uses_denser_8wide_fragments():
    # RDNA 4 drops the wave32 lane-replication -> A/B are <8 x ...>, not gfx11's <16 x ...>.
    ir = emit_wmma_rdna4_llvmir("f16")
    assert "<8 x half>" in ir
    assert "<16 x half>" not in ir
    bf = emit_wmma_rdna4_llvmir("bf16")
    assert "<8 x i16>" in bf and "<16 x i16>" not in bf


def test_rdna4_fp8_intrinsic_names():
    assert wmma_intrinsic_rdna4("fp8_e4m3") == "llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8"
    assert wmma_intrinsic_rdna4("fp8_e5m2") == "llvm.amdgcn.wmma.f32.16x16x16.bf8.bf8"
    assert wmma_intrinsic_rdna4("f16") == "llvm.amdgcn.wmma.f32.16x16x16.f16"


def test_rdna4_rejects_rdna3_and_gfx1250_arch():
    # gfx1151 is the RDNA3 emitter; gfx1250/1251 use the later mods/reuse ABI.
    for bad in ("gfx1151", "gfx1100", "gfx1250", "gfx1251"):
        with pytest.raises(ValueError):
            emit_wmma_rdna4_llvmir("f16", arch=bad)


def test_rdna4_rejects_unknown_dtype():
    with pytest.raises(ValueError):
        wmma_intrinsic_rdna4("int4")


@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
@pytest.mark.parametrize("dtype,want", [
    ("f16", "v_wmma_f32_16x16x16_f16"),
    ("bf16", "v_wmma_f32_16x16x16_bf16"),
    ("fp8_e4m3", "v_wmma_f32_16x16x16_fp8_fp8"),
    ("fp8_e5m2", "v_wmma_f32_16x16x16_bf8_bf8"),
])
def test_rung3_rdna4_lowers_on_gfx1200(dtype, want):
    """RDNA 4 lowers on gfx1200 — machine-confirmed here (LLVM 22 AMDGPU). The FP8
    forms are the RDNA 4 unlock that gfx1151/RDNA 3.5 cannot select."""
    ir = emit_wmma_rdna4_llvmir(dtype, arch="gfx1200")
    r = llc_assemble(ir, arch="gfx1200")
    assert r.status == "ok", r.detail
    assert want in r.wmma_instruction


@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
def test_rung3_rdna4_fp8_is_not_selectable_on_rdna3():
    """Cross-check: the RDNA 4 FP8 intrinsic 'Cannot select' on gfx1151 — the FP8
    unlock is genuinely RDNA-4-only, not a naming nicety."""
    ir = emit_wmma_rdna4_llvmir("fp8_e4m3", arch="gfx1200")
    r = llc_assemble(ir, arch="gfx1151")   # try to lower the fp8 form on RDNA 3.5
    assert r.status == "failed"


# ── gfx1250/1251: the v2 mods/reuse ABI (K-doubled 16x16x32, native bfloat) ──

@pytest.mark.parametrize("arch", ["gfx1250", "gfx1251"])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_gfx1250_emit_and_validate(arch, dtype):
    ir = emit_wmma_gfx1250_llvmir(dtype, arch=arch)
    v = validate_wmma_gfx1250_structure(ir, dtype=dtype, arch=arch)
    assert v.ok, v.reasons


def test_gfx1250_intrinsic_names_are_k32():
    assert wmma_intrinsic_gfx1250("f16") == "llvm.amdgcn.wmma.f32.16x16x32.f16"
    assert wmma_intrinsic_gfx1250("bf16") == "llvm.amdgcn.wmma.f32.16x16x32.bf16"


def test_gfx1250_carries_v2_mods_reuse_abi():
    ir = emit_wmma_gfx1250_llvmir("f16")
    assert "i16 0," in ir            # C-mod immediate
    assert "i1 0, i1 0)" in ir       # the two operand-reuse flags
    assert "16x16x32" in ir          # K doubled vs gfx11/RDNA4's 16x16x16
    assert "<16 x half>" in ir       # wave32 K=32 -> 16 elements/lane


def test_gfx1250_bf16_is_native_bfloat_not_i16():
    # The grounded gfx1250 difference: bf16 is native <16 x bfloat>, NOT the
    # <_ x i16> bit-pattern gfx11/RDNA4 require.
    ir = emit_wmma_gfx1250_llvmir("bf16")
    assert "<16 x bfloat>" in ir
    assert "x i16>" not in ir


def test_gfx1250_rejects_other_arch_classes():
    for bad in ("gfx1151", "gfx1100", "gfx1200", "gfx1201"):
        with pytest.raises(ValueError):
            emit_wmma_gfx1250_llvmir("f16", arch=bad)


def test_gfx1250_fp8_is_scoped_out():
    # FP8 on gfx1250 uses a different class (16x16x64/128, ModsC) — a follow-on.
    with pytest.raises(ValueError):
        wmma_intrinsic_gfx1250("fp8_e4m3")


def test_gfx1250_validator_catches_dropped_reuse_flags():
    ir = emit_wmma_gfx1250_llvmir("f16").replace("i1 0, i1 0)", "i1 0)")
    v = validate_wmma_gfx1250_structure(ir, dtype="f16")
    assert not v.ok
    assert any("reuse" in r for r in v.reasons)


@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
@pytest.mark.parametrize("arch", ["gfx1250", "gfx1251"])
@pytest.mark.parametrize("dtype,want", [
    ("f16", "v_wmma_f32_16x16x32_f16"),
    ("bf16", "v_wmma_f32_16x16x32_bf16"),
])
def test_rung3_gfx1250_lowers_with_k32_wmma(arch, dtype, want):
    """gfx1250/1251 lower the v2 16x16x32 WMMA — machine-confirmed here. The K=32
    mnemonic distinguishes it from the gfx11/RDNA4 16x16x16 forms."""
    ir = emit_wmma_gfx1250_llvmir(dtype, arch=arch)
    r = llc_assemble(ir, arch=arch)
    assert r.status == "ok", r.detail
    assert want in r.wmma_instruction


@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
def test_rung3_gfx1250_k32_not_selectable_on_rdna4():
    """Cross-check: the gfx1250 16x16x32 v2 intrinsic 'Cannot select' on gfx1200 —
    the K-doubled mods/reuse ABI is genuinely gfx1250-class, not RDNA 4."""
    ir = emit_wmma_gfx1250_llvmir("f16", arch="gfx1250")
    r = llc_assemble(ir, arch="gfx1200")
    assert r.status == "failed"


# ── review-hardening guards (corner cases / glass jaws) ──

def test_threadgroup_lds_slots_are_non_overlapping():
    # Each (block, lane) fragment must own a 256-strided slot (16x16), not the old
    # 2-index <16 x elem> GEP that strided blocks by 1 element (overlapping).
    ir = emit_wmma_gemm_threadgroup_llvmir("f16", mf=2, nf=2)
    assert "%lane_x16 = mul i32 %lane16, 16" in ir       # per-lane fragment offset
    assert "add i32 %lane_x16, 256" in ir                # block 1 base = 256
    assert "i32 %lane16, i32 1\n" not in ir              # the old overlapping GEP is gone


def test_threadgroup_rejects_oversized_fragment_grid():
    with pytest.raises(ValueError, match="single-wave cap"):
        emit_wmma_gemm_threadgroup_llvmir("f16", mf=8, nf=8)   # 64 > 16


def test_emitters_reject_malformed_entry_name():
    for bad in ("has space", "1leading_digit", "bad-dash", ""):
        with pytest.raises(ValueError, match="entry name"):
            emit_wmma_llvmir("f16", entry=bad)
    with pytest.raises(ValueError, match="entry name"):
        emit_wmma_gemm_threadgroup_llvmir("f16", entry="no good")


def test_llc_assemble_rejects_malformed_arch():
    from tessera.compiler.rocdl_emit import llc_assemble
    with pytest.raises(ValueError, match="invalid arch"):
        llc_assemble(emit_wmma_llvmir("f16"), arch="x86; rm -rf")


def test_rdna4_validator_now_catches_gfx1250_operands():
    # Regression: the old `"i16," in ...` heuristic never matched the real `i16 0,`
    # operand. Splice a gfx1250 reuse-flag tail into an RDNA4 emit -> must be caught.
    ir = emit_wmma_rdna4_llvmir("f16").replace(
        "%a, <8 x float> zeroinitializer)", "%a, <8 x float> zeroinitializer, i1 0, i1 0)")
    v = validate_wmma_rdna4_structure(ir, dtype="f16")
    assert not v.ok
    assert any("plain 3-arg ABI" in r for r in v.reasons)
