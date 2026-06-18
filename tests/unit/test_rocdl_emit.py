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
    emit_wmma_gemm_threadgroup_llvmir,
    emit_wmma_llvmir,
    llc_assemble,
    validate_wmma_gemm_layout_structure,
    validate_wmma_gemm_store_structure,
    validate_wmma_gemm_structure,
    validate_wmma_gemm_threadgroup_structure,
    validate_wmma_llvmir_structure,
    wmma_intrinsic,
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


@pytest.mark.skipif(not _llc_available(), reason="LLVM `llc` (AMDGPU backend) not found")
@pytest.mark.parametrize("dtype,want", [
    ("f16", "v_wmma_f32_16x16x16_f16"),
    ("bf16", "v_wmma_f32_16x16x16_bf16"),
])
def test_rung3_gemm_lowers_to_wmma_inside_a_loop(dtype, want):
    """The K-reduction GEMM lowers to AMDGCN with the v_wmma_* inside a real loop."""
    ir = emit_wmma_gemm_llvmir(dtype, arch="gfx1151")
    r = llc_assemble(ir, arch="gfx1151")
    assert r.status == "ok", r.detail
    assert want in r.wmma_instruction
    # the wmma is inside a loop body (a conditional branch / loop label exists).
    assert "s_cbranch" in r.asm or ".LBB" in r.asm


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
@pytest.mark.parametrize("dtype,want", [
    ("f16", "v_wmma_f32_16x16x16_f16"),
    ("bf16", "v_wmma_f32_16x16x16_bf16"),
])
def test_rung3_layout_gemm_emits_replication_in_amdgcn(dtype, want):
    """The operand-layout GEMM lowers to AMDGCN carrying BOTH a real v_wmma_* and
    the lane-replication mask (`v_and_b32 _, 15, _`) — the ISA 7.9 requirement,
    machine-confirmed on this host."""
    ir = emit_wmma_gemm_layout_llvmir(dtype, arch="gfx1151")
    r = llc_assemble(ir, arch="gfx1151")
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
@pytest.mark.parametrize("dtype,want", [
    ("f16", "v_wmma_f32_16x16x16_f16"),
    ("bf16", "v_wmma_f32_16x16x16_bf16"),
])
def test_rung3_store_gemm_lowers_with_strided_global_stores(dtype, want):
    """The grounded-D-store GEMM lowers to AMDGCN with a real v_wmma_*, the lane
    replication, AND strided per-register global stores — the D->C output mapping,
    machine-confirmed on this host."""
    ir = emit_wmma_gemm_store_llvmir(dtype, arch="gfx1151")
    r = llc_assemble(ir, arch="gfx1151")
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
