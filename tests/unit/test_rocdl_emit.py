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
    emit_wmma_llvmir,
    llc_assemble,
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
