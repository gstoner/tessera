"""Tests for the unified cooperative-matrix MMA descriptor (A1).

Locks the rocWMMA/Gluon "shape-is-the-anchor" contract: MFMA on CDNA, WMMA on
RDNA, operand layouts + k_width derived from the chosen instruction, and stable
diagnostics (never a silent fallback) when a dtype has no matrix-core path.
"""

from __future__ import annotations

import dataclasses

import pytest

from tessera.compiler.rocm_mma import (
    MmaDescriptor,
    MmaOperand,
    mma_for_matmul,
    select_mma,
)
from tessera.compiler.rocm_target import (
    AMDArch,
    ROCmTargetProfile,
    TesseraROCmTargetError,
)


# ── CDNA / MFMA path ────────────────────────────────────────────────────────

def test_cdna_bf16_selects_mfma_16x16x16():
    d = select_mma(AMDArch.GFX_942, "bf16")
    assert d.kind == "mfma"
    assert d.shape == (16, 16, 16)
    assert d.k_blocks == 1
    assert d.in_dtype == "bf16"
    assert d.acc_dtype == "fp32"
    assert d.operand_a.role == "matrix_a"
    assert d.operand_a.layout == "row_major"
    assert d.operand_b.layout == "col_major"
    assert d.accumulator.role == "accumulator"
    assert d.accumulator.dtype == "fp32"
    assert d.operand_a.k_width == 2  # 32 // 16
    assert d.accumulator.k_width == 1


def test_cdna_fp8_selects_k32_packing4():
    d = select_mma(AMDArch.GFX_942, "fp8_e4m3")
    assert d.kind == "mfma"
    assert d.shape == (16, 16, 32)
    assert d.operand_a.k_width == 4  # 32 // 8
    assert d.acc_dtype == "fp32"


def test_cdna4_fp4_selects_k64_packing8():
    d = select_mma(AMDArch.GFX_950, "fp4_e2m1")
    assert d.shape == (16, 16, 64)
    assert d.operand_a.k_width == 8  # 32 // 4


def test_cdna_fp32_routes_through_xf32_lane():
    d = select_mma(AMDArch.GFX_942, "fp32")
    assert d.kind == "mfma"
    assert d.shape == (16, 16, 8)  # xf32 K=8
    assert d.operand_a.k_width == 1
    assert d.acc_dtype == "fp32"


def test_cdna_int8_accumulates_in_int32():
    d = select_mma(AMDArch.GFX_942, "int8")
    assert d.shape == (16, 16, 16)
    assert d.acc_dtype == "int32"
    assert d.accumulator.dtype == "int32"
    assert d.operand_a.k_width == 4  # 32 // 8


def test_cdna2_has_no_fp8_mfma():
    with pytest.raises(TesseraROCmTargetError, match="mfma_f8"):
        select_mma(AMDArch.GFX_90A, "fp8_e4m3")


def test_cdna3_has_no_fp4_mfma():
    with pytest.raises(TesseraROCmTargetError, match="mfma_f4"):
        select_mma(AMDArch.GFX_942, "fp4_e2m1")


# ── RDNA / WMMA path ────────────────────────────────────────────────────────

def test_rdna3_fp16_selects_wmma_16x16x16():
    d = select_mma(AMDArch.GFX_1100, "fp16")
    assert d.kind == "wmma"
    assert d.shape == (16, 16, 16)
    assert d.k_blocks == 1
    assert d.acc_dtype == "fp32"
    assert d.operand_a.k_width == 2


def test_rdna35_strix_halo_fp16_wmma():
    d = select_mma(AMDArch.GFX_1151, "bf16")
    assert d.kind == "wmma"
    assert d.shape == (16, 16, 16)


def test_rdna35_has_no_fp8_wmma():
    # The load-bearing distinction from gfx1200.
    with pytest.raises(TesseraROCmTargetError, match="no FP8 WMMA"):
        select_mma(AMDArch.GFX_1151, "fp8_e4m3")


def test_rdna4_gfx1200_has_fp8_wmma():
    d = select_mma(AMDArch.GFX_1200, "fp8_e4m3")
    assert d.kind == "wmma"
    assert d.shape == (16, 16, 16)
    assert d.operand_a.k_width == 4


def test_rdna4_gfx1201_has_its_own_descriptor_identity():
    d = select_mma(AMDArch.GFX_1201, "bf16")
    assert d.arch is AMDArch.GFX_1201
    assert d.kind == "wmma"
    assert d.shape == (16, 16, 16)
    assert d.fragment_layout.family.value == "rdna4_wmma"


def test_gfx1250_doubles_k_for_f16():
    d = select_mma(AMDArch.GFX_1250, "fp16")
    assert d.shape == (16, 16, 32)  # v2 ABI K-doubled


def test_gfx1250_fp8_k64():
    d = select_mma(AMDArch.GFX_1251, "fp8_e5m2")
    assert d.shape == (16, 16, 64)


# ── prefer_shape + dtype gating ─────────────────────────────────────────────

def test_prefer_shape_picks_legal_alternative():
    d = select_mma(AMDArch.GFX_942, "bf16", prefer_shape=(32, 32, 8))
    assert d.shape == (32, 32, 8)
    assert d.k_blocks == 1


def test_prefer_shape_rejects_illegal():
    with pytest.raises(TesseraROCmTargetError, match="not a legal"):
        select_mma(AMDArch.GFX_942, "bf16", prefer_shape=(99, 99, 99))


def test_unknown_dtype_rejected():
    with pytest.raises(TesseraROCmTargetError, match="no cooperative-matrix path"):
        select_mma(AMDArch.GFX_942, "complex64")


def test_out_dtype_override():
    d = select_mma(AMDArch.GFX_942, "fp16", out_dtype="fp32")
    assert d.acc_dtype == "fp32"


# ── mma_for_matmul convenience ──────────────────────────────────────────────

def test_mma_for_matmul_cdna_fp32_xf32():
    prof = ROCmTargetProfile(arch=AMDArch.GFX_942)
    d = mma_for_matmul(prof, "fp32")
    assert d.kind == "mfma"
    assert d.shape == (16, 16, 8)


def test_mma_for_matmul_rdna_fp32_rejected():
    prof = ROCmTargetProfile(arch=AMDArch.GFX_1151)
    with pytest.raises(TesseraROCmTargetError, match="no WMMA path on RDNA"):
        mma_for_matmul(prof, "fp32")


# ── intrinsic + metadata ────────────────────────────────────────────────────

def test_intrinsic_string_format():
    assert select_mma(AMDArch.GFX_942, "bf16").intrinsic == "v_mfma_f32_16x16x16_bf16"
    assert select_mma(AMDArch.GFX_1100, "fp16").intrinsic == "v_wmma_f32_16x16x16_f16"
    assert select_mma(AMDArch.GFX_942, "int8").intrinsic == "v_mfma_i32_16x16x16_i8"
    assert select_mma(AMDArch.GFX_942, "fp8_e5m2").intrinsic == "v_mfma_f32_16x16x32_bf8"


def test_descriptor_metadata_roundtrip():
    d = select_mma(AMDArch.GFX_942, "bf16")
    md = d.as_metadata_dict()
    assert md["arch"] == "GFX_942"
    assert md["kind"] == "mfma"
    assert md["shape"] == [16, 16, 16]
    assert md["acc_dtype"] == "fp32"
    assert len(md["operands"]) == 3
    assert md["operands"][0]["role"] == "matrix_a"
    assert md["intrinsic"] == "v_mfma_f32_16x16x16_bf16"


def test_operands_tuple_order():
    d = select_mma(AMDArch.GFX_942, "bf16")
    roles = [op.role for op in d.operands]
    assert roles == ["matrix_a", "matrix_b", "accumulator"]


def test_m_n_k_properties():
    d = select_mma(AMDArch.GFX_942, "fp8_e4m3")
    assert (d.m, d.n, d.k) == (16, 16, 32)


# ── MmaOperand validation ───────────────────────────────────────────────────

def test_operand_rejects_bad_role():
    with pytest.raises(TesseraROCmTargetError, match="role"):
        MmaOperand("matrix_c", "bf16", "row_major", 2)


def test_operand_rejects_bad_layout():
    with pytest.raises(TesseraROCmTargetError, match="layout"):
        MmaOperand("matrix_a", "bf16", "diagonal", 2)


def test_operand_rejects_bad_kwidth():
    with pytest.raises(TesseraROCmTargetError, match="k_width"):
        MmaOperand("matrix_a", "bf16", "row_major", 0)


def test_descriptor_is_frozen():
    d = select_mma(AMDArch.GFX_942, "bf16")
    assert isinstance(d, MmaDescriptor)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.shape = (32, 32, 8)  # type: ignore[misc]
