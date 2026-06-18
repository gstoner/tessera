"""Tests for A6 (arch-keyed FP8 FNUZ/OCP semantics + hipBLASLt scale modes) and
A7 (batched-vs-grouped GEMM dispatch classification).

A6's correctness point: the SAME canonical fp8 dtype is different bits on
gfx942 (FNUZ) vs gfx950 (OCP). A "complete FP8 kernel" claim is arch-ambiguous
without this flag.
"""

from __future__ import annotations

import pytest

from tessera.compiler.grouped_layout import (
    HIPBLASLT_SCALE_MODES,
    classify_gemm_dispatch,
    scale_mode_to_layout,
)
from tessera.compiler.rocm_target import (
    AMDArch,
    ROCmTargetProfile,
    TesseraROCmTargetError,
    fp8_dtype_flavor,
    fp8_semantics,
)


# ── A6: FP8 semantics per arch ──────────────────────────────────────────────

@pytest.mark.parametrize("arch,expected", [
    (AMDArch.GFX_90A, "none"),
    (AMDArch.GFX_940, "fnuz"),
    (AMDArch.GFX_942, "fnuz"),
    (AMDArch.GFX_950, "ocp"),
    (AMDArch.GFX_1100, "none"),
    (AMDArch.GFX_1151, "none"),
    (AMDArch.GFX_1200, "ocp"),
    (AMDArch.GFX_1250, "ocp"),
    (AMDArch.GFX_1251, "ocp"),
])
def test_fp8_semantics_per_arch(arch, expected):
    assert fp8_semantics(arch) == expected


def test_cdna3_vs_cdna4_fp8_differ():
    # The load-bearing correctness distinction: same dtype, different bits.
    assert fp8_semantics(AMDArch.GFX_942) != fp8_semantics(AMDArch.GFX_950)


def test_fp8_flavor_fnuz_on_cdna3():
    assert fp8_dtype_flavor(AMDArch.GFX_942, "fp8_e4m3") == "e4m3fnuz"
    assert fp8_dtype_flavor(AMDArch.GFX_942, "fp8_e5m2") == "e5m2fnuz"


def test_fp8_flavor_ocp_on_cdna4():
    assert fp8_dtype_flavor(AMDArch.GFX_950, "fp8_e4m3") == "e4m3"
    assert fp8_dtype_flavor(AMDArch.GFX_950, "fp8_e5m2") == "e5m2"


def test_fp8_flavor_ocp_on_rdna4():
    assert fp8_dtype_flavor(AMDArch.GFX_1200, "fp8_e4m3") == "e4m3"


def test_fp8_flavor_rejects_arch_without_fp8():
    with pytest.raises(TesseraROCmTargetError, match="no FP8 matrix path"):
        fp8_dtype_flavor(AMDArch.GFX_1151, "fp8_e4m3")


def test_fp8_flavor_rejects_non_fp8_dtype():
    with pytest.raises(TesseraROCmTargetError, match="not an fp8 dtype"):
        fp8_dtype_flavor(AMDArch.GFX_942, "bf16")


def test_profile_exposes_fp8_semantics():
    assert ROCmTargetProfile(arch=AMDArch.GFX_942).fp8_semantics == "fnuz"
    assert ROCmTargetProfile(arch=AMDArch.GFX_950).fp8_semantics == "ocp"


# ── A6: hipBLASLt scale-mode → ScaleLayout ──────────────────────────────────

def test_scale_mode_scalar():
    sl = scale_mode_to_layout("scalar_32f")
    assert sl.granularity == "per_tensor"
    assert sl.packing == "none"


def test_scale_mode_vec16_ue4m3():
    sl = scale_mode_to_layout("vec16_ue4m3")
    assert sl.granularity == "block"
    assert sl.block == (1, 16)
    assert sl.packing == "e4m3"
    assert sl.vector_size == 16


def test_scale_mode_vec32_ue8m0_mx():
    sl = scale_mode_to_layout("vec32_ue8m0")
    assert sl.block == (1, 32)
    assert sl.packing == "ue8m0"


def test_scale_mode_blk128():
    sl = scale_mode_to_layout("blk128x128_32f")
    assert sl.granularity == "block"
    assert sl.block == (128, 128)
    assert sl.alignment == 128


def test_all_scale_modes_resolve():
    for mode in HIPBLASLT_SCALE_MODES:
        assert scale_mode_to_layout(mode) is not None


def test_scale_mode_unknown_rejected():
    with pytest.raises(ValueError, match="unknown hipBLASLt scale mode"):
        scale_mode_to_layout("vec64_fp16")


# ── A7: batched vs grouped GEMM dispatch ────────────────────────────────────

def test_batched_gemm_is_uniform_host_known():
    c = classify_gemm_dispatch("batched_gemm")
    assert c.family == "batched"
    assert c.uniform_shape is True
    assert c.device_resident_args is False
    assert c.one_launch is True


def test_bmm_classifies_as_batched():
    assert classify_gemm_dispatch("bmm").family == "batched"


def test_grouped_gemm_is_device_resident():
    c = classify_gemm_dispatch("grouped_gemm")
    assert c.family == "grouped"
    assert c.uniform_shape is False
    assert c.device_resident_args is True  # per-expert sizes resolve on-device


def test_dequant_grouped_gemm_is_grouped():
    assert classify_gemm_dispatch("dequant_grouped_gemm").family == "grouped"


def test_plain_matmul_is_single():
    c = classify_gemm_dispatch("matmul")
    assert c.family == "single"
    assert c.device_resident_args is False


def test_batched_and_grouped_are_distinct():
    b = classify_gemm_dispatch("batched_gemm")
    g = classify_gemm_dispatch("grouped_gemm")
    assert b.family != g.family
    assert b.uniform_shape != g.uniform_shape
    assert b.device_resident_args != g.device_resident_args


def test_non_gemm_op_rejected():
    with pytest.raises(ValueError, match="not a GEMM-family op"):
        classify_gemm_dispatch("softmax")


def test_dispatch_metadata_dict():
    md = classify_gemm_dispatch("grouped_gemm").as_metadata_dict()
    assert md == {
        "family": "grouped",
        "uniform_shape": False,
        "device_resident_args": True,
        "one_launch": True,
    }
