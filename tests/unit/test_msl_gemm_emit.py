"""Apple rung-2.5 — MSL simdgroup_matrix GEMM emit spike (host-free).

Mirrors tests/unit/test_ptx_emit.py (NVIDIA WGMMA). The structural rung runs
anywhere (no Metal toolchain); the metal-compile rung skip-cleans off a
Metal-capable host.
"""

from __future__ import annotations

import pytest

from tessera.compiler.msl_gemm_emit import (
    SIMDGROUP_FRAG,
    MslGemmShape,
    emit_simdgroup_gemm_msl,
    metal_compile,
    min_metal_std,
    validate_msl_gemm_structure,
)


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
def test_emit_and_validate_roundtrip(dtype):
    msl = emit_simdgroup_gemm_msl(dtype, 8, 8, 8)
    v = validate_msl_gemm_structure(msl, dtype=dtype)
    assert v.ok, v.reasons
    # The documented simdgroup_matrix sequence is present.
    assert "simdgroup_multiply_accumulate(acc, a, b, acc)" in msl
    assert "make_filled_simdgroup_matrix" in msl
    assert "#include <metal_simdgroup_matrix>" in msl


def test_accumulator_is_fp32_for_low_precision_inputs():
    msl = emit_simdgroup_gemm_msl("bf16", 8, 8, 8)
    # bf16 inputs, fp32 accumulator (the production / numeric_policy pattern).
    assert "simdgroup_matrix<bfloat, 8, 8> a, b" in msl
    assert "simdgroup_matrix<float, 8, 8> acc" in msl
    assert "device float*      C" in msl


def test_bf16_requires_metal_3_1():
    assert min_metal_std("bf16") == "metal3.1"   # bfloat is an MSL 3.1 type
    assert min_metal_std("f16") == "metal3.0"
    assert min_metal_std("f32") == "metal3.0"


@pytest.mark.parametrize("m,n,k,ok", [
    (8, 8, 8, True),
    (32, 32, 16, True),
    (16, 8, 24, True),
    (8, 8, 7, False),     # K not a multiple of 8
    (12, 8, 8, False),    # M not a multiple of 8
    (0, 8, 8, False),     # zero dim
])
def test_tile_shape_validity(m, n, k, ok):
    assert MslGemmShape(m, n, k).is_valid() is ok
    if ok:
        msl = emit_simdgroup_gemm_msl("f16", m, n, k)
        assert validate_msl_gemm_structure(
            msl, dtype="f16", shape=MslGemmShape(m, n, k)).ok
    else:
        with pytest.raises(ValueError):
            emit_simdgroup_gemm_msl("f16", m, n, k)


def test_fragment_size_is_8():
    assert SIMDGROUP_FRAG == 8


def test_validator_rejects_corrupted_msl():
    msl = emit_simdgroup_gemm_msl("f16", 8, 8, 8)
    broken = msl.replace("simdgroup_multiply_accumulate", "bogus_mma")
    v = validate_msl_gemm_structure(broken, dtype="f16")
    assert not v.ok
    assert any("simdgroup_multiply_accumulate" in r for r in v.reasons)


def test_validator_rejects_wrong_dtype_fragment():
    # Emitted for f16 but validated as bf16 → fragment dtype mismatch caught.
    msl = emit_simdgroup_gemm_msl("f16", 8, 8, 8)
    v = validate_msl_gemm_structure(msl, dtype="bf16")
    assert not v.ok
    assert any("bfloat" in r for r in v.reasons)


def test_unsupported_dtype_raises():
    with pytest.raises(ValueError):
        emit_simdgroup_gemm_msl("int8", 8, 8, 8)


def test_metal_compile_is_skip_clean_without_toolchain():
    # On this CommandLineTools-only arm64 Mac the offline `metal` compiler is
    # absent, so rung-3 must skip cleanly (never error), like ptxas.
    msl = emit_simdgroup_gemm_msl("f16", 8, 8, 8)
    r = metal_compile(msl, dtype="f16")
    assert r.status in ("skipped", "ok", "failed")
    # If skipped, it's because the toolchain is absent — not a crash.
    if r.status == "skipped":
        assert "not available" in r.detail
