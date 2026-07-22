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
    emit_steel_gemm_msl,
    metal_compile,
    min_metal_std,
    validate_msl_gemm_structure,
    validate_steel_gemm_structure,
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
        # The single-fragment emitter only accepts m==n==8; larger valid tiles are
        # the steel (multi-fragment) emitter's job.
        if m == n == 8:
            msl = emit_simdgroup_gemm_msl("f16", m, n, k)
        else:
            msl = emit_steel_gemm_msl("f16", m, n, k)
        assert validate_msl_gemm_structure(
            msl, dtype="f16", shape=MslGemmShape(m, n, k)).ok
    else:
        with pytest.raises(ValueError):
            emit_simdgroup_gemm_msl("f16", m, n, k)


def test_single_fragment_emitter_rejects_multi_fragment_tile():
    # m/n > 8 would silently compute only the top-left 8x8 -> rejected (use steel).
    for (m, n) in [(16, 8), (8, 16), (32, 32)]:
        with pytest.raises(ValueError, match="single-output-fragment"):
            emit_simdgroup_gemm_msl("f16", m, n, 8)
    # k > 8 is fine (the K-loop handles it).
    msl = emit_simdgroup_gemm_msl("f16", 8, 8, 32)
    assert validate_msl_gemm_structure(msl, dtype="f16").ok


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


# ── steel-structured emit (multi-fragment + threadgroup staging + edge masking) ──

@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
def test_steel_emit_and_validate(dtype):
    msl = emit_steel_gemm_msl(dtype, 32, 32, 16)
    v = validate_steel_gemm_structure(msl, dtype=dtype)
    assert v.ok, v.reasons


def test_steel_has_production_shape_features():
    msl = emit_steel_gemm_msl("bf16", 32, 32, 16)
    # 4x4 output fragments per threadgroup.
    assert "simdgroup_matrix<float, 8, 8> acc[4 * 4]" in msl
    # threadgroup staging + barrier.
    assert "threadgroup bfloat As[32 * 16]" in msl
    assert "threadgroup bfloat Bs[16 * 32]" in msl
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in msl
    # ragged-edge masking on the staged load (zero-pad out-of-bounds).
    assert "(m0 + r < M && k0 + c < K) ? A[(m0 + r) * K + (k0 + c)] : bfloat(0)" in msl
    # whole-fragment guarded store.
    assert "if (cr + F <= M && cc + F <= N)" in msl


def test_steel_fragment_count_scales_with_tile():
    # 64x32 tile -> 8x4 = 32 output fragments.
    msl = emit_steel_gemm_msl("f16", 64, 32, 16)
    assert "simdgroup_matrix<float, 8, 8> acc[8 * 4]" in msl


def test_steel_rejects_non_fragment_multiple_tile():
    with pytest.raises(ValueError):
        emit_steel_gemm_msl("f16", 20, 32, 16)   # BM not a multiple of 8


def test_steel_validator_catches_dropped_staging():
    msl = emit_steel_gemm_msl("f16", 32, 32, 16)
    broken = msl.replace("threadgroup_barrier(mem_flags::mem_threadgroup)", "/*dropped*/")
    v = validate_steel_gemm_structure(broken, dtype="f16")
    assert not v.ok
    assert any("barrier" in r for r in v.reasons)


# ── Metal-CI rung-3 lane: actually compile when a Metal toolchain is present ──

@pytest.mark.compiler_tool
@pytest.mark.compiler_apple
@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
def test_rung3_simdgroup_gemm_compiles_on_metal_host(dtype):
    """On a Metal-capable runner, the emitted minimal kernel must actually compile
    to AIR — the rung-3 gate that catches MSL the host-free validator can't."""
    from tests._support.apple import require_metal_compiler

    require_metal_compiler()
    msl = emit_simdgroup_gemm_msl(dtype, 8, 8, 8)
    r = metal_compile(msl, dtype=dtype)
    assert r.status == "ok", f"{dtype}: {r.detail}"


@pytest.mark.compiler_tool
@pytest.mark.compiler_apple
@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
def test_rung3_steel_gemm_compiles_on_metal_host(dtype):
    from tests._support.apple import require_metal_compiler

    require_metal_compiler()
    msl = emit_steel_gemm_msl(dtype, 32, 32, 16)
    r = metal_compile(msl, dtype=dtype)
    assert r.status == "ok", f"{dtype}: {r.detail}"


# ── B1 partial-edge store + B2 double-buffer staging (opt-in refinements) ──

def test_steel_default_path_unchanged():
    # The refinements are opt-in: default output keeps the whole-fragment store and
    # single-buffered staging (no scratch / no ping-pong).
    msl = emit_steel_gemm_msl("bf16", 32, 32, 16)
    assert "Cs[" not in msl and "As[2]" not in msl


def test_steel_partial_edge_store():
    msl = emit_steel_gemm_msl("f16", 32, 32, 16, partial_edge=True)
    v = validate_steel_gemm_structure(msl, dtype="f16", partial_edge=True)
    assert v.ok, v.reasons
    assert "threadgroup float Cs[8 * 8]" in msl          # 8x8 scratch
    assert "uint rows = min(F, M - cr), cols = min(F, N - cc)" in msl  # valid-element bounds
    assert "C[(cr + rr) * N + (cc + cl)] = Cs[rr * F + cl]" in msl     # cooperative copy
    # full-fragment fast path is still present.
    assert "simdgroup_store(acc[im * 4u + in], C + cr * N + cc, N);" in msl


def test_steel_double_buffer_staging():
    msl = emit_steel_gemm_msl("f16", 32, 32, 16, double_buffer=True)
    v = validate_steel_gemm_structure(msl, dtype="f16", double_buffer=True)
    assert v.ok, v.reasons
    assert "threadgroup half As[2][32 * 16]" in msl      # ping-pong slots
    assert "uint buf = 0u" in msl                        # prologue index
    assert "uint nbuf = buf ^ 1u" in msl                 # alternate slot
    # double-buffer drops to ONE barrier per K-step (prologue + 1/iter) vs two.
    single = emit_steel_gemm_msl("f16", 32, 32, 16)
    assert msl.count("threadgroup_barrier") < single.count("threadgroup_barrier") * 2


def test_steel_partial_edge_branch_is_threadgroup_uniform():
    # The full/edge test must be keyed on tgid + uniform loop counters (not per-thread
    # data) so the scratch barriers are hit uniformly — never in divergent control flow.
    msl = emit_steel_gemm_msl("bf16", 16, 24, 16, partial_edge=True)
    assert "if (cr + F <= M && cc + F <= N) {" in msl    # uniform branch (m0/n0/im/in)


def test_steel_refinements_compose():
    msl = emit_steel_gemm_msl("bf16", 16, 24, 16, partial_edge=True, double_buffer=True)
    v = validate_steel_gemm_structure(
        msl, dtype="bf16", partial_edge=True, double_buffer=True)
    assert v.ok, v.reasons


def test_steel_validator_catches_missing_partial_scratch():
    msl = emit_steel_gemm_msl("f16", 32, 32, 16, partial_edge=True).replace(
        "threadgroup float Cs[8 * 8];", "/* no scratch */")
    v = validate_steel_gemm_structure(msl, dtype="f16", partial_edge=True)
    assert not v.ok
    assert any("partial-edge" in r for r in v.reasons)


@pytest.mark.compiler_tool
@pytest.mark.compiler_apple
@pytest.mark.parametrize("partial_edge,double_buffer", [(True, False), (False, True), (True, True)])
def test_rung3_steel_refinements_compile_on_metal_host(partial_edge, double_buffer):
    """B3: the B1/B2 refinements compile to AIR on a Metal-capable runner — the
    real verification for the structures the host-free validator can only token-check."""
    from tests._support.apple import require_metal_compiler

    require_metal_compiler()
    msl = emit_steel_gemm_msl("f16", 32, 32, 16,
                              partial_edge=partial_edge, double_buffer=double_buffer)
    r = metal_compile(msl, dtype="f16")
    assert r.status == "ok", r.detail
