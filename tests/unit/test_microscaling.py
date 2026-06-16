"""Low-precision (FP8/FP4/MX) quantization contract — the hardware-free
compiler-side surface (M3): scale layout as a first-class operand + the
metamorphic proof that survives the eventual real-silicon lowering.

FP8/FP4/MX tensor execution is toolchain-gated on this machine (the macOS 26.5
SDK doesn't expose the MTLTensorDataType FP8/FP4 cases the feature-set PDF
lists), so these are the *contract* + bit-accurate numpy reference. The
metamorphic invariants (idempotence, power-of-2 scale-invariance, microscaled
GEMM ≈ fp32 within the format grid) are backend-independent and are the proof a
future Metal lowering must preserve. See
docs/audit/backend/apple/APPLE_GPU_CODEGEN_PLAN.md (M3) and the
deepgemm_compiler_extraction memo (scale layout as IR operand type).
"""

from __future__ import annotations

import numpy as np
import pytest

ms = pytest.importorskip("tessera.compiler.microscaling")

_BLOCK_FORMATS = ["mxfp8_e4m3", "mxfp4_e2m1", "nvfp4"]
_ALL_FORMATS = ["fp8_e4m3", "fp8_e5m2", *_BLOCK_FORMATS]
_RNG = np.random.default_rng(20260615)


# ── scale layout as a first-class operand ──────────────────────────────────
def test_scale_layout_shapes():
    x = _RNG.standard_normal((4, 64)).astype(np.float32)
    # registry FP8 → 1x128 block over a 64-wide axis → 1 scale
    assert ms.quantize(x, "fp8_e4m3").scales.shape == (4, 1)
    # MX block-32 over a 64-wide axis → 2 scales
    assert ms.quantize(x, "mxfp8_e4m3").scales.shape == (4, 2)
    # NVFP4 block-16 → 4 scales
    assert ms.quantize(x, "nvfp4").scales.shape == (4, 4)
    # int8 → per-tensor scalar scale
    assert ms.quantize(x, "int8").scales.shape == ()


def test_scale_layout_axis_and_ragged_block():
    # 50 is not a multiple of 32 → ceil(50/32) = 2 scales (last block padded).
    x = _RNG.standard_normal((3, 50)).astype(np.float32)
    q = ms.quantize(x, "mxfp8_e4m3")
    assert q.scales.shape == (3, 2)
    assert q.codes.shape == (3, 50)            # codes stay the true shape


def test_microscaled_array_enforces_layout():
    x = _RNG.standard_normal((4, 64)).astype(np.float32)
    q = ms.quantize(x, "mxfp8_e4m3")
    with pytest.raises(ValueError, match="scale layout violation"):
        ms.MicroscaledArray(q.codes, q.scales[:, :1], q.format, q.shape)


def test_scale_dtype_is_the_format_scale_type():
    import ml_dtypes
    assert ms.quantize(_RNG.standard_normal((4, 64)).astype(np.float32),
                       "mxfp8_e4m3").scales.dtype == ml_dtypes.float8_e8m0fnu
    assert ms.quantize(_RNG.standard_normal((4, 64)).astype(np.float32),
                       "nvfp4").scales.dtype == ml_dtypes.float8_e4m3fn


# ── metamorphic invariants (exact — the proof) ──────────────────────────────
@pytest.mark.parametrize("fmt", _ALL_FORMATS)
def test_metamorphic_idempotence(fmt):
    # Re-quantizing an already-quantized value is exact: the value is on the grid.
    x = _RNG.standard_normal((8, 96)).astype(np.float32) * 2.5
    fq = ms.fake_quantize(x, fmt)
    np.testing.assert_array_equal(ms.fake_quantize(fq, fmt), fq)


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
def test_metamorphic_power_of_two_scale_invariance(fmt):
    # quant(x·2^k) dequantizes to 2^k·quant(x) exactly — power-of-2 just shifts
    # the (e8m0/fp8) block scale, codes unchanged.
    x = _RNG.standard_normal((8, 96)).astype(np.float32)
    for k in (2.0, -3.0):
        lhs = ms.dequantize(ms.quantize(x * (2.0 ** k), fmt))
        rhs = (2.0 ** k) * ms.dequantize(ms.quantize(x, fmt))
        np.testing.assert_array_equal(lhs, rhs)


# ── round-trip + GEMM within the format's error grid ────────────────────────
_RTOL = {"fp8_e4m3": 0.10, "fp8_e5m2": 0.16, "mxfp8_e4m3": 0.12,
         "mxfp4_e2m1": 0.30, "nvfp4": 0.30}


@pytest.mark.parametrize("fmt", _ALL_FORMATS)
def test_round_trip_within_format_grid(fmt):
    x = _RNG.standard_normal((8, 128)).astype(np.float32) * 3.0
    fq = ms.fake_quantize(x, fmt)
    rel = np.abs(fq - x).max() / np.abs(x).max()
    assert rel < _RTOL[fmt], f"{fmt}: {rel:.3f}"


@pytest.mark.parametrize("fmt,bound", [("mxfp8_e4m3", 0.10), ("nvfp4", 0.35)])
def test_mx_matmul_metamorphic_bound(fmt, bound):
    # Microscaled GEMM ≈ fp32 GEMM within the format grid — the invariant a real
    # FP8/FP4 tensor-core GEMM (fp32 accumulate) must preserve.
    A = _RNG.standard_normal((16, 64)).astype(np.float32)
    B = _RNG.standard_normal((64, 32)).astype(np.float32)
    ref = A @ B
    got = ms.mx_matmul(A, B, fmt)
    assert np.abs(got - ref).max() / np.abs(ref).max() < bound


# ── numeric_policy contract ─────────────────────────────────────────────────
def test_numeric_policy_carries_scale_layout():
    pol = ms.numeric_policy_for("nvfp4")
    assert pol["storage"] == "fp4_e2m1"
    assert pol["accum"] == "fp32"               # storage ≠ accumulator (Decision #15a)
    assert pol["scale_block_size"] == 16
    assert pol["scale_dtype"] == "fp8_e4m3"


def test_int8_is_per_tensor():
    pol = ms.numeric_policy_for("int8")
    assert pol["storage"] == "int8" and pol["scale_block_size"] == 0


# ── registry contract ↔ executable reference (single source of truth) ───────
@pytest.mark.parametrize("dtype", ["fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "nvfp4",
                                   "int8"])
def test_format_derived_from_registry_scale_layout(dtype):
    # microscaling derives its executable format from grouped_layout's declared
    # contract — block size + packing must agree, so the audit contract is
    # backed by this bit-accurate, metamorphic-proven reference.
    from tessera.compiler.grouped_layout import scale_layout_for
    sl = scale_layout_for(dtype)
    f = ms.format_for_dtype(dtype)
    assert f is not None and sl is not None
    if sl.granularity == "per_tensor":
        assert f.layout.block_size == 0
    else:
        assert f.layout.block_size == sl.block[1]    # cols share one scale
    # round-trip + metamorphic invariants hold for the registry-derived format.
    x = _RNG.standard_normal((4, 128)).astype(np.float32)
    fq = ms.fake_quantize(x, f)
    np.testing.assert_array_equal(ms.fake_quantize(fq, f), fq)   # idempotent


# ── int8 cross-path validation on REAL Metal silicon (DESIL-style) ──────────
def test_int8_quantized_matmul_agrees_numpy_vs_metal():
    # FP8/FP4 can't execute here (SDK gate), but int8 quantization + an f32
    # matmul of the dequantized operands CAN — so validate the contract on real
    # silicon: the int8-quantized GEMM computed via numpy must equal the same
    # dequantized operands matmul'd on the apple_gpu Metal lane.
    import tessera as ts
    A = _RNG.standard_normal((8, 32)).astype(np.float32)
    B = _RNG.standard_normal((32, 16)).astype(np.float32)
    a_dq = ms.fake_quantize(A, "int8").astype(np.float32)        # bit-accurate int8 round-trip
    b_dq = ms.fake_quantize(B, "int8").astype(np.float32)

    def mm(a, b):
        return ts.ops.matmul(a, b)
    metal = np.asarray(ts.jit(target="apple_gpu")(mm)(a_dq, b_dq))
    numpy_ref = a_dq @ b_dq
    # Same quantized operands, two execution paths → must agree (the contract
    # executes correctly on Metal). f32 matmul tolerance.
    np.testing.assert_allclose(metal, numpy_ref, rtol=1e-4, atol=1e-4)
